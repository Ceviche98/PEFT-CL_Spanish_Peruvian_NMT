#!/usr/bin/env python3
import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np

# CRITICAL VAST.AI FIX: Force HuggingFace cache to /workspace volume before hub imports
if os.path.exists("/workspace"):
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


import torch
import sacrebleu
from datasets import Dataset, concatenate_datasets, interleave_datasets
from peft import LoraConfig, TaskType, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq,
    NllbTokenizer,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
)
from transformers.trainer_utils import get_last_checkpoint

# =============================================================================
# DUAL CLASS-LEVEL PATCH: Fix transformers 4.5x M2M100 regression.
#
# Root cause: M2M100Model.forward pre-computes decoder_inputs_embeds AND then
# passes BOTH decoder_input_ids AND decoder_inputs_embeds positionally to the
# decoder. Because the args are positional, kwargs-based checks don't intercept.
#
# Fix: Patch BOTH M2M100Model AND M2M100Decoder to unconditionally convert
# any input_ids to embeddings and null the ids, so the decoder never sees both.
# =============================================================================
from transformers.models.m2m_100.modeling_m2m_100 import (
    M2M100Model as _M2M100Model,
    M2M100Decoder as _M2M100Decoder,
)

# --- Patch 1: M2M100Model.forward -------------------------------------------
_orig_m2m_model_forward = _M2M100Model.forward

def _fixed_m2m_model_forward(
    self, input_ids=None, attention_mask=None,
    decoder_input_ids=None, decoder_attention_mask=None,
    head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None,
    encoder_outputs=None, past_key_values=None,
    inputs_embeds=None, decoder_inputs_embeds=None,
    use_cache=None, output_attentions=None, output_hidden_states=None,
    return_dict=None, **kwargs
):
    # Always convert decoder_input_ids → embeddings, then null ids.
    # Handles: (a) only ids, (b) both provided. Prevents dual-pass into decoder.
    if decoder_input_ids is not None:
        if decoder_inputs_embeds is None:
            _scale = math.sqrt(self.config.d_model) if getattr(self.config, 'scale_embedding', False) else 1.0
            decoder_inputs_embeds = self.shared(decoder_input_ids) * _scale
        decoder_input_ids = None
    return _orig_m2m_model_forward(
        self, input_ids=input_ids, attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        head_mask=head_mask, decoder_head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        encoder_outputs=encoder_outputs, past_key_values=past_key_values,
        inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        **kwargs
    )

_M2M100Model.forward = _fixed_m2m_model_forward

# --- Patch 2: M2M100Decoder.forward (safety net) ----------------------------
# If M2M100Model internally re-populates decoder_input_ids from cache/positions
# (a 4.57.x behaviour), this second patch catches it at the last possible point.
_orig_m2m_decoder_forward = _M2M100Decoder.forward

def _fixed_m2m_decoder_forward(
    self, input_ids=None, attention_mask=None,
    encoder_hidden_states=None, encoder_attention_mask=None,
    head_mask=None, cross_attn_head_mask=None,
    past_key_values=None, inputs_embeds=None,
    use_cache=None, output_attentions=None,
    output_hidden_states=None, return_dict=None,
    **kwargs,
):
    # Always convert input_ids → embeddings and null ids.
    if input_ids is not None:
        if inputs_embeds is None:
            _scale = math.sqrt(self.config.d_model) if getattr(self.config, 'scale_embedding', False) else 1.0
            inputs_embeds = self.embed_tokens(input_ids) * _scale
        input_ids = None
    return _orig_m2m_decoder_forward(
        self,
        input_ids=input_ids, attention_mask=attention_mask,
        encoder_hidden_states=encoder_hidden_states,
        encoder_attention_mask=encoder_attention_mask,
        head_mask=head_mask, cross_attn_head_mask=cross_attn_head_mask,
        past_key_values=past_key_values, inputs_embeds=inputs_embeds,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        **kwargs,
    )

_M2M100Decoder.forward = _fixed_m2m_decoder_forward
# =============================================================================

# =============================================================================
# CUSTOM TRAINER: Pre-compute decoder embeddings in compute_loss so that
# M2M100ForConditionalGeneration.forward never receives decoder_input_ids at all.
# This bypasses all internal M2M100 routing and is the permanent reliable fix.
# =============================================================================
class NLLBTrainer(Trainer):
    """Trainer that bypasses the M2M100 transformers 4.5x decoder dual-input bug."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)

        if labels is not None:
            # Resolve config (handles plain model and PEFT-wrapped models)
            cfg = getattr(model, "config", None)
            if cfg is None:
                cfg = model.base_model.model.config

            pad_id   = cfg.pad_token_id
            start_id = cfg.decoder_start_token_id

            # Shift labels right: [start, t0, t1, ...] — same as shift_tokens_right
            safe = labels.clone()
            safe[safe == -100] = pad_id          # replace ignore-index with pad
            dec_ids = safe.new_zeros(safe.shape)
            dec_ids[:, 1:] = safe[:, :-1].clone()
            dec_ids[:, 0]  = start_id

            # Embed: use the model's own embedding + embed_scale
            emb = model.get_input_embeddings()   # shared embedding layer
            scale = math.sqrt(cfg.d_model) if getattr(cfg, "scale_embedding", False) else 1.0
            model_dtype = next(model.parameters()).dtype
            dec_embeds = emb(dec_ids).to(dtype=model_dtype) * scale

            # Call model with *only* decoder_inputs_embeds (no decoder_input_ids).
            # M2M100ForConditionalGeneration skips creating decoder_input_ids when
            # decoder_inputs_embeds is already provided, breaking the conflict chain.
            outputs = model(
                input_ids=inputs.get("input_ids"),
                attention_mask=inputs.get("attention_mask"),
                decoder_inputs_embeds=dec_embeds,
                labels=labels,
            )
            inputs["labels"] = labels   # restore for any downstream logging
        else:
            outputs = model(**inputs)

        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss
# =============================================================================


LANG_TO_NLLB = {
    "quy": "quy_Latn",
    "ayr": "ayr_Latn",
    "shp": "shp_Latn",
    "agr": "agr_Latn",
    "cni": "cni_Latn",
}
NEW_LANGUAGES = ["shp", "agr", "cni"]
BASE_LANGUAGES = ["quy", "ayr"]
ALL_LANGS = list(LANG_TO_NLLB.keys())
SRC_LANG = "spa_Latn"

# ---------------------------------------------------------------------------
# Data Loading & Sampling
# ---------------------------------------------------------------------------

def load_language_data(lang_code: str, data_dir: Path, tokenizer: NllbTokenizer, max_length: int = 256) -> Optional[Dataset]:
    es_path = data_dir / "train" / lang_code / "train.filtered.es" # Use filtered data
    lang_path = data_dir / "train" / lang_code / f"train.filtered.{lang_code}"
    
    if not es_path.exists() or not lang_path.exists():
        es_path = data_dir / "train" / lang_code / "train.es"
        lang_path = data_dir / "train" / lang_code / f"train.{lang_code}"
        if not es_path.exists() or not lang_path.exists():
            print(f"[SKIP] Data not found for {lang_code}")
            return None

    with open(es_path, encoding="utf-8") as f:
        src_lines = [l.strip() for l in f if l.strip()]
    with open(lang_path, encoding="utf-8") as f:
        tgt_lines = [l.strip() for l in f if l.strip()]

    min_len = min(len(src_lines), len(tgt_lines))
    src_lines, tgt_lines = src_lines[:min_len], tgt_lines[:min_len]
    tgt_nllb = LANG_TO_NLLB[lang_code]

    def tokenize(batch):
        tokenizer.src_lang = SRC_LANG
        tokenizer.tgt_lang = tgt_nllb # Let the tokenizer know the target routing
        
        # text_target handles the label tokenization and bos/eos placement natively
        model_inputs = tokenizer(
            batch["src"], 
            text_target=batch["tgt"], 
            truncation=True, 
            max_length=max_length, 
            padding=False
        )
        return model_inputs

    raw = Dataset.from_dict({"src": src_lines, "tgt": tgt_lines})
    tokenized = raw.map(tokenize, batched=True, batch_size=512, remove_columns=["src", "tgt"])
    return tokenized

def load_dev_raw(lang_code: str, data_dir: Path) -> Optional[tuple[list[str], list[str]]]:
    es_path = data_dir / "dev" / lang_code / "dev.es"
    lang_path = data_dir / "dev" / lang_code / f"dev.{lang_code}"
    if not es_path.exists() or not lang_path.exists(): return None
    with open(es_path, encoding="utf-8") as f:
        es_lines = [l.strip() for l in f if l.strip()]
    with open(lang_path, encoding="utf-8") as f:
        lang_lines = [l.strip() for l in f if l.strip()]
    return es_lines, lang_lines

# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

class MultiLangEvalCallback(TrainerCallback):
    def __init__(self, model, tokenizer, active_langs, data_dir, output_dir, patience=5, gen_batch_size=16, max_new_tokens=256):
        self.model = model
        self.tokenizer = tokenizer
        self.active_langs = active_langs
        self.output_dir = output_dir
        self.patience = patience
        self.gen_batch_size = gen_batch_size
        self.max_new_tokens = max_new_tokens
        
        self.log_csv_path = output_dir / "training_log.csv"
        self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.log_csv_path.exists():
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["step", "train_loss", "val_loss", "avg_chrf"] + [f"{lc}_chrf" for lc in active_langs])
                
        self.dev_data = {}
        for lc in active_langs:
            pair = load_dev_raw(lc, data_dir)
            if pair:
                self.dev_data[lc] = pair
                
        self.best_avg_chrf = -1.0
        self.no_improve_evals = 0

    def generate_translations(self, src_lines, tgt_nllb_token):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        forced_bos_id = self.tokenizer.convert_tokens_to_ids(tgt_nllb_token)
        translations = []
        for i in range(0, len(src_lines), self.gen_batch_size):
            batch = src_lines[i : i + self.gen_batch_size]
            self.tokenizer.src_lang = SRC_LANG
            inputs = self.tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
            with torch.no_grad():
                outputs = self.model.generate(**inputs, forced_bos_token_id=forced_bos_id, max_new_tokens=self.max_new_tokens, num_beams=1)
            translations.extend(self.tokenizer.batch_decode(outputs, skip_special_tokens=True))
        self.model.train()
        return translations

    def on_evaluate(self, args, state, control, **kwargs):
        step = state.global_step
        all_chrf = []
        all_comet = []
        lang_metrics = {}
        
        for lc, (src, ref) in self.dev_data.items():
            hyps = self.generate_translations(src, LANG_TO_NLLB[lc])
            chrf = sacrebleu.corpus_chrf(hyps, [ref], word_order=2).score
            all_chrf.append(chrf)
            lang_metrics[lc] = chrf

        avg_chrf = np.mean(all_chrf) if all_chrf else float("nan")
        
        train_loss = next((x["loss"] for x in reversed(state.log_history) if "loss" in x), float("nan"))
        val_loss = kwargs.get("metrics", {}).get("eval_loss", float("nan"))

        print(f"\nStep {step} Eval | Val Loss: {val_loss:.4f} | Avg chrF++: {avg_chrf:.2f}")

        with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
            row = [step, train_loss, val_loss, avg_chrf]
            row.extend([lang_metrics.get(lc, float("nan")) for lc in self.active_langs])
            csv.writer(f).writerow(row)

        if not math.isnan(avg_chrf) and avg_chrf > self.best_avg_chrf:
            self.best_avg_chrf = avg_chrf
            self.no_improve_evals = 0
            
            p = self.output_dir / "best_chrf_checkpoint"
            p.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(p))
            self.tokenizer.save_pretrained(str(p))
            
            # Explicitly save embedding layer to avoid PEFT discard of non-LoRA modules
            emb_layer = self.model.get_input_embeddings()
            torch.save(emb_layer.weight.data, p / "embeddings.pt")
            
            with open(p / "metrics.json", "w") as f:
                json.dump({"step": step, "avg_chrf": avg_chrf}, f)
            print("  ✓ Saved best chrF++ checkpoint (and explicit embeddings.pt).")
        else:
            self.no_improve_evals += 1

        if self.no_improve_evals >= self.patience:
            print(f"Early stopping triggered at step {step}!")
            control.should_training_stop = True

        return control

# ---------------------------------------------------------------------------
# Setup Methods
# ---------------------------------------------------------------------------

def initialize_embeddings(model, tokenizer):
    print("Initializing unstructured target language embeddings using (Quechua + Aymara)/2...")
    quy_id = tokenizer.convert_tokens_to_ids(LANG_TO_NLLB["quy"])
    ayr_id = tokenizer.convert_tokens_to_ids(LANG_TO_NLLB["ayr"])
    
    emb_layer = model.get_input_embeddings()
    
    with torch.no_grad():
        emb = emb_layer.weight
        avg_vec = (emb[quy_id] + emb[ayr_id]) / 2.0
        
        for lang in NEW_LANGUAGES:
            target_id = tokenizer.convert_tokens_to_ids(LANG_TO_NLLB[lang])
            emb[target_id] = avg_vec.clone()

def apply_embedding_masking(model, tokenizer, active_langs):
    print("Applying selective embedding gradient mask for target languages only...")
    target_ids = [tokenizer.convert_tokens_to_ids(LANG_TO_NLLB[lang]) for lang in active_langs]
    
    emb_layer = model.get_input_embeddings()
    emb_layer.weight.requires_grad = True # Make sure it gets gradients

    target_ids_tensor = torch.tensor(target_ids, dtype=torch.long, device=emb_layer.weight.device)
    
    def hook(grad):
        mask = torch.zeros_like(grad)
        mask[target_ids_tensor] = 1.0
        return grad * mask
        
    emb_layer.weight.register_hook(hook)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["fft", "lora", "dora", "qlora", "pissa", "galore"], required=True)
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lr", type=float, default=None,
                        help="Override the default per-method learning rate.")
    parser.add_argument("--max_steps", type=int, default=600000,
                        help="Total training steps (use with epoch-based scripts).")
    parser.add_argument("--experiment_name", type=str, default="experiment_A",
                        help="Top-level folder under models/ for this run group.")
    parser.add_argument("--patience", type=int, default=5,
                        help="Early-stopping patience (evals without chrF++ gain). "
                             "Set to a large number (e.g. 999) to disable early stopping.")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data_in"
    # Include rank in folder name for PEFT methods so different rank runs don't collide.
    # e.g. lora_r32, lora_r128, dora_r128, qlora_r128, pissa_r128; fft/galore unchanged.
    if args.method in ["lora", "dora", "qlora", "pissa"]:
        method_tag = f"{args.method}_r{args.lora_r}"
    else:
        method_tag = args.method

    # Learning rates: CLI override takes priority, then per-method defaults.
    if args.lr is not None:
        lr = args.lr
    elif args.method == "fft":
        lr = 5e-5
    elif args.method in ["galore", "pissa"]:
        lr = 1e-4
    else:
        lr = 3e-4

    # Format lr for folder name (e.g. lr5e-5, lr3e-4, lr1e-4)
    lr_tag = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")

    # Path: models/<experiment_name>/<method_tag>/lr<rate>/
    output_dir = project_root / "models" / args.experiment_name / method_tag / lr_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer_path = project_root / "models/v0/tokenizer"
    tokenizer = NllbTokenizer.from_pretrained(str(tokenizer_path))
    
    base_model_id = "facebook/nllb-200-1.3B"

    # Load WITHOUT device_map for non-QLoRA methods.
    # device_map installs accelerate dispatch hooks that intercept module.__call__,
    # bypassing .forward patches and corrupting kwarg routing across the M2M100 decoder.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading '{base_model_id}' under method '{args.method}'…")
    
    if args.method == "qlora":
        # bitsandbytes quantization requires device_map; use single-device map.
        bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, quantization_config=bnb_config, device_map={"" : device})
        model = prepare_model_for_kbit_training(model)
    else:
        # No device_map: plain load then .to(device). Avoids ALL accelerate hooks.
        model = AutoModelForSeq2SeqLM.from_pretrained(base_model_id, dtype=torch.bfloat16)
        model = model.to(device)
        
    model.resize_token_embeddings(len(tokenizer))

    if args.method not in ["fft", "galore"]:
        peft_config = LoraConfig(
            task_type=TaskType.SEQ_2_SEQ_LM,
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"],
            lora_dropout=0.05,
            bias="none",
            use_dora=(args.method == "dora"),
            init_lora_weights="pissa" if args.method == "pissa" else True
        )
        model = get_peft_model(model, peft_config)



    # Core customizations
    initialize_embeddings(model, tokenizer)
    apply_embedding_masking(model, tokenizer, ALL_LANGS)
    
    print("\nPreparing Datasets with Temperature T=0.7…")
    all_ds = []
    lengths = []
    for lc in ALL_LANGS:
        ds = load_language_data(lc, data_dir, tokenizer, max_length=256)
        if ds:
            all_ds.append(ds)
            lengths.append(len(ds))
            
    lengths = np.array(lengths)
    T = 0.7  # Temperature: T<1 flattens distribution, upsampling low-resource langs
    probs = (lengths ** T) / np.sum(lengths ** T)
    print(f"Sampling Probabilities: {dict(zip(ALL_LANGS, [round(p, 3) for p in probs]))}")
    
    # Compute how many times each dataset needs to repeat so none is exhausted
    # before max_steps. Worst case: smallest dataset (cni) sampled at its prob rate.
    #   total samples from lang_i = max_steps * eff_batch * prob_i
    #   repeats_needed_i = ceil(total_samples_i / len_i)
    eff_batch = 8 * 2  # per_device_batch * gradient_accumulation_steps
    samples_needed = args.max_steps * eff_batch * probs  # per language
    repeats_needed = np.ceil(samples_needed / lengths).astype(int)
    num_repeats = int(repeats_needed.max()) + 2   # +2 as safety margin
    print(f"Dataset repeats (max needed): {num_repeats}  (most constrained: {ALL_LANGS[repeats_needed.argmax()]})")
    train_ds = interleave_datasets(
        [d.to_iterable_dataset().repeat(num_repeats) for d in all_ds],
        probabilities=probs, seed=42
    )
    
    # We must construct an evaluation dataset for Trainer to compute val loss autonomously.
    print("\nPreparing Evaluation Datasets...")
    dev_datasets = []
    for lc in ALL_LANGS:
        es_lines, lang_lines = load_dev_raw(lc, data_dir) or ([], [])
        if es_lines:
            def tok(batch):
                tokenizer.src_lang = SRC_LANG
                tokenizer.tgt_lang = LANG_TO_NLLB[lc]
                return tokenizer(
                    batch["src"], 
                    text_target=batch["tgt"], 
                    truncation=True, 
                    max_length=256, 
                    padding=False
                )
            ds_dev = Dataset.from_dict({"src": es_lines, "tgt": lang_lines})
            ds_dev = ds_dev.map(tok, batched=True, batch_size=512, remove_columns=["src", "tgt"])
            dev_datasets.append(ds_dev)
    eval_ds = concatenate_datasets(dev_datasets) if dev_datasets else None

    # Optimizer
    optimizer_cls = "paged_adamw_8bit"
    if args.method == "galore":
        try:
            import galore_torch
            optimizer_cls = "galore_adamw_8bit" 
            print("Using GaLORE 8-bit optimizer. Make sure galore_torch is available in huggingface transformers context, or provide explicit target_modules in training args.")
            # Note: For full proper GaLORE, one might pass optim="galore_adamw_8bit_layerwise", optim_args="rank=128, update_proj_gap=200, scale=2.0"
        except ImportError:
            print("[ERROR] galore_torch not installed!")
            sys.exit(1)

    # Eval/save cadence: 1 eval per epoch, derived from actual dataset size.
    # steps_per_epoch = total_filtered_pairs / effective_batch
    # This is computed after dataset loading so we use the real filtered lengths.
    _steps_per_epoch = max(100, int(sum(lengths) / (8 * 2)))
    _eval_every  = _steps_per_epoch          # 1 eval per epoch
    _save_every  = _steps_per_epoch          # 1 checkpoint per epoch
    _log_every   = max(10, _steps_per_epoch // 30)  # ~30 log lines per epoch
    print(f"Steps per epoch: {_steps_per_epoch}  |  Eval every: {_eval_every} steps")

    training_args = TrainingArguments(
        output_dir=str(output_dir / "hf"),
        max_steps=args.max_steps,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2, # Effective batch = 16
        learning_rate=lr,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        eval_strategy="steps",
        eval_steps=_eval_every,
        logging_steps=_log_every,
        # Checkpointing strategy:
        #   - MultiLangEvalCallback saves the best chrF++ model at output_dir/best_chrf_checkpoint
        #   - Trainer saves the *latest* rolling checkpoint (for crash recovery)
        #   - save_total_limit=1 means ONLY the single most recent Trainer checkpoint is kept;
        #     the previous one is deleted automatically, preventing disk explosion.
        save_strategy="steps",
        save_steps=_save_every,
        save_total_limit=1,          # keep only the single latest Trainer checkpoint
        ignore_data_skip=True,       # CRITICAL: Resume instantly without spinning CPU for hours fast-forwarding data

        gradient_checkpointing=False,
        bf16=True,
        optim=optimizer_cls,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        weight_decay=0.0,  # CRITICAL: Must be 0.0 to prevent AdamW from erasing frozen tokens

    )

    if args.method == "galore":
        training_args.optim_args = "rank=128,update_proj_gap=200,scale=2.0"
        training_args.optim_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]



    # NLLBTrainer overrides compute_loss to pre-compute decoder embeddings,
    # bypassing the M2M100 transformers 4.5x decoder_input_ids conflict.
    trainer = NLLBTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[MultiLangEvalCallback(model, tokenizer, ALL_LANGS, data_dir, output_dir, patience=args.patience)]
    )

    last_checkpoint = get_last_checkpoint(str(output_dir / "hf"))
    if last_checkpoint is not None:
        print(f"\n[RESUME] Found existing checkpoint at {last_checkpoint}. Resuming training...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("\nStarting Benchmark Training from scratch…")
        trainer.train()

if __name__ == "__main__":
    main()
