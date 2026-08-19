#!/usr/bin/env python3
"""
Sequential LoRA (r=128) Continual Learning Script for NLLB-200-1.3B (Experiment B)
================================================================================
Trains a single LoRA adapter sequentially across the 5 Peruvian Indigenous languages
without data replay to measure catastrophic forgetting (FM), average accuracy (AA),
and backward transfer (BWT) using the Triangular Evaluation Matrix protocol.

Supports both experimental orders defined in Section 7.1 of Protocol:
  - order1 (Andino -> Amazónico / High to Low): quy -> ayr -> shp -> agr -> cni
  - order2 (Amazónico -> Andino / Low to High): cni -> agr -> shp -> ayr -> quy

Features & Parity with scripts/experiment_A/train_benchmarks.py:
  - Output directory structure matches traditional folders / Vast.ai workspace:
    models/experiment_B/lora_sequential_r128/<order>/lr<rate>/
  - Clean Vast.ai / local environment support: respects /workspace/.cache/huggingface
  - Language Token Training & Selective Embedding Masking:
    Properly calls `initialize_new_language_embeddings` and `apply_embedding_masking`
    so that only the active language tokens receive gradients during LoRA fine-tuning.
  - Explicit Checkpointing & Propagation of `embeddings.pt`:
    Because PEFT `save_pretrained()` only saves LoRA adapter weights (`adapter_model.safetensors`),
    this script explicitly saves and reloads `embeddings.pt` across sequential tasks (`task_idx > 0`)
    and during Triangular Evaluation to guarantee 100% preservation of new language token representations.
  - Bypasses M2M100 4.5x decoder regression via Dual Class-level Forward Patch and `CleanPEFTTrainer`.
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional

# CRITICAL VAST.AI / LOCAL FIX: Force HuggingFace cache to /workspace volume if present before hub imports
if os.path.exists("/workspace"):
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import sacrebleu
from datasets import Dataset

try:
    import peft.import_utils
    _orig_is_torchao_available = getattr(peft.import_utils, "is_torchao_available", None)
    if _orig_is_torchao_available:
        def _safe_is_torchao_available():
            try:
                return _orig_is_torchao_available()
            except ImportError:
                return False
        peft.import_utils.is_torchao_available = _safe_is_torchao_available
except Exception:
    pass

from peft import LoraConfig, get_peft_model, TaskType, PeftModel, set_peft_model_state_dict
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    NllbTokenizer,
    set_seed,
    Trainer,
    TrainingArguments,
    TrainerCallback,
)

# =============================================================================
# DUAL CLASS-LEVEL PATCH: Fix transformers 4.5x M2M100 regression.
# Root cause: M2M100Model.forward pre-computes decoder_inputs_embeds AND then
# passes BOTH decoder_input_ids AND decoder_inputs_embeds positionally to the
# decoder. Because the args are positional, kwargs-based checks don't intercept.
# Fix: Patch BOTH M2M100Model AND M2M100Decoder to unconditionally convert
# any input_ids to embeddings and null the ids, so the decoder never sees both.
# =============================================================================
from transformers.models.m2m_100.modeling_m2m_100 import (
    M2M100Model as _M2M100Model,
    M2M100Decoder as _M2M100Decoder,
)

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
    if decoder_input_ids is not None:
        if decoder_inputs_embeds is None:
            # M2M100ScaledWordEmbedding already applies ``embed_scale``.
            decoder_inputs_embeds = self.shared(decoder_input_ids)
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
    if input_ids is not None:
        if inputs_embeds is None:
            # M2M100ScaledWordEmbedding already applies ``embed_scale``.
            inputs_embeds = self.embed_tokens(input_ids)
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


# ═════════════════════════════════════════════════════════════════════════
# CONSTANTS & CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════
MODEL_NAME = "facebook/nllb-200-1.3B"
SRC_LANG = "spa_Latn"

LANG_TO_NLLB = {
    "quy": "quy_Latn",
    "ayr": "ayr_Latn",
    "shp": "shp_Latn",
    "agr": "agr_Latn",
    "cni": "cni_Latn",
}

ORDERS = {
    "order1": ["quy", "ayr", "shp", "agr", "cni"],
    "order2": ["cni", "agr", "shp", "ayr", "quy"],
    "order3": ["ayr", "cni", "agr", "shp", "quy"],
}

NEW_LANGUAGES = ["shp", "agr", "cni"]
BASE_LANGUAGES = ["quy", "ayr"]
ALL_LANGS = list(LANG_TO_NLLB.keys())

# Base filtered sentence counts (<=256 tokens) directly from data_in/summary_counts.txt
PAIRS_FILTERED_256 = {
    "quy": 276297,
    "ayr": 155751,
    "shp": 31143,
    "agr": 21963,
    "cni": 3880,
}

DYNAMIC_EPOCHS_MAP = {
    "quy": 4,    # 276k pairs -> 4 epochs = ~9,208 steps (at eff_batch=120/128)
    "ayr": 6,    # 155k pairs -> 6 epochs = ~7,788 steps
    "shp": 8,    # 31k pairs  -> 8 epochs = ~2,072 steps
    "agr": 10,   # 22k pairs  -> 10 epochs = ~1,830 steps
    "cni": 15,   # 3.8k pairs -> 15 epochs = ~450 steps (increased to ensure full morphological convergence on Asháninka)
}

def compute_expected_steps(batch_size: int, grad_accum: int, epochs_per_task: int, dynamic_epochs: bool = True) -> dict[str, dict]:
    """Dynamically computes steps per epoch and total steps per task based on batch size and gradient accumulation."""
    eff_batch = batch_size * grad_accum
    results = {}
    for lang, pairs in PAIRS_FILTERED_256.items():
        steps_per_epoch = max(1, pairs // eff_batch)
        lang_epochs = DYNAMIC_EPOCHS_MAP.get(lang, epochs_per_task) if dynamic_epochs else epochs_per_task
        total_steps = steps_per_epoch * lang_epochs
        results[lang] = {
            "pairs": pairs,
            "steps_per_epoch": steps_per_epoch,
            "epochs": lang_epochs,
            "total_steps": total_steps
        }
    return results


def is_main_process() -> bool:
    """Returns True if the current process is rank 0 in DDP/multi-GPU execution."""
    return int(os.environ.get("LOCAL_RANK", 0)) <= 0 and int(os.environ.get("RANK", 0)) <= 0


# ═════════════════════════════════════════════════════════════════════════
# CUSTOM TRAINER (`compute_loss` parity with NLLBTrainer in train_benchmarks.py)
# ═════════════════════════════════════════════════════════════════════════
class CleanPEFTTrainer(Trainer):
    """
    Bypasses M2M100Model.forward double-input bug while evaluating cross-entropy
    over PEFT/LoRA outputs with label smoothing (0.1).
    Ensures `self.label_smoother` is invoked if label smoothing is active in `TrainingArguments`.
    """
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        inputs.pop("decoder_input_ids", None)

        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        real_model = getattr(model, "module", model)
        cfg = getattr(real_model, "config", None)
        if cfg is None:
            cfg = getattr(real_model.base_model, "config", None)
        if cfg is None:
            cfg = real_model.base_model.model.config

        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = cfg.decoder_start_token_id
        shifted.masked_fill_(shifted == -100, cfg.pad_token_id)

        base_m = getattr(real_model, "base_model", real_model)
        if hasattr(base_m, "model"):
            inner = base_m.model
        else:
            inner = base_m

        emb = inner.get_input_embeddings()
        model_dtype = next(model.parameters()).dtype
        # ``emb`` is M2M100ScaledWordEmbedding, so applying sqrt(d_model)
        # again here would multiply decoder token vectors by 32 for NLLB-1.3B.
        dec_embeds = emb(shifted).to(dtype=model_dtype)

        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            decoder_inputs_embeds=dec_embeds,
            labels=labels,
        )
        inputs["labels"] = labels

        del dec_embeds, shifted

        if self.label_smoother is not None:
            # ``shifted`` already is [BOS, y0, y1, ...].  The logits must be
            # compared with labels at their matching positions, not shifted a
            # second time by LabelSmoother.
            loss = self.label_smoother(outputs, labels, shift_labels=False)
        else:
            loss = outputs.loss

        if return_outputs:
            return loss, outputs
        return loss


# ═════════════════════════════════════════════════════════════════════════
# TASK LOGGING & BEST ADAPTER CALLBACK (With Explicit embeddings.pt handling)
# ═════════════════════════════════════════════════════════════════════════
class SequentialTaskLogCallback(TrainerCallback):
    """
    Records training and validation metrics into training_log_X.csv / .json (e.g. quy_1).
    Evaluates dev set translations at each evaluation epoch.
    Explicitly tracks the highest chrF++ score and saves both:
      1. Lightweight LoRA adapter weights (`adapter_model.safetensors`)
      2. Explicit token embedding matrix (`embeddings.pt`)
    so the next sequential language starts from the exact optimal checkpoint with trained language tokens.
    """
    def __init__(
        self,
        model,
        tokenizer,
        lang_code: str,
        task_order_idx: int,
        dev_pair: Optional[tuple[list[str], list[str]]],
        output_dir: Path,
        task_dir: Path,
        gen_batch_size: int = 16,
        num_beams: int = 4
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.lang_code = lang_code
        self.task_order_idx = task_order_idx
        self.log_name = f"training_log_{lang_code}_{task_order_idx}"
        self.dev_pair = dev_pair
        self.output_dir = output_dir
        self.task_dir = task_dir
        self.gen_batch_size = gen_batch_size
        self.num_beams = num_beams

        self.log_csv_path = self.output_dir / f"{self.log_name}.csv"
        self.log_json_path = self.output_dir / f"{self.log_name}.json"
        self.task_csv_path = self.task_dir / f"{self.log_name}.csv"
        self.task_json_path = self.task_dir / f"{self.log_name}.json"

        self.best_chrf = -1.0
        self.best_adapter_dir = self.task_dir / "best_adapter"
        self.last_adapter_dir = self.task_dir / "last_adapter"

        self.history = []
        self._init_csv(self.log_csv_path)
        self._init_csv(self.task_csv_path)

    def _init_csv(self, path: Path):
        if not is_main_process():
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            with open(path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["step", "epoch", "train_loss", "val_loss", f"{self.lang_code}_chrf"])

    def on_evaluate(self, args, state, control, **kwargs):
        if not is_main_process():
            return control
        step = state.global_step
        epoch = round(state.epoch, 2) if state.epoch else float("nan")

        train_loss = next((x["loss"] for x in reversed(state.log_history) if "loss" in x), float("nan"))
        val_loss = kwargs.get("metrics", {}).get("eval_loss", float("nan"))

        chrf_score = float("nan")
        if self.dev_pair:
            device = next(self.model.parameters()).device
            dev_dict = {"src": self.dev_pair[0], "tgt": self.dev_pair[1]}
            chrf_score = evaluate_language_chrf(
                self.model,
                self.tokenizer,
                dev_dict,
                self.lang_code,
                device,
                num_beams=self.num_beams,
                batch_size=self.gen_batch_size
            )

        print(f"   [Task Log {self.log_name}] Step: {step} | Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dev chrF++: {chrf_score:.2f}")
        log_gpu_memory(f"Eval @ Step {step}")

        row = [step, epoch, train_loss, val_loss, round(chrf_score, 2) if not math.isnan(chrf_score) else float("nan")]
        for p in [self.log_csv_path, self.task_csv_path]:
            with open(p, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

        record = {"step": step, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, f"{self.lang_code}_chrf": round(chrf_score, 2) if not math.isnan(chrf_score) else None}
        self.history.append(record)
        for p in [self.log_json_path, self.task_json_path]:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)

        # Track and save best adapter (LoRA weights + explicit embeddings.pt)
        if not math.isnan(chrf_score) and chrf_score > self.best_chrf:
            self.best_chrf = chrf_score
            self.best_adapter_dir.mkdir(parents=True, exist_ok=True)
            real_m = getattr(self.model, "module", self.model)
            real_m.save_pretrained(self.best_adapter_dir)
            self.tokenizer.save_pretrained(self.best_adapter_dir)

            # Explicitly save embedding layer to avoid PEFT discard of non-LoRA modules
            emb_layer = real_m.get_input_embeddings() if hasattr(real_m, "get_input_embeddings") else self.model.get_input_embeddings()
            torch.save(emb_layer.weight.data, self.best_adapter_dir / "embeddings.pt")

            with open(self.best_adapter_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"step": step, "epoch": epoch, "best_chrf": round(chrf_score, 2)}, f, indent=2)
            print(f"   ✓ [BEST ADAPTER] New highest chrF++ ({chrf_score:.2f}) at epoch {epoch}! Saved LoRA + embeddings.pt to {self.best_adapter_dir.name}/")

        return control


# ═════════════════════════════════════════════════════════════════════════
# DATA LOADING & EVALUATION HELPERS
# ═════════════════════════════════════════════════════════════════════════
def resolve_paths(args):
    """Resolves data_in and models directories cleanly across Vast.ai, local, and Kaggle environments."""
    project_root = Path(__file__).resolve().parent.parent.parent

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
        # Traditional folders / Vast.ai workspace priority
        data_dir = project_root / "data_in"
        if not (data_dir / "train").exists():
            if (project_root / "data" / "train").exists():
                data_dir = project_root / "data"
            elif Path("/workspace/data_in/train").exists():
                data_dir = Path("/workspace/data_in")
            elif Path("/kaggle/input/data_in/train").exists():
                data_dir = Path("/kaggle/input/data_in")
            elif Path("/kaggle/input/data-in/train").exists():
                data_dir = Path("/kaggle/input/data-in")

    if args.output_dir:
        base_out = Path(args.output_dir)
    else:
        base_out = project_root / "models"
        if not base_out.parent.exists() and os.path.exists("/workspace"):
            base_out = Path("/workspace/models")
        elif not base_out.parent.exists() and os.path.exists("/kaggle/working"):
            base_out = Path("/kaggle/working/models")

    return data_dir, base_out


def load_single_language_dataset(lang_code: str, data_dir: Path, tokenizer: NllbTokenizer, max_length: int = 256) -> Optional[Dataset]:
    """Loads parallel train.{es,lang} (preferring train.filtered.* if present)."""
    train_dir = data_dir / "train" / lang_code
    es_path = train_dir / "train.filtered.es"
    lang_path = train_dir / f"train.filtered.{lang_code}"

    if not es_path.exists() or not lang_path.exists():
        es_path = train_dir / "train.es"
        lang_path = train_dir / f"train.{lang_code}"
        if not es_path.exists() or not lang_path.exists():
            print(f"WARNING: Training files missing for {lang_code} inside {train_dir}")
            return None

    with open(es_path, encoding="utf-8") as f:
        src_lines = [l.strip() for l in f if l.strip()]
    with open(lang_path, encoding="utf-8") as f:
        tgt_lines = [l.strip() for l in f if l.strip()]

    min_len = min(len(src_lines), len(tgt_lines))
    src_lines, tgt_lines = src_lines[:min_len], tgt_lines[:min_len]

    print(f"Loaded {min_len:,} parallel sentences for task [{lang_code.upper()}] from {es_path.name}")

    def tokenize(batch):
        tokenizer.src_lang = SRC_LANG
        tokenizer.tgt_lang = LANG_TO_NLLB[lang_code]
        return tokenizer(batch["src"], text_target=batch["tgt"], truncation=True, max_length=max_length, padding=False)

    raw = Dataset.from_dict({"src": src_lines, "tgt": tgt_lines})
    return raw.map(tokenize, batched=True, batch_size=1024, remove_columns=["src", "tgt"], keep_in_memory=True)


def load_dev_raw(lang_code: str, data_dir: Path) -> Optional[tuple[list[str], list[str]]]:
    """Loads dev.{es,lang} text lines for generation."""
    es_path = data_dir / "dev" / lang_code / "dev.es"
    lang_path = data_dir / "dev" / lang_code / f"dev.{lang_code}"
    if not es_path.exists() or not lang_path.exists():
        return None
    with open(es_path, encoding="utf-8") as f:
        es_lines = [l.strip() for l in f if l.strip()]
    with open(lang_path, encoding="utf-8") as f:
        lang_lines = [l.strip() for l in f if l.strip()]
    min_len = min(len(es_lines), len(lang_lines))
    return es_lines[:min_len], lang_lines[:min_len]


def evaluate_language_chrf(
    model,
    tokenizer,
    dev_data: dict,
    lang_code: str,
    device: torch.device,
    num_beams: int = 4,
    batch_size: int = 16
) -> float:
    """Runs generation and returns chrF++ corpus score for a specific language."""
    sources = dev_data["src"]
    refs = dev_data["tgt"]
    tgt_tag = LANG_TO_NLLB[lang_code]
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_tag)

    preds = []
    chrf_calculator = sacrebleu.metrics.CHRF(word_order=2)

    for i in range(0, len(sources), batch_size):
        batch_src = sources[i : i + batch_size]
        tokenizer.src_lang = SRC_LANG
        inputs = tokenizer(batch_src, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)

        real_model = getattr(model, "module", model)
        with torch.no_grad():
            gen_tokens = real_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=256,
                num_beams=num_beams,
            )
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        preds.extend([d.strip() for d in decoded])

    # SacreBLEU expects one complete reference stream per reference set.  With
    # one reference per source sentence this is ``[refs]``.  The old
    # ``[[r] for r in refs]`` shape silently scored only the first dev example.
    return chrf_calculator.corpus_score(preds, [refs]).score


def log_gpu_memory(prefix: str = ""):
    """Logs current GPU VRAM allocation and reserved cache to stdout."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"[GPU VRAM {prefix}] Allocated: {allocated:.2f} GB | Reserved (Cache): {reserved:.2f} GB | Total: {total:.2f} GB ({allocated/total*100:.1f}%)")


# ═════════════════════════════════════════════════════════════════════════
# EMBEDDING INITIALIZATION & SELECTIVE MASKING (Exact train_benchmarks.py parity)
# ═════════════════════════════════════════════════════════════════════════
def initialize_new_language_embeddings(model, tokenizer):
    """Initializes target rows (shp_Latn, agr_Latn, cni_Latn) as average of Quechua + Aymara."""
    print("\n---> Initializing target embeddings for new languages (shp, agr, cni) using (Quechua + Aymara)/2...")
    model.resize_token_embeddings(len(tokenizer))
    quy_id = tokenizer.convert_tokens_to_ids(LANG_TO_NLLB["quy"])
    ayr_id = tokenizer.convert_tokens_to_ids(LANG_TO_NLLB["ayr"])

    emb_layer = model.get_input_embeddings()
    with torch.no_grad():
        emb = emb_layer.weight
        avg_vec = (emb[quy_id] + emb[ayr_id]) / 2.0
        for lang in NEW_LANGUAGES:
            target_id = tokenizer.convert_tokens_to_ids(LANG_TO_NLLB[lang])
            if target_id == tokenizer.unk_token_id or target_id is None:
                raise ValueError(f"CRITICAL ERROR: Token '{LANG_TO_NLLB[lang]}' is still <unk> (ID={target_id})! Tokenizer was not properly resized.")
            emb[target_id] = avg_vec.clone()
            print(f"     -> Initialized [{lang.upper()}] ({LANG_TO_NLLB[lang]} | ID={target_id}) with avg(quy, ayr) embedding.")
    print("     Initialization complete.")


def apply_embedding_masking(model, tokenizer, active_langs):
    """
    Applies selective embedding gradient masking for active target languages only.
    Because `LoraConfig` freezes the embedding table (`requires_grad = False`),
    we re-enable `requires_grad = True` and register a backward hook so ONLY the
    5 Peruvian indigenous language tokens receive gradients while preserving NLLB-200 global space.
    """
    print("\n---> Applying selective embedding gradient mask for target language tokens only...")
    target_ids = [tokenizer.convert_tokens_to_ids(LANG_TO_NLLB[lang]) for lang in active_langs]

    emb_layer = model.get_input_embeddings()
    emb_layer.weight.requires_grad = True # Enforce gradient susceptibility

    target_ids_tensor = torch.tensor(target_ids, dtype=torch.long, device=emb_layer.weight.device)

    def hook(grad):
        mask = torch.zeros_like(grad)
        mask[target_ids_tensor] = 1.0
        return grad * mask

    emb_layer.weight.register_hook(hook)
    print(f"     -> Gradient hook registered for token IDs: {target_ids} ({active_langs})")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Sequential LoRA (r=128) Continual Learning for Experiment B")
    parser.add_argument("--order", type=str, default="order1", choices=["order1", "order2", "order3"], help="Sequential language order (order1: quy->cni, order2: cni->quy, order3: ayr->quy)")
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=256, help="LoRA alpha scaling (alpha=2*r=256 for exact parity with Experiment A)")
    parser.add_argument("--lr", type=float, default=1e-4, help="LoRA learning rate (1e-4 recommended to prevent catastrophic forgetting under r=128)")
    parser.add_argument("--epochs_per_task", type=int, default=12, help="Number of epochs to train on each sequential language task (if static_epochs is set)")
    parser.add_argument("--dynamic_epochs", action="store_true", default=True, help="Dynamically scale epochs per task based on dataset size to prevent overfitting")
    parser.add_argument("--static_epochs", action="store_true", default=False, help="Force exact --epochs_per_task for all languages (disables dynamic_epochs)")
    # Label smoothing constructs a full-vocabulary log-softmax.  A per-device
    # batch of 14 OOMs on 31-GB cards for long NLLB batches; 8 x 15 preserves a
    # comparable effective batch (120) with enough headroom for that tensor.
    parser.add_argument("--batch_size", type=int, default=8, help="Per-device batch size (8 fits 31-GB GPUs with NLLB label smoothing)")
    parser.add_argument("--grad_accum", type=int, default=15, help="Gradient accumulation steps (8 * 15 = effective batch 120)")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams during generation & evaluation (protocol default 4/5)")
    parser.add_argument("--experiment_name", type=str, default="experiment_B", help="Top-level output folder")
    parser.add_argument("--data_dir", type=str, default=None, help="Explicit path to data directory containing train/ and dev/ folders")
    parser.add_argument("--output_dir", type=str, default=None, help="Explicit path to top-level models output directory")
    parser.add_argument("--start_task", type=int, default=1, help="Task index (1 to 5) to start training from if resuming across sessions")
    parser.add_argument("--seed", type=int, default=42, help="Random seed applied before model and adapter construction")
    args = parser.parse_args()

    # Reproducible model, adapter, data-order and generation setup.  This must
    # precede base-model and PEFT LoRA construction.
    set_seed(args.seed)

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if local_rank != -1 and torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    use_dynamic_epochs = args.dynamic_epochs and not args.static_epochs
    order_langs = ORDERS[args.order]
    data_dir, base_out = resolve_paths(args)

    method_tag = f"lora_sequential_r{args.lora_r}"
    lr_tag = f"lr{args.lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    output_dir = base_out / args.experiment_name / method_tag / args.order / lr_tag
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    use_bf16 = False
    if torch.cuda.is_available():
        try:
            dev_idx = local_rank if local_rank >= 0 else 0
            cap = torch.cuda.get_device_capability(dev_idx)
            if cap[0] >= 8 and torch.cuda.is_bf16_supported():
                use_bf16 = True
        except Exception:
            pass
    use_fp16 = torch.cuda.is_available() and not use_bf16
    dtype_str = "bfloat16" if use_bf16 else ("float16" if use_fp16 else "float32")

    # Dynamic step calculation based on configured batch size, grad accum, and epochs
    expected_steps_dict = compute_expected_steps(args.batch_size, args.grad_accum, args.epochs_per_task, dynamic_epochs=use_dynamic_epochs)
    eff_batch = args.batch_size * args.grad_accum

    print("════════════════════════════════════════════════════════════════════════")
    print(" EXPERIMENT B: SEQUENTIAL LORA CONTINUAL LEARNING (Traditional Folders / Vast.ai)")
    print(f" Order:        {args.order.upper()} -> {' -> '.join(order_langs)}")
    print(f" Data Dir:     {data_dir}")
    print(f" Output Dir:   {output_dir}")
    print(f" LoRA Rank:    r={args.lora_r}, alpha={args.lora_alpha}, targets=[q, k, v, o, fc1, fc2]")
    print(f" Seed:         {args.seed} (set before model and adapter construction)")
    print(f" Precision:    {dtype_str}")
    print(f" Batching:     batch_size={args.batch_size} * grad_accum={args.grad_accum} = Effective Batch {eff_batch}")
    print(f" Epoch Mode:   {'Dynamic Scaling (prevents overfitting on QUY/AYR)' if use_dynamic_epochs else f'Static ({args.epochs_per_task} epochs/task)'}")
    print(f" Num Beams:    {args.num_beams} (for evaluation & generation)")
    print(" Expected Dynamic Steps per Language:")
    for l_c, info in expected_steps_dict.items():
        print(f"   - {l_c.upper()} ({LANG_TO_NLLB[l_c]}): {info['epochs']} epochs | ~{info['total_steps']:,} total steps (~{info['steps_per_epoch']:,} steps/epoch | {info['pairs']:,} pairs)")
    print("════════════════════════════════════════════════════════════════════════\n")

    # 1. Load Tokenizer
    tok_path = MODEL_NAME
    project_root = Path(__file__).resolve().parent.parent.parent
    local_tokenizer_dirs = [
        project_root / "models/v0/tokenizer",
        Path("models/v0/tokenizer"),
        Path("/workspace/models/v0/tokenizer"),
        Path("/kaggle/input/datasets/ceviche98/data-in/models/v0/tokenizer"),
    ]
    for cand in local_tokenizer_dirs:
        if cand.exists() and (cand / "tokenizer_config.json").exists():
            tok_path = str(cand)
            break

    if tok_path != MODEL_NAME:
        print(f"---> Loading custom modified Tokenizer from local path: {tok_path}...")
    else:
        print(f"---> Loading Tokenizer from Hub ({MODEL_NAME})...")

    tokenizer = NllbTokenizer.from_pretrained(tok_path, src_lang=SRC_LANG)

    # Ensure shp_Latn, agr_Latn, cni_Latn exist in vocabulary as unique tokens
    new_tokens_needed = []
    for lang in NEW_LANGUAGES:
        t_str = LANG_TO_NLLB[lang]
        tid = tokenizer.convert_tokens_to_ids(t_str)
        if tid == tokenizer.unk_token_id or tid is None:
            new_tokens_needed.append(t_str)

    if new_tokens_needed:
        print(f"---> Adding new language tokens to vocabulary: {new_tokens_needed}")
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens_needed})

    # 2. Load Base Model with auto-detected dtype (Checking local models/nllb-200-1.3B first)
    model_load_path = MODEL_NAME
    local_model_dirs = [
        project_root / "models/nllb-200-1.3B",
        Path("models/nllb-200-1.3B"),
        Path("/workspace/models/nllb-200-1.3B"),
        Path("/kaggle/input/models/nllb-200-1.3B"),
    ]
    for cand_m in local_model_dirs:
        if cand_m.exists() and (cand_m / "config.json").exists():
            model_load_path = str(cand_m)
            break

    if model_load_path != MODEL_NAME:
        print(f"---> Loading Base Model from local folder: {model_load_path}...")
    else:
        print(f"---> Loading Base Model from Hub ({MODEL_NAME})...")

    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_load_path, torch_dtype=load_dtype)

    # 3. Initialize embeddings BEFORE wrapping with LoRA
    initialize_new_language_embeddings(base_model, tokenizer)

    # 4. Create LoRA Adapter Configuration
    print(f"\n---> Configuring LoRA Adapter (r={args.lora_r})...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        # Final Experiment-B capacity: the full attention block, without the
        # much larger FFN adapters. Must match CaLoRA's Q/K/V/O modules.
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(base_model, lora_config)

    # 5. CRITICAL: Apply selective embedding gradient masking right after wrapping with LoRA
    # This ensures active language tokens receive gradients and updates are preserved.
    apply_embedding_masking(model, tokenizer, ALL_LANGS)

    if is_main_process():
        model.print_trainable_parameters()

    model.to(device)

    # 6. Load Dev Raw sets for Triangular Evaluation & Task Logging
    dev_data_dict = {}
    for lc in order_langs:
        pair = load_dev_raw(lc, data_dir)
        if pair:
            dev_data_dict[lc] = {"src": pair[0], "tgt": pair[1]}
        else:
            print(f"WARNING: Dev files missing for {lc} inside {data_dir / 'dev' / lc}")

    eval_matrix = {}

    # 7. SEQUENTIAL CONTINUAL LEARNING LOOP
    for task_idx, lang_code in enumerate(order_langs):
        print("\n" + "█" * 70)
        print(f" TASK {task_idx + 1}/{len(order_langs)}: TRAINING ON [{lang_code.upper()}] ({LANG_TO_NLLB[lang_code]})")
        print("█" * 70)
        log_gpu_memory(f"Start Task {task_idx + 1} ({lang_code.upper()})")

        # Skip training if resuming from a later task index (--start_task)
        if task_idx + 1 < args.start_task:
            print(f"---> Skipping Task {task_idx + 1} ({lang_code.upper()}) because --start_task is set to {args.start_task}.")
            continue

        # If this is task 2 or later, guarantee starting exactly from the BEST adapter (~80MB) + embeddings.pt of previous task
        if task_idx > 0:
            prev_lang = order_langs[task_idx - 1]
            prev_best_dir = output_dir / f"task_{task_idx}_{prev_lang}" / "best_adapter"
            print(f"---> Propagating optimal checkpoint: Loading BEST LoRA adapter and embeddings.pt from [{prev_lang.upper()}] ({prev_best_dir.name})...")

            from safetensors.torch import load_file

            adapter_file = prev_best_dir / "adapter_model.safetensors"
            if not adapter_file.exists():
                adapter_file = prev_best_dir / "adapter_model.bin"
                if adapter_file.exists():
                    state_dict = torch.load(adapter_file, map_location=device, weights_only=True)
                else:
                    print(f"WARNING: Could not find adapter file in {prev_best_dir}; using current weights.")
                    state_dict = None
            else:
                state_dict = load_file(adapter_file)

            if state_dict is not None:
                set_peft_model_state_dict(model, state_dict)
                print(f"     ✓ Successfully loaded best LoRA weights from [{prev_lang.upper()}]!")

            # CRITICAL: Reload embeddings.pt so language token modifications carry over!
            emb_file = prev_best_dir / "embeddings.pt"
            if emb_file.exists():
                emb_data = torch.load(emb_file, map_location=device, weights_only=True)
                model.get_input_embeddings().weight.data.copy_(emb_data)
                print(f"     ✓ Successfully loaded best embeddings.pt from [{prev_lang.upper()}]!")
            else:
                print(f"WARNING: Could not find embeddings.pt inside {prev_best_dir}! Language tokens might be out of sync.")

        train_ds = load_single_language_dataset(lang_code, data_dir, tokenizer, max_length=256)
        if not train_ds:
            print(f"Skipping task {lang_code} due to missing training data.")
            continue

        dev_pair = load_dev_raw(lang_code, data_dir)
        dev_ds = None
        if dev_pair:
            dev_raw = Dataset.from_dict({"src": dev_pair[0], "tgt": dev_pair[1]})
            def tokenize_dev(batch):
                tokenizer.src_lang = SRC_LANG
                tokenizer.tgt_lang = LANG_TO_NLLB[lang_code]
                return tokenizer(batch["src"], text_target=batch["tgt"], truncation=True, max_length=256, padding=False)
            dev_ds = dev_raw.map(tokenize_dev, batched=True, batch_size=1024, remove_columns=["src", "tgt"], keep_in_memory=True)

        steps_per_epoch = max(1, len(train_ds) // eff_batch)
        lang_epochs = DYNAMIC_EPOCHS_MAP.get(lang_code, args.epochs_per_task) if use_dynamic_epochs else args.epochs_per_task
        max_steps = steps_per_epoch * lang_epochs
        print(f"Task dataset size: {len(train_ds):,} | Epochs for [{lang_code.upper()}]: {lang_epochs} | Steps per epoch: {steps_per_epoch} | Total task steps: {max_steps}")

        task_out_dir = output_dir / f"task_{task_idx + 1}_{lang_code}"
        task_out_dir.mkdir(parents=True, exist_ok=True)

        has_eval = dev_ds is not None and len(dev_ds) > 0
        eval_strat = "epoch" if has_eval else "no"

        training_args = TrainingArguments(
            output_dir=str(task_out_dir / "hf"),
            max_steps=max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.06,
            lr_scheduler_type="cosine",
            eval_strategy=eval_strat if hasattr(TrainingArguments, "eval_strategy") else eval_strat,
            save_strategy="no",               # We explicitly save best_adapter & last_adapter manually via callback
            logging_steps=max(5, steps_per_epoch // 4),
            bf16=use_bf16,
            fp16=use_fp16,
            optim="adamw_torch",
            max_grad_norm=1.0,
            label_smoothing_factor=0.1,       # CRITICAL: Parity with train_benchmarks.py & Protocol
            weight_decay=0.0,                 # CRITICAL: Must be 0.0 to prevent AdamW from erasing frozen tokens
            ignore_data_skip=True,            # CRITICAL: Resume instantly without fast-forwarding data
            group_by_length=False,            # CRITICAL: Constant tensor shapes allow cudnn.benchmark to lock in peak GPU throughput
            dataloader_num_workers=4 if os.name != "nt" else 0,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True if os.name != "nt" else False,
            dataloader_prefetch_factor=2 if os.name != "nt" else None,
            tf32=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
        )

        task_logger = SequentialTaskLogCallback(
            model=model,
            tokenizer=tokenizer,
            lang_code=lang_code,
            task_order_idx=task_idx + 1,
            dev_pair=dev_pair,
            output_dir=output_dir,
            task_dir=task_out_dir,
            gen_batch_size=16,
            num_beams=args.num_beams
        )

        trainer = CleanPEFTTrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds if has_eval else None,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100),
            callbacks=[task_logger],
        )

        # Train on current task
        trainer.train()

        # Save final epoch adapter (LoRA weights + explicit embeddings.pt) to last_adapter/
        last_dir = task_out_dir / "last_adapter"
        real_m = getattr(model, "module", model)
        if is_main_process():
            last_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n---> Saving final epoch (last_adapter) LoRA weights and embeddings.pt to: {last_dir.name}/")
            real_m.save_pretrained(last_dir)
            tokenizer.save_pretrained(last_dir)

            emb_layer = real_m.get_input_embeddings() if hasattr(real_m, "get_input_embeddings") else model.get_input_embeddings()
            torch.save(emb_layer.weight.data, last_dir / "embeddings.pt")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Before running Triangular Evaluation, make sure active model holds the BEST adapter of this task
        best_dir = task_out_dir / "best_adapter"
        if best_dir.exists():
            if is_main_process():
                print(f"---> Reloading BEST adapter + embeddings.pt from {best_dir.name}/ for Triangular Evaluation and next task baseline...")
            from safetensors.torch import load_file
            adapter_file = best_dir / "adapter_model.safetensors" if (best_dir / "adapter_model.safetensors").exists() else (best_dir / "adapter_model.bin")
            if adapter_file.exists():
                st_dict = load_file(adapter_file) if adapter_file.suffix == ".safetensors" else torch.load(adapter_file, map_location=device, weights_only=True)
                set_peft_model_state_dict(real_m, st_dict)

            emb_file = best_dir / "embeddings.pt"
            if emb_file.exists():
                emb_data = torch.load(emb_file, map_location=device, weights_only=True)
                model.get_input_embeddings().weight.data.copy_(emb_data)
        else:
            # If for some reason best_adapter didn't save (e.g. no dev set), copy last_adapter as best
            if is_main_process():
                best_dir.mkdir(parents=True, exist_ok=True)
                real_m.save_pretrained(best_dir)
                tokenizer.save_pretrained(best_dir)
                emb_layer = real_m.get_input_embeddings() if hasattr(real_m, "get_input_embeddings") else model.get_input_embeddings()
                torch.save(emb_layer.weight.data, best_dir / "embeddings.pt")
            if torch.distributed.is_initialized():
                torch.distributed.barrier()

        # ── TRIANGULAR EVALUATION MATRIX (Protocolo de Evaluación Triangular) ──
        if is_main_process():
            print("\n" + "─" * 70)
            print(f" [TRIANGULAR EVALUATION] Evaluating Dev sets (num_beams={args.num_beams}) after Task {task_idx + 1} ({lang_code.upper()})...")
            print("─" * 70)
            model.eval()
            eval_matrix[lang_code] = {}

            for k_idx in range(task_idx + 1):
                past_lang = order_langs[k_idx]
                if past_lang in dev_data_dict:
                    chrf = evaluate_language_chrf(
                        model,
                        tokenizer,
                        dev_data_dict[past_lang],
                        past_lang,
                        device,
                        num_beams=args.num_beams,
                        batch_size=16
                    )
                    eval_matrix[lang_code][past_lang] = round(chrf, 2)
                    print(f"   Dev [{past_lang.upper()}]: chrF++ = {chrf:.2f}")

            with open(output_dir / "triangular_eval_matrix.json", "w", encoding="utf-8") as f:
                json.dump(eval_matrix, f, indent=2, ensure_ascii=False)
            model.train()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # ── EXPLICIT GPU MEMORY CLEANUP TO PREVENT OOM ACROSS SEQUENTIAL TASKS ──
        print("\n---> [MEMORY CLEANUP] Freeing trainer, dataloaders and clearing PyTorch CUDA cache...")
        del trainer
        del train_ds
        if dev_ds is not None:
            del dev_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("     ✓ CUDA VRAM cache cleaned successfully.\n")
        log_gpu_memory(f"Post-Cleanup Task {task_idx + 1}")

    # 8. Print Summary Report & Calculate Continual Learning Metrics (FM, AA, BWT)
    if is_main_process():
        print("\n" + "═" * 70)
        print(f" EXPERIMENT B (SEQUENTIAL LORA r={args.lora_r} | {args.order.upper()}) FINAL REPORT")
        print("═" * 70)

        header = "Task Trained \\ Dev Eval | " + " | ".join([f"{lc.upper():10s}" for lc in order_langs])
        print(header)
        print("-" * len(header))
        for t_lang in order_langs:
            if t_lang in eval_matrix:
                row_str = f"{t_lang.upper():21s} | "
                for k_lang in order_langs:
                    val = eval_matrix[t_lang].get(k_lang, "---")
                    row_str += f"{str(val):10s} | "
                print(row_str)

        final_lang = order_langs[-1]
        if final_lang in eval_matrix:
            final_scores = eval_matrix[final_lang]
            aa = np.mean(list(final_scores.values()))
            print(f"\n---> Average Accuracy (AA) across all 5 languages: {aa:.2f} chrF++")

            forgetting_vals = {}
            bwt_vals = {}
            for k_lang in order_langs[:-1]:
                max_k = max([eval_matrix[t_lang].get(k_lang, 0.0) for t_lang in eval_matrix])
                diag_k = eval_matrix.get(k_lang, {}).get(k_lang, 0.0)
                final_k = final_scores.get(k_lang, 0.0)
                forgetting_vals[k_lang] = round(max_k - final_k, 2)
                bwt_vals[k_lang] = round(final_k - diag_k, 2)

            mean_fm = np.mean(list(forgetting_vals.values())) if forgetting_vals else 0.0
            mean_bwt = np.mean(list(bwt_vals.values())) if bwt_vals else 0.0

            print(f"---> Mean Forgetting Measure (FM): {mean_fm:.2f} chrF++ drop across past tasks")
            print(f"     Per-language Forgetting (max - final): {forgetting_vals}")
            print(f"---> Mean Backward Transfer (BWT): {mean_bwt:+.2f} chrF++ change across past tasks")
            print(f"     Per-language BWT (final - diag):       {bwt_vals}")

            summary_data = {
                "order": args.order,
                "lora_r": args.lora_r,
                "epochs_per_task": args.epochs_per_task,
                "average_accuracy_AA": round(aa, 2),
                "mean_forgetting_measure_FM": round(mean_fm, 2),
                "mean_backward_transfer_BWT": round(mean_bwt, 2),
                "per_language_forgetting": forgetting_vals,
                "per_language_bwt": bwt_vals,
                "triangular_matrix": eval_matrix
            }
            with open(output_dir / "continual_learning_summary.json", "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)

        print("═" * 70)
        print(" SEQUENTIAL CONTINUAL LEARNING COMPLETE!")
        print("═" * 70)


if __name__ == "__main__":
    main()
