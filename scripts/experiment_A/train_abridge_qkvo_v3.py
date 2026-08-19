#!/usr/bin/env python3
"""A-Bridge v3: standalone joint multilingual Q/K/V/O LoRA control.

This is deliberately independent of ``train_benchmarks.py``.  The legacy
Experiment-A code and its prior results remain untouched; this script owns the
current Transformers compatibility patch, objective, data pipeline and
multilingual chrF++ evaluation used to bridge Experiment A to Experiment B.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path

import numpy as np
import sacrebleu
import torch
from datasets import Dataset, concatenate_datasets, interleave_datasets
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    NllbTokenizer,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from transformers.models.m2m_100.modeling_m2m_100 import (
    M2M100Decoder,
    M2M100Model,
)
from transformers.trainer_utils import get_last_checkpoint


if os.path.exists("/workspace"):
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")


LANG_TO_NLLB = {
    "quy": "quy_Latn",
    "ayr": "ayr_Latn",
    "shp": "shp_Latn",
    "agr": "agr_Latn",
    "cni": "cni_Latn",
}
ALL_LANGS = list(LANG_TO_NLLB)
NEW_LANGUAGES = ("shp", "agr", "cni")
SRC_LANG = "spa_Latn"


# Transformers versions used on Vast can route both decoder_input_ids and
# decoder_inputs_embeds to M2M100Decoder.  The decoder accepts only one.  Its
# embedding module is M2M100ScaledWordEmbedding, which already applies the
# configured embed_scale: do not multiply it by sqrt(d_model) again.
_ORIGINAL_MODEL_FORWARD = M2M100Model.forward
_ORIGINAL_DECODER_FORWARD = M2M100Decoder.forward


def _safe_m2m_model_forward(
    self, input_ids=None, attention_mask=None,
    decoder_input_ids=None, decoder_attention_mask=None,
    head_mask=None, decoder_head_mask=None, cross_attn_head_mask=None,
    encoder_outputs=None, past_key_values=None,
    inputs_embeds=None, decoder_inputs_embeds=None,
    use_cache=None, output_attentions=None, output_hidden_states=None,
    return_dict=None, **kwargs,
):
    if decoder_input_ids is not None:
        if decoder_inputs_embeds is None:
            decoder_inputs_embeds = self.shared(decoder_input_ids)
        decoder_input_ids = None
    return _ORIGINAL_MODEL_FORWARD(
        self,
        input_ids=input_ids, attention_mask=attention_mask,
        decoder_input_ids=decoder_input_ids,
        decoder_attention_mask=decoder_attention_mask,
        head_mask=head_mask, decoder_head_mask=decoder_head_mask,
        cross_attn_head_mask=cross_attn_head_mask,
        encoder_outputs=encoder_outputs, past_key_values=past_key_values,
        inputs_embeds=inputs_embeds, decoder_inputs_embeds=decoder_inputs_embeds,
        use_cache=use_cache, output_attentions=output_attentions,
        output_hidden_states=output_hidden_states, return_dict=return_dict,
        **kwargs,
    )


def _safe_m2m_decoder_forward(
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
            inputs_embeds = self.embed_tokens(input_ids)
        input_ids = None
    return _ORIGINAL_DECODER_FORWARD(
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


M2M100Model.forward = _safe_m2m_model_forward
M2M100Decoder.forward = _safe_m2m_decoder_forward


def _model_config(model):
    return getattr(model, "config", None) or model.base_model.model.config


def _input_embedding(model):
    base = getattr(model, "base_model", model)
    inner = getattr(base, "model", base)
    return inner.get_input_embeddings()


class ABridgeTrainer(Trainer):
    """Teacher-forced loss with the safe M2M100 decoder input path."""

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        inputs.pop("decoder_input_ids", None)
        if labels is None:
            return super().compute_loss(
                model, inputs, return_outputs=return_outputs,
                num_items_in_batch=num_items_in_batch,
            )

        cfg = _model_config(model)
        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1]
        shifted[:, 0] = cfg.decoder_start_token_id
        shifted.masked_fill_(shifted == -100, cfg.pad_token_id)
        embeds = _input_embedding(model)(shifted).to(
            dtype=next(model.parameters()).dtype
        )
        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            decoder_inputs_embeds=embeds,
            labels=labels,
        )
        inputs["labels"] = labels
        if self.label_smoother is not None:
            loss = self.label_smoother(outputs, labels, shift_labels=False)
        else:
            loss = outputs.loss
        return (loss, outputs) if return_outputs else loss


def load_train_language(lang_code: str, data_dir: Path, tokenizer: NllbTokenizer) -> Dataset:
    es_path = data_dir / "train" / lang_code / "train.filtered.es"
    tgt_path = data_dir / "train" / lang_code / f"train.filtered.{lang_code}"
    if not es_path.exists() or not tgt_path.exists():
        es_path = data_dir / "train" / lang_code / "train.es"
        tgt_path = data_dir / "train" / lang_code / f"train.{lang_code}"
    if not es_path.exists() or not tgt_path.exists():
        raise FileNotFoundError(f"No parallel training files for {lang_code}.")

    with es_path.open(encoding="utf-8") as src_f:
        src = [line.strip() for line in src_f if line.strip()]
    with tgt_path.open(encoding="utf-8") as tgt_f:
        tgt = [line.strip() for line in tgt_f if line.strip()]
    size = min(len(src), len(tgt))
    raw = Dataset.from_dict({"src": src[:size], "tgt": tgt[:size]})

    def tokenize(batch):
        tokenizer.src_lang = SRC_LANG
        tokenizer.tgt_lang = LANG_TO_NLLB[lang_code]
        return tokenizer(
            batch["src"], text_target=batch["tgt"], truncation=True,
            max_length=256, padding=False,
        )

    return raw.map(tokenize, batched=True, batch_size=512, remove_columns=["src", "tgt"])


def load_dev_language(lang_code: str, data_dir: Path) -> tuple[list[str], list[str]]:
    es_path = data_dir / "dev" / lang_code / "dev.es"
    tgt_path = data_dir / "dev" / lang_code / f"dev.{lang_code}"
    with es_path.open(encoding="utf-8") as src_f:
        src = [line.strip() for line in src_f if line.strip()]
    with tgt_path.open(encoding="utf-8") as tgt_f:
        tgt = [line.strip() for line in tgt_f if line.strip()]
    size = min(len(src), len(tgt))
    return src[:size], tgt[:size]


def tokenise_dev_language(lang_code: str, sources: list[str], targets: list[str], tokenizer: NllbTokenizer) -> Dataset:
    raw = Dataset.from_dict({"src": sources, "tgt": targets})

    def tokenize(batch):
        tokenizer.src_lang = SRC_LANG
        tokenizer.tgt_lang = LANG_TO_NLLB[lang_code]
        return tokenizer(
            batch["src"], text_target=batch["tgt"], truncation=True,
            max_length=256, padding=False,
        )

    return raw.map(tokenize, batched=True, remove_columns=["src", "tgt"])


def initialise_language_embeddings(model, tokenizer: NllbTokenizer) -> None:
    model.resize_token_embeddings(len(tokenizer))
    embedding = _input_embedding(model)
    quy_id = tokenizer.convert_tokens_to_ids(LANG_TO_NLLB["quy"])
    ayr_id = tokenizer.convert_tokens_to_ids(LANG_TO_NLLB["ayr"])
    with torch.no_grad():
        average = (embedding.weight[quy_id] + embedding.weight[ayr_id]) / 2
        for lang in NEW_LANGUAGES:
            language_id = tokenizer.convert_tokens_to_ids(LANG_TO_NLLB[lang])
            if language_id is None or language_id == tokenizer.unk_token_id:
                raise ValueError(f"Tokenizer lacks {LANG_TO_NLLB[lang]}.")
            embedding.weight[language_id] = average


def mask_embedding_gradients(model, tokenizer: NllbTokenizer) -> None:
    embedding = _input_embedding(model)
    ids = torch.tensor(
        [tokenizer.convert_tokens_to_ids(LANG_TO_NLLB[lang]) for lang in ALL_LANGS],
        device=embedding.weight.device,
        dtype=torch.long,
    )
    embedding.weight.requires_grad_(True)

    def mask(gradient):
        selected = torch.zeros_like(gradient)
        selected[ids] = 1
        return gradient * selected

    embedding.weight.register_hook(mask)


class MultilingualChrFCallback(TrainerCallback):
    def __init__(self, model, tokenizer, dev_data, output_dir: Path, patience: int):
        self.model = model
        self.tokenizer = tokenizer
        self.dev_data = dev_data
        self.output_dir = output_dir
        self.patience = patience
        self.best_score = -float("inf")
        self.no_improvement = 0
        self.log_path = output_dir / "training_log.csv"
        if not self.log_path.exists():
            with self.log_path.open("w", newline="", encoding="utf-8") as file:
                csv.writer(file).writerow(
                    ["step", "train_loss", "val_loss", "avg_chrf"]
                    + [f"{lang}_chrf" for lang in ALL_LANGS]
                )

    def _generate(self, sources: list[str], target_token: str, num_beams: int) -> list[str]:
        device = next(self.model.parameters()).device
        target_id = self.tokenizer.convert_tokens_to_ids(target_token)
        predictions = []
        self.model.eval()
        for start in range(0, len(sources), 16):
            batch = sources[start : start + 16]
            self.tokenizer.src_lang = SRC_LANG
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True, truncation=True,
                max_length=256,
            ).to(device)
            with torch.no_grad():
                generated = self.model.generate(
                    **inputs, forced_bos_token_id=target_id,
                    max_new_tokens=256, num_beams=num_beams,
                )
            predictions.extend(
                text.strip()
                for text in self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            )
        self.model.train()
        if predictions and not any(predictions):
            raise RuntimeError(
                "All generated translations are empty. The run is invalid; "
                "inspect NLLB target-token routing before continuing."
            )
        return predictions

    def evaluate_all(self, num_beams: int, show_samples: bool) -> tuple[dict[str, float], dict[str, list[str]]]:
        metric = sacrebleu.metrics.CHRF(word_order=2)
        scores, predictions_by_language = {}, {}
        for lang, (sources, references) in self.dev_data.items():
            predictions = self._generate(sources, LANG_TO_NLLB[lang], num_beams=num_beams)
            predictions_by_language[lang] = predictions
            scores[lang] = metric.corpus_score(predictions, [references]).score
            if show_samples:
                print(f"\n[Translation samples: Spanish -> {lang.upper()} | beams={num_beams}]")
                for source, reference, prediction in zip(sources[:3], references[:3], predictions[:3]):
                    print(f"  SRC : {source}")
                    print(f"  REF : {reference}")
                    print(f"  PRED: {prediction}")
        return scores, predictions_by_language

    def on_evaluate(self, args, state, control, **kwargs):
        # Greedy decoding is deliberately used for all validation epochs so
        # checkpoint selection follows the original Experiment-A protocol.
        scores, _ = self.evaluate_all(num_beams=1, show_samples=True)

        average = float(np.mean(list(scores.values())))
        val_loss = kwargs.get("metrics", {}).get("eval_loss", float("nan"))
        train_loss = next(
            (item["loss"] for item in reversed(state.log_history) if "loss" in item),
            float("nan"),
        )
        print(f"\nStep {state.global_step} | val_loss={val_loss:.4f} | avg chrF++={average:.2f}")
        print(" | ".join(f"{lang}={scores[lang]:.2f}" for lang in ALL_LANGS))
        with self.log_path.open("a", newline="", encoding="utf-8") as file:
            csv.writer(file).writerow(
                [state.global_step, train_loss, val_loss, average]
                + [scores[lang] for lang in ALL_LANGS]
            )

        if average > self.best_score:
            self.best_score = average
            self.no_improvement = 0
            best = self.output_dir / "best_chrf_checkpoint"
            best.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(str(best))
            self.tokenizer.save_pretrained(str(best))
            torch.save(_input_embedding(self.model).weight.detach().cpu(), best / "embeddings.pt")
            with (best / "metrics.json").open("w", encoding="utf-8") as file:
                json.dump(
                    {"step": state.global_step, "avg_chrf_greedy": average, "scores_greedy": scores},
                    file, indent=2,
                )
            print("Saved new best greedy-chrF++ checkpoint.")
        else:
            self.no_improvement += 1
        if self.no_improvement >= self.patience:
            print(f"Early stopping after {self.patience} evaluations without improvement.")
            control.should_training_stop = True
        return control


def generation_preflight(model, tokenizer: NllbTokenizer, dev_data) -> None:
    """Fail before training if the installed Transformers generation route is broken."""
    device = next(model.parameters()).device
    model.eval()
    print("\nGeneration preflight (one sentence per target language):")
    for lang, (sources, _) in dev_data.items():
        tokenizer.src_lang = SRC_LANG
        inputs = tokenizer(
            [sources[0]], return_tensors="pt", padding=True, truncation=True,
            max_length=256,
        ).to(device)
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                forced_bos_token_id=tokenizer.convert_tokens_to_ids(LANG_TO_NLLB[lang]),
                max_new_tokens=256,
                num_beams=1,
            )
        text = tokenizer.batch_decode(generated, skip_special_tokens=True)[0].strip()
        if not text:
            raise RuntimeError(
                f"Generation preflight produced an empty {lang.upper()} translation. "
                "Do not start this run."
            )
        print(f"  {lang.upper()}: {text}")
    model.train()


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone A-Bridge Q/K/V/O LoRA")
    parser.add_argument("--lora_r", required=True, type=int, choices=(32, 128))
    parser.add_argument("--lora_alpha", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    # 489,034 filtered pairs / effective batch 16 = 30,564 whole optimizer
    # steps per joint pass under the legacy protocol.  20 passes = 611,280.
    parser.add_argument("--max_steps", type=int, default=611280)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=2)
    parser.add_argument("--experiment_name", default="experiment_A_bridge_qkvo_v3")
    args = parser.parse_args()
    if args.batch_size * args.grad_accum != 16:
        parser.error("Require BATCH_SIZE * GRAD_ACCUM = 16 for the registered A-Bridge protocol.")

    root = Path(__file__).resolve().parents[2]
    data_dir = root / "data_in"
    output_dir = root / "models" / args.experiment_name / f"lora_r{args.lora_r}" / f"lr{args.lr:.0e}"
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = NllbTokenizer.from_pretrained(str(root / "models" / "v0" / "tokenizer"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32

    print("=" * 72)
    print("A-BRIDGE V3: STANDALONE JOINT MULTILINGUAL Q/K/V/O LoRA")
    print(f"rank={args.lora_r} alpha={args.lora_alpha or 2 * args.lora_r} | batch={args.batch_size} x {args.grad_accum} = 16")
    print("validation: greedy decoding | final selected checkpoint: beam=4 | full-corpus chrF++")
    print("=" * 72)

    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-1.3B", dtype=dtype).to(device)
    model.resize_token_embeddings(len(tokenizer))
    model = get_peft_model(model, LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha or 2 * args.lora_r,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj"),
        lora_dropout=0.05,
        bias="none",
    ))
    initialise_language_embeddings(model, tokenizer)
    mask_embedding_gradients(model, tokenizer)
    model.print_trainable_parameters()

    datasets = [load_train_language(lang, data_dir, tokenizer) for lang in ALL_LANGS]
    lengths = np.array([len(dataset) for dataset in datasets])
    probabilities = lengths ** 0.7
    probabilities /= probabilities.sum()
    repeats = int(np.ceil((args.max_steps * 16 * probabilities / lengths).max())) + 2
    train_data = interleave_datasets(
        [dataset.to_iterable_dataset().repeat(repeats) for dataset in datasets],
        probabilities=probabilities.tolist(), seed=42,
    )
    dev_data = {lang: load_dev_language(lang, data_dir) for lang in ALL_LANGS}
    generation_preflight(model, tokenizer, dev_data)
    eval_data = concatenate_datasets([
        tokenise_dev_language(lang, src, tgt, tokenizer)
        for lang, (src, tgt) in dev_data.items()
    ])
    steps_per_epoch = max(100, int(lengths.sum() / 16))
    print(f"Sampling probabilities: {dict(zip(ALL_LANGS, np.round(probabilities, 3)))}")
    print(
        f"Steps per joint-data pass: {steps_per_epoch}; evaluation interval: {steps_per_epoch}; "
        f"20 passes: {steps_per_epoch * 20} steps"
    )

    train_args = TrainingArguments(
        output_dir=str(output_dir / "hf"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=0.06,
        lr_scheduler_type="cosine",
        bf16=(dtype == torch.bfloat16),
        fp16=False,
        optim="paged_adamw_8bit",
        weight_decay=0.0,
        max_grad_norm=1.0,
        label_smoothing_factor=0.1,
        eval_strategy="steps",
        eval_steps=steps_per_epoch,
        save_strategy="steps",
        save_steps=steps_per_epoch,
        save_total_limit=1,
        logging_steps=max(10, steps_per_epoch // 30),
        gradient_checkpointing=False,
        ignore_data_skip=True,
        report_to="none",
        seed=42,
        data_seed=42,
    )
    trainer = ABridgeTrainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=eval_data,
        processing_class=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True, label_pad_token_id=-100),
        callbacks=[MultilingualChrFCallback(model, tokenizer, dev_data, output_dir, args.patience)],
    )
    checkpoint = get_last_checkpoint(str(output_dir / "hf"))
    trainer.train(resume_from_checkpoint=checkpoint) if checkpoint else trainer.train()

    # The best epoch was selected strictly with greedy chrF++.  Reload its
    # adapter and language-token embeddings, then perform one beam-4 final
    # evaluation.  This is intentionally outside the validation callback.
    best_dir = output_dir / "best_chrf_checkpoint"
    if not best_dir.exists():
        raise RuntimeError("No greedy best checkpoint was saved; final beam-4 evaluation cannot run.")
    model.load_adapter(str(best_dir), adapter_name="best_final", is_trainable=False)
    model.set_adapter("best_final")
    best_embeddings = torch.load(best_dir / "embeddings.pt", map_location="cpu", weights_only=True)
    with torch.no_grad():
        _input_embedding(model).weight.copy_(best_embeddings.to(_input_embedding(model).weight.device))

    callback = next(item for item in trainer.callback_handler.callbacks if isinstance(item, MultilingualChrFCallback))
    final_scores, final_predictions = callback.evaluate_all(num_beams=4, show_samples=True)
    final_average = float(np.mean(list(final_scores.values())))
    final_path = output_dir / "final_beam4_metrics.json"
    with final_path.open("w", encoding="utf-8") as file:
        json.dump(
            {"selection_metric": "greedy chrF++", "final_decoding": "beam=4", "avg_chrf": final_average,
             "scores": final_scores, "predictions": final_predictions},
            file, ensure_ascii=False, indent=2,
        )
    print(f"\nFINAL beam=4 evaluation of greedy-selected checkpoint: {final_average:.2f} chrF++")


if __name__ == "__main__":
    main()
