#!/usr/bin/env python3
"""Experiment B: isolated CaLoRA-X sequential runner.

CaLoRA-X preserves the stable CaLoRA model/data/evaluation protocol but
replaces legacy single-gradient CaGA with compact representative gradient
subspaces and continuous conflict-gated soft projection.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from datasets import Dataset
from transformers import DataCollatorForSeq2Seq, NllbTokenizer, Seq2SeqTrainingArguments, TrainingArguments, set_seed

import train_calora_sequential as base
from calora_x_utils import (
    GradientSnapshotCollector,
    build_representative_memory,
    load_representative_memory,
    run_toy_projection_check,
    save_representative_memory,
    soft_project_gradient,
)
from nllb_prompt import NLLBForConditionalGenerationWithCaLoRA


def _lora_parameters(model: torch.nn.Module) -> tuple[list[str], list[torch.nn.Parameter]]:
    pairs = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and any(f".lora_{target}." in name for target in ("q", "k", "v", "o"))
    ]
    return [name for name, _ in pairs], [parameter for _, parameter in pairs]


def load_shared_calorax_checkpoint(
    model: NLLBForConditionalGenerationWithCaLoRA,
    cur_task_idx: int,
    order_langs: list[str],
    output_dir: Path,
    device: torch.device,
) -> list[dict[str, Any]]:
    """Restore the shared best adapter and compact CaLoRA-X memories only."""
    previous_lang = order_langs[cur_task_idx - 1]
    previous_dir = output_dir / f"task_{cur_task_idx}_{previous_lang}" / "best_adapter"
    checkpoint = previous_dir / "calora_model.pt"
    print(f"---> Restoring shared best CaLoRA-X adapter from [{previous_lang.upper()}] ({previous_dir.name}/):")
    if not checkpoint.exists():
        raise FileNotFoundError(f"Missing shared CaLoRA-X checkpoint: {checkpoint}")
    state = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(state, strict=True)
    print(f"     ✓ Restored shared LoRA weights and embeddings from [{previous_lang.upper()}]")

    names, parameters = _lora_parameters(model)
    shapes = [tuple(parameter.shape) for parameter in parameters]
    memories = []
    print(f"---> Loading CaLoRA-X representative memory across {cur_task_idx} completed task(s):")
    for task_id in range(cur_task_idx):
        old_lang = order_langs[task_id]
        memory_path = output_dir / f"task_{task_id + 1}_{old_lang}" / "best_adapter" / "representative_memory.pt"
        memory = load_representative_memory(memory_path, names, shapes, device)
        memory["task_id"] = task_id
        memory["lang_code"] = old_lang
        memories.append(memory)
        size_mb = memory.get("storage_bytes", 0) / (1024 ** 2)
        print(
            f"     ✓ Loaded Task {task_id + 1} [{old_lang.upper()}] representative basis "
            f"(rank={memory.get('memory_rank')}, snapshots={memory.get('snapshot_count')}, {size_mb:.1f} MB)."
        )
    return memories


class CaLoRAXTrainer(base.CleanCaLoRATrainer):
    """Native trainer with PaCA retained and CaGA replaced by CaLoRA-X."""

    def __init__(
        self,
        *args,
        previous_memory: Optional[list[dict[str, Any]]] = None,
        x_memory_samples: int = 4,
        x_memory_start_fraction: float = 0.4,
        x_memory_rank: int = 8,
        x_attenuation: float = 0.5,
        x_min_scale: float = 0.1,
        **kwargs,
    ) -> None:
        # ``previous_grad={}`` bypasses legacy CaGA SVD caching in the parent.
        kwargs["previous_grad"] = {}
        super().__init__(*args, **kwargs)
        if not 0.0 <= x_attenuation <= 1.0:
            raise ValueError("--x_attenuation must be in [0, 1].")
        if not 0.0 < x_min_scale <= 1.0:
            raise ValueError("--x_min_scale must be in (0, 1].")
        self.previous_memory = previous_memory or []
        self.x_memory_rank = x_memory_rank
        self.x_attenuation = x_attenuation
        self.x_min_scale = x_min_scale
        self.snapshot_collector = GradientSnapshotCollector(
            self.cur_p_names,
            total_steps=max(1, int(self.args.max_steps)),
            samples=x_memory_samples,
            start_fraction=x_memory_start_fraction,
        )
        for memory in self.previous_memory:
            if len(memory["entries"]) != len(self.cur_p):
                raise ValueError("CaLoRA-X loaded memory tensor count does not match active LoRA tensors.")
        if base.is_main_process():
            print(
                f"---> CaLoRAXTrainer initialized: Tracking {len(self.cur_p)} LoRA tensors across "
                f"{len(self.previous_memory)} compact prior memory/memories | "
                f"snapshots={self.snapshot_collector.schedule} | rank={x_memory_rank} | "
                f"attenuation={x_attenuation} | min_scale={x_min_scale}."
            )

    def build_task_memory(self) -> dict[str, Any]:
        """Compress this task's CPU snapshots after training has completed."""
        device = self.args.device if torch.cuda.is_available() else torch.device("cpu")
        memory = build_representative_memory(
            self.snapshot_collector,
            [tuple(parameter.shape) for parameter in self.cur_p],
            self.x_memory_rank,
            device,
        )
        self.snapshot_collector.clear()
        return memory

    def training_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Apply PaCA then a single CaLoRA-X soft-projection correction per update."""
        model.train()
        inputs = self._prepare_inputs(inputs)
        is_accum_boundary = bool(getattr(self.accelerator, "sync_gradients", True))
        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)
        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1 and not self.is_deepspeed_enabled:
            accumulation_size = getattr(self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps)
            loss = loss / max(1, accumulation_size)

        if hasattr(self, "accelerator") and self.accelerator is not None:
            self.accelerator.backward(loss, retain_graph=is_accum_boundary)
        elif getattr(self, "do_grad_scaling", False) and getattr(self, "scaler", None) is not None:
            self.scaler.scale(loss).backward(retain_graph=is_accum_boundary)
        else:
            loss.backward(retain_graph=is_accum_boundary)

        if not is_accum_boundary:
            return loss.detach()

        paca_active = self.enable_paca_correction and self.state.global_step >= self.paca_warmup_steps
        self.cur_E = base.compute_Et(loss, self.cur_p, self.args.device) if paca_active else []
        # Snapshot the raw accumulated task gradient before either PaCA or
        # historical-memory correction; this makes each task memory comparable.
        self.cur_grad = [
            parameter.grad.detach().clone() if parameter.grad is not None else torch.zeros_like(parameter)
            for parameter in self.cur_p
        ]
        captured = self.snapshot_collector.maybe_capture(self.cur_grad)
        if captured and base.is_main_process():
            print(
                f"[CaLoRA-X Memory] captured snapshot {self.snapshot_collector.captured}/"
                f"{len(self.snapshot_collector.schedule)} at optimizer boundary {self.snapshot_collector.boundary_count}."
            )

        correction_deltas = [torch.zeros_like(gradient) for gradient in self.cur_grad]
        projection_ratios: list[float] = []
        conflicts: list[float] = []
        scales: list[float] = []
        for memory in self.previous_memory:
            for index, gradient in enumerate(self.cur_grad):
                corrected, stats = soft_project_gradient(
                    gradient,
                    memory["entries"][index],
                    attenuation=self.x_attenuation,
                    min_scale=self.x_min_scale,
                )
                correction_deltas[index].add_(corrected - gradient)
                projection_ratios.append(stats["projection_ratio"])
                conflicts.append(stats["conflict"])
                scales.append(stats["mean_scale"])

        with torch.no_grad():
            prior_count = len(self.previous_memory)
            for index, parameter in enumerate(self.cur_p):
                if parameter.grad is None:
                    continue
                corrected_grad = self.cur_grad[index]
                if prior_count:
                    corrected_grad = corrected_grad + correction_deltas[index] / prior_count
                if index < len(self.cur_E):
                    corrected_grad = corrected_grad * self.cur_E[index].detach()
                parameter.grad.copy_(corrected_grad)

        if base.is_main_process() and (self.state.global_step == 0 or self.state.global_step % 100 == 0):
            if self.cur_E:
                mean_paca = sum(scale.float().mean().item() for scale in self.cur_E) / len(self.cur_E)
                print(f"[PaCA] step={self.state.global_step} active | scale mean={mean_paca:.4f}")
            else:
                print(f"[PaCA] step={self.state.global_step} inactive | raw LoRA gradients retained")
            if projection_ratios:
                print(
                    f"[CaLoRA-X] step={self.state.global_step} active | prior_tasks={len(self.previous_memory)} | "
                    f"mean projection ratio={sum(projection_ratios) / len(projection_ratios):.4f} | "
                    f"mean conflict={sum(conflicts) / len(conflicts):.4f} | "
                    f"mean soft scale={sum(scales) / len(scales):.4f}"
                )
        return loss.detach()


def _resolve_tokenizer_and_model_path() -> tuple[str, str]:
    tokenizer_path = base.MODEL_NAME
    for candidate in [
        base.project_root / "models/v0/tokenizer", Path("models/v0/tokenizer"), Path("/workspace/models/v0/tokenizer"),
    ]:
        if candidate.exists() and (candidate / "tokenizer_config.json").exists():
            tokenizer_path = str(candidate)
            break
    model_path = base.MODEL_NAME
    for candidate in [
        base.project_root / "models/nllb-200-1.3B", Path("models/nllb-200-1.3B"), Path("/workspace/models/nllb-200-1.3B"),
    ]:
        if candidate.exists() and (candidate / "config.json").exists():
            model_path = str(candidate)
            break
    return tokenizer_path, model_path


def _print_report(eval_matrix: dict[str, dict[str, float]], order_langs: list[str], lora_rank: int, order: str) -> None:
    print("\n" + "=" * 70)
    print(f" EXPERIMENT B (SEQUENTIAL CaLoRA-X r={lora_rank} | {order.upper()}) FINAL REPORT")
    print("=" * 70)
    header = "Task Trained \\ Dev Eval | " + " | ".join(f"{lang.upper():10s}" for lang in order_langs)
    print(header)
    print("-" * len(header))
    for trained_lang in order_langs:
        if trained_lang in eval_matrix:
            print(f"{trained_lang.upper():21s} | " + " | ".join(
                f"{str(eval_matrix[trained_lang].get(eval_lang, '---')):10s}" for eval_lang in order_langs
            ) + " | ")
    final_lang = order_langs[-1]
    if final_lang not in eval_matrix:
        return
    final_scores = eval_matrix[final_lang]
    aa = sum(final_scores.values()) / len(final_scores)
    forgetting, bwt = {}, {}
    for language in order_langs[:-1]:
        maximum = max(row.get(language, 0.0) for row in eval_matrix.values())
        diagonal = eval_matrix.get(language, {}).get(language, 0.0)
        final = final_scores.get(language, 0.0)
        forgetting[language] = round(maximum - final, 2)
        bwt[language] = round(final - diagonal, 2)
    print(f"\n---> Average Accuracy (AA) across all 5 languages: {aa:.2f} chrF++")
    print(f"---> Mean Forgetting Measure (FM): {sum(forgetting.values()) / len(forgetting):.2f} chrF++")
    print(f"---> Mean Backward Transfer (BWT): {sum(bwt.values()) / len(bwt):.2f} chrF++")
    print(f"     Per-language Forgetting: {forgetting}")
    print(f"     Per-language BWT:       {bwt}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sequential CaLoRA-X (r=128) for Experiment B")
    parser.add_argument("--order", choices=["order1", "order2"], default="order1")
    parser.add_argument("--lora_r", type=int, default=128)
    parser.add_argument("--lora_alpha", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs_per_task", type=int, default=12)
    parser.add_argument("--dynamic_epochs", action="store_true", default=True)
    parser.add_argument("--static_epochs", action="store_true", default=False)
    parser.add_argument("--batch_size", type=int, default=6)
    parser.add_argument("--grad_accum", type=int, default=20)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--experiment_name", default="experiment_B")
    parser.add_argument("--run_tag", default=None)
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--start_task", type=int, default=1)
    parser.add_argument("--end_task", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--disable_paca_correction", action="store_true")
    parser.add_argument("--paca_warmup_steps", type=int, default=0)
    parser.add_argument("--x_memory_samples", type=int, default=4)
    parser.add_argument("--x_memory_start_fraction", type=float, default=0.4)
    parser.add_argument("--x_memory_rank", type=int, default=8)
    parser.add_argument("--x_attenuation", type=float, default=0.5)
    parser.add_argument("--x_min_scale", type=float, default=0.1)
    args = parser.parse_args()
    order_langs = base.ORDERS[args.order]
    if not 1 <= args.start_task <= args.end_task <= len(order_langs):
        parser.error("Require 1 <= --start_task <= --end_task <= 5.")
    run_toy_projection_check()
    set_seed(args.seed)
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16
    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)
    dynamic = args.dynamic_epochs and not args.static_epochs
    data_dir, base_output = base.resolve_paths(args)
    output_dir = base_output / args.experiment_name / f"calora_x_sequential_r{args.lora_r}" / args.order / f"lr{args.lr:.0e}".replace("e-0", "e-")
    if args.run_tag:
        output_dir = output_dir / args.run_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "calorax_config.json").write_text(
        json.dumps(vars(args), indent=2, sort_keys=True), encoding="utf-8"
    )
    effective_batch = args.batch_size * args.grad_accum
    print("=" * 70)
    print(" EXPERIMENT B: SEQUENTIAL CaLoRA-X CONTINUAL LEARNING")
    print(f" Order: {args.order.upper()} -> {' -> '.join(order_langs)}")
    print(f" Output: {output_dir}")
    print(f" Q/K/V/O LoRA r={args.lora_r}; effective batch={effective_batch}; seed={args.seed}")
    print(f" Representative memory: {args.x_memory_samples} snapshots after {args.x_memory_start_fraction:.0%}, rank={args.x_memory_rank}")
    print(f" Soft projection: attenuation={args.x_attenuation}, min_scale={args.x_min_scale}; PaCA={'OFF' if args.disable_paca_correction else 'ON'}")
    print("=" * 70)

    tokenizer_path, model_path = _resolve_tokenizer_and_model_path()
    tokenizer = NllbTokenizer.from_pretrained(tokenizer_path, src_lang=base.SRC_LANG)
    missing_tokens = [base.LANG_TO_NLLB[lang] for lang in base.NEW_LANGUAGES if tokenizer.convert_tokens_to_ids(base.LANG_TO_NLLB[lang]) in (None, tokenizer.unk_token_id)]
    if missing_tokens:
        tokenizer.add_special_tokens({"additional_special_tokens": missing_tokens})
    dev_data = {lang: {"src": pair[0], "tgt": pair[1]} for lang in order_langs if (pair := base.load_dev_raw(lang, data_dir))}
    matrix_path = output_dir / "triangular_eval_matrix.json"
    eval_matrix: dict[str, dict[str, float]] = {}
    if args.start_task > 1:
        if not matrix_path.exists():
            raise FileNotFoundError(f"Cannot resume without {matrix_path}.")
        eval_matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
        missing_rows = [lang for lang in order_langs[:args.start_task - 1] if lang not in eval_matrix]
        if missing_rows:
            raise ValueError(f"Cannot resume: missing triangular rows {missing_rows}.")

    for task_idx, lang_code in enumerate(order_langs):
        if task_idx + 1 < args.start_task:
            continue
        if task_idx + 1 > args.end_task:
            break
        print("\n" + "█" * 70)
        print(f" TASK {task_idx + 1}/5: TRAINING CaLoRA-X ON [{lang_code.upper()}]")
        print("█" * 70)
        prompt_config = {"task_id": 0, "lora_r": args.lora_r, "lora_alpha": args.lora_alpha, "lora_dropout": 0.05, "trans_hidden_dim": 100, "attn_temperature": 1.0, "previous_lora_path": None, "previous_prompt_key_path": None}
        model = NLLBForConditionalGenerationWithCaLoRA(model_path, prompt_config, torch_dtype=load_dtype)
        base.initialize_new_language_embeddings(model, tokenizer)
        previous_memory = load_shared_calorax_checkpoint(model, task_idx, order_langs, output_dir, device) if task_idx else []
        base.apply_embedding_masking(model, tokenizer, order_langs[:task_idx + 1])
        model.to(device)
        train_ds = base.load_single_language_dataset(lang_code, data_dir, tokenizer, max_length=256)
        if not train_ds:
            raise FileNotFoundError(f"Missing training data for {lang_code}.")
        dev_pair = base.load_dev_raw(lang_code, data_dir)
        dev_ds = None
        if dev_pair:
            raw = Dataset.from_dict({"src": dev_pair[0], "tgt": dev_pair[1]})
            def tokenize_dev(batch):
                tokenizer.src_lang = base.SRC_LANG
                tokenizer.tgt_lang = base.LANG_TO_NLLB[lang_code]
                return tokenizer(batch["src"], text_target=batch["tgt"], truncation=True, max_length=256, padding=False)
            dev_ds = raw.map(tokenize_dev, batched=True, batch_size=1024, remove_columns=["src", "tgt"], keep_in_memory=True)
        epochs = base.DYNAMIC_EPOCHS_MAP.get(lang_code, args.epochs_per_task) if dynamic else args.epochs_per_task
        steps_per_epoch = max(1, len(train_ds) // effective_batch)
        max_steps = steps_per_epoch * epochs
        print(f"Task dataset size: {len(train_ds):,} | Epochs: {epochs} | Optimizer steps: {max_steps}")
        task_dir = output_dir / f"task_{task_idx + 1}_{lang_code}"
        task_dir.mkdir(parents=True, exist_ok=True)
        has_eval = dev_ds is not None and len(dev_ds) > 0
        training_args = Seq2SeqTrainingArguments(
            output_dir=str(task_dir / "hf"), max_steps=max_steps, per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum, learning_rate=args.lr, warmup_ratio=0.06,
            lr_scheduler_type="cosine", eval_strategy="epoch" if has_eval else "no", save_strategy="no",
            logging_steps=max(5, steps_per_epoch // 4), predict_with_generate=True, generation_num_beams=args.num_beams,
            bf16=use_bf16, fp16=use_fp16, optim="adamw_torch", max_grad_norm=1.0, label_smoothing_factor=0.1,
            weight_decay=0.0, ignore_data_skip=True, group_by_length=False, dataloader_num_workers=4 if os.name != "nt" else 0,
            dataloader_pin_memory=True, dataloader_persistent_workers=os.name != "nt", dataloader_prefetch_factor=2 if os.name != "nt" else None,
            tf32=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8, load_best_model_at_end=False, ddp_find_unused_parameters=False,
        )
        # Reuse the verified full-corpus evaluation callback, but intentionally
        # leave ``trainer`` unset so it does not save legacy lora_grad.pt files.
        task_logger = base.SequentialCaLoRATaskLogCallback(model, tokenizer, lang_code, task_idx + 1, dev_pair, output_dir, task_dir, gen_batch_size=16, num_beams=args.num_beams)
        trainer = CaLoRAXTrainer(
            model=model, args=training_args, train_dataset=train_ds, eval_dataset=dev_ds if has_eval else None,
            processing_class=tokenizer, tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100), callbacks=[task_logger],
            cur_task_id=task_idx, task_order=order_langs[:task_idx + 1], previous_memory=previous_memory,
            enable_paca_correction=not args.disable_paca_correction, paca_warmup_steps=args.paca_warmup_steps,
            x_memory_samples=args.x_memory_samples, x_memory_start_fraction=args.x_memory_start_fraction,
            x_memory_rank=args.x_memory_rank, x_attenuation=args.x_attenuation, x_min_scale=args.x_min_scale,
        )
        trainer.train()
        print("---> Compressing this task's representative gradient memory (one tensor at a time)...")
        memory = trainer.build_task_memory()
        real_model = getattr(model, "module", model)
        best_dir, last_dir = task_dir / "best_adapter", task_dir / "last_adapter"
        for destination in (best_dir, last_dir):
            destination.mkdir(parents=True, exist_ok=True)
            if destination == last_dir or not (destination / "calora_model.pt").exists():
                torch.save(real_model.state_dict(), destination / "calora_model.pt")
                tokenizer.save_pretrained(destination)
                torch.save(base.get_emb_layer(real_model).weight.data, destination / "embeddings.pt")
            save_representative_memory(memory, destination / "representative_memory.pt")
        print(f"     ✓ Saved rank-{memory['memory_rank']} representative memory ({memory['storage_bytes'] / 1024 ** 2:.1f} MB) to best_adapter/ and last_adapter/.")

        state = torch.load(best_dir / "calora_model.pt", map_location=device, weights_only=True)
        real_model.load_state_dict(state, strict=True)
        embedding_file = best_dir / "embeddings.pt"
        if embedding_file.exists():
            base.get_emb_layer(real_model).weight.data.copy_(torch.load(embedding_file, map_location=device, weights_only=True))
        print(f"\n[TRIANGULAR EVALUATION] after Task {task_idx + 1} ({lang_code.upper()})")
        eval_matrix[lang_code] = {}
        for old_lang in order_langs[:task_idx + 1]:
            if old_lang in dev_data:
                score = base.evaluate_language_chrf(model, tokenizer, dev_data[old_lang], old_lang, device, num_beams=args.num_beams, batch_size=16, target_task_idx=None)
                eval_matrix[lang_code][old_lang] = round(score, 2)
                print(f"   Dev [{old_lang.upper()}]: chrF++ = {score:.2f}")
        matrix_path.write_text(json.dumps(eval_matrix, indent=2, ensure_ascii=False), encoding="utf-8")
        del trainer, train_ds, model, previous_memory, memory
        if dev_ds is not None:
            del dev_ds
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    _print_report(eval_matrix, order_langs, args.lora_r, args.order)


if __name__ == "__main__":
    main()
