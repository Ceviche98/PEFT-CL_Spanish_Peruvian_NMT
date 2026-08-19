#!/usr/bin/env python3
"""
Sequential CaLoRA (Causal-Aware LoRA) Continual Learning Script for NLLB-200-1.3B (Experiment B)
================================================================================================
Trains CaLoRA adapters sequentially across the 5 Peruvian Indigenous languages without data replay
to measure catastrophic forgetting (FM), average accuracy (AA), and backward transfer (BWT) using
the Triangular Evaluation Matrix protocol.

Supports both experimental orders defined in Section 7.1 of Protocol:
  - order1 (Andino -> Amazónico / High to Low): quy -> ayr -> shp -> agr -> cni
  - order2 (Amazónico -> Andino / Low to High): cni -> agr -> shp -> ayr -> quy

Key Features, Compatibility & Top NLP Researcher Engineering:
  - Pure Transformers Native Trainer Loop (Bypasses CaLora_Trainer & vendored 2023 _inner_training_loop):
    Inherits directly from `transformers.Seq2SeqTrainer`, eliminating any dependence on deprecated/removed
    symbols (`pythonShardedDDPOption`, `smp_forward_backward`, `IS_SAGEMAKER_MP_POST_1_10`, `attn_lr`,
    `data_replay_freq`) and avoiding double `optimizer.step()` / double clipping crashes.
  - Zero Source File Modification (`calora_utils.py` separation):
    Imports pure CaLoRA mathematics (`compute_Et`, `lora_project_svd`, `vector_or_matrix_cosine_similarity`)
    from `calora_utils.py`, bypassing original `cl_collator.py` `Optional` / `dataclass` import crashes.
  - Causal-Aware Adaptation (CaLoRA):
    Implements PaCA (Parameter-level Counterfactual Attribution) via Taylor approximation and
    CaGA (Cross-task Gradient Adaptation) via SVD projection onto previous task gradient spaces.
  - Explicit Parameter Segregation inside `CleanCaLoRATrainer`:
    Excludes base vocabulary embeddings (`embed_tokens`, `shared`, `lm_head`) from the PaCA Hessian
    and CaGA SVD computations to maintain exact linear complexity while allowing selective embedding
    masking to update new target language tokens (`shp_Latn`, `agr_Latn`, `cni_Latn`).
  - Strict Protocol Section 7.3 Compliance (`apply_embedding_masking`):
    Unlocks gradient susceptibility specifically for seen target languages up to the current task
    (`order_langs[:task_idx+1]`) rather than globally across all future tokens.
  - Strict Weight Decay 0.0 & Cosine Scheduler:
    Enforces `weight_decay=0.0` as mandated by CaLoRA technical specifications (`calora_technical_documentation.md`)
    to guarantee zero passive erosion of base language tokens.
  - Shared-adapter checkpoint propagation (`load_shared_calora_checkpoint`):
    Restores the preceding best adapter into the current trainable LoRA modules and loads all older
    gradient snapshots only as CaGA memory. This makes BWT measurable on the shared adapter.
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional, Dict, Union, Any

# CRITICAL VAST.AI / LOCAL FIX: Force HuggingFace cache to /workspace volume if present before hub imports
if os.path.exists("/workspace"):
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import sacrebleu
from datasets import Dataset

from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    NllbTokenizer,
    TrainingArguments,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    Seq2SeqTrainer,
    set_seed,
)

# Add paths for CaLoRA modules and prompt models
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(Path(__file__).resolve().parent))

try:
    from calora_utils import (
        compute_Et,
        build_lora_svd_projection_basis,
        lora_project_with_cached_basis,
        vector_or_matrix_cosine_similarity,
    )
    from nllb_prompt import (
        NLLBForConditionalGenerationWithCaLoRA,
        CustomM2M100Attention,
        CaLoRALinear,
    )
except ImportError as e:
    raise ImportError(f"Could not import CaLoRA modules: {e}. Ensure calora_utils.py and nllb_prompt.py are inside scripts/experiment_B.")

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
            # M2M100ScaledWordEmbedding already applies ``embed_scale`` internally.
            # Applying sqrt(d_model) here again made decoder inputs 32x too large
            # for NLLB-1.3B (d_model=1024), destabilising Q/V-only CaLoRA.
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
            # ``embed_tokens`` is M2M100ScaledWordEmbedding, so it returns
            # correctly scaled vectors without an additional multiplication.
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
    "cni": 15,   # 3.8k pairs -> 15 epochs = ~450 steps (ensures morphological convergence on Asháninka)
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


def log_gpu_memory(prefix: str = "", task_idx: int = 0, lang_code: str = "", csv_path: Optional[Path] = None):
    """Logs current and peak GPU VRAM allocation to stdout and optionally saves to a CSV file."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated(0) / (1024**3)
        reserved = torch.cuda.memory_reserved(0) / (1024**3)
        max_allocated = torch.cuda.max_memory_allocated(0) / (1024**3)
        total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        pct = (allocated / total) * 100
        print(f"[GPU VRAM {prefix}] Allocated: {allocated:.2f} GB | Reserved (Cache): {reserved:.2f} GB | Max Peak: {max_allocated:.2f} GB | Total: {total:.2f} GB ({pct:.1f}%)")
        
        if csv_path is not None and is_main_process():
            csv_path.parent.mkdir(parents=True, exist_ok=True)
            write_header = not csv_path.exists()
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                if write_header:
                    writer.writerow(["task_idx", "lang_code", "stage", "allocated_gb", "reserved_gb", "max_allocated_gb", "total_gb", "pct_allocated"])
                writer.writerow([task_idx, lang_code, prefix, round(allocated, 3), round(reserved, 3), round(max_allocated, 3), round(total, 3), round(pct, 2)])


# ═════════════════════════════════════════════════════════════════════════
# CUSTOM NATIVE TRANSFORMERS CaLoRA TRAINER (Inherits from Seq2SeqTrainer)
# ═════════════════════════════════════════════════════════════════════════
class CleanCaLoRATrainer(Seq2SeqTrainer):
    """
    Native `transformers.Seq2SeqTrainer` subclass that:
      1. Bypasses CaLora_Trainer entirely, allowing `Trainer.train()` to run the modern, native,
         and fully supported `_inner_training_loop` of installed transformers (multi-GPU, DDP, AMP).
      2. Isolates `self.cur_p` strictly to Q/K/V/O LoRA factors (`lora_A`, `lora_B`), preventing massive vocabulary
         matrices and unused routing parameters from entering PaCA/CaGA calculations.
      3. In `training_step`: executes `loss.backward()`, calculates PaCA (`cur_E`) and CaGA (`proj`) specifically
         for `self.cur_p`, and returns `loss.detach()` without calling `optimizer.step()` or `zero_grad()`, so native
         `_inner_training_loop` handles multi-step accumulation (`--grad_accum`) and clipping cleanly without collisions.
    """
    def __init__(
        self,
        model: nn.Module,
        args: TrainingArguments,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Dataset] = None,
        tokenizer: Optional[NllbTokenizer] = None,
        processing_class: Optional[Any] = None,
        data_collator: Optional[Any] = None,
        compute_metrics: Optional[Any] = None,
        callbacks: Optional[list] = None,
        cur_task_id: int = 0,
        task_order: Optional[list[str]] = None,
        previous_grad: Optional[Dict[int, list[torch.Tensor]]] = None,
        enable_paca_correction: bool = True,
        paca_warmup_steps: int = 0,
        **kwargs
    ):
        proc_class = processing_class or tokenizer
        if "processing_class" in kwargs:
            proc_class = kwargs.pop("processing_class") or proc_class
        if not hasattr(args, "generation_config"):
            args.generation_config = getattr(model, "generation_config", None)
        if not hasattr(args, "predict_with_generate"):
            args.predict_with_generate = False
        if not hasattr(args, "generation_max_length"):
            args.generation_max_length = None
        if not hasattr(args, "generation_num_beams"):
            args.generation_num_beams = None
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=proc_class,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            **kwargs
        )
        self.cur_task_id = cur_task_id
        self.pre_task_num = cur_task_id
        self.task_order = task_order or []
        self.pre_grad = previous_grad or {}
        self.pre_grad_bases = {}
        self.enable_paca_correction = enable_paca_correction
        self.paca_warmup_steps = max(0, paca_warmup_steps)

        # PaCA/CaGA are defined for the LoRA factors.  The routing MLP/key is not
        # consumed by the attention blocks during supervised current-task training
        # (routing is deliberately bypassed there), so including it creates a list
        # of permanently zero gradients and corrupts saved gradient alignment.
        self.cur_p = []
        self.cur_p_names = []
        for n, p in self.model.named_parameters():
            if p.requires_grad and any(f".lora_{target}." in n for target in ("q", "k", "v", "o")):
                self.cur_p.append(p)
                self.cur_p_names.append(n)

        # CaGA historically re-ran an SVD for every stored task gradient on
        # every optimizer step.  Those old gradients are immutable, so cache
        # their orthonormal bases once.  The subsequent projection U(U^T g) is
        # algebraically identical to the previous repeated-SVD computation.
        if self.pre_task_num > 0 and self.pre_grad:
            if is_main_process():
                print(f"---> Building cached CaGA SVD bases for {self.pre_task_num} prior task(s)...")
            for task_id, gradients in self.pre_grad.items():
                if len(gradients) != len(self.cur_p):
                    raise ValueError(
                        f"Cannot cache CaGA memory for Task {task_id + 1}: expected {len(self.cur_p)} tensors, "
                        f"found {len(gradients)}."
                    )
                self.pre_grad_bases[task_id] = [
                    build_lora_svd_projection_basis(gradient, self.args.device)
                    for gradient in gradients
                ]
            if is_main_process():
                print("     -> Cached CaGA SVD bases are ready; optimizer steps reuse them without refactorisation.")
        
        if is_main_process():
            paca_status = "enabled" if self.enable_paca_correction else "DISABLED (LoRA-equivalent ablation)"
            print(
                f"---> CleanCaLoRATrainer initialized: Tracking {len(self.cur_p)} LoRA tensors "
                f"across {self.pre_task_num} prior tasks | PaCA: {paca_status} "
                f"| warm-up: {self.paca_warmup_steps} optimizer steps."
            )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels", None)
        inputs.pop("decoder_input_ids", None)

        if labels is None:
            return super().compute_loss(model, inputs, return_outputs=return_outputs, num_items_in_batch=num_items_in_batch)

        real_model = getattr(model, "module", model)
        # Locate M2M100 inner model whether wrapped in NLLBForConditionalGenerationWithCaLoRA or DDP
        m2m_inner = real_model.model if hasattr(real_model, "model") else real_model
        cfg = getattr(m2m_inner, "config", None)
        if cfg is None:
            cfg = getattr(real_model, "config", None)

        shifted = labels.new_zeros(labels.shape)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0] = cfg.decoder_start_token_id
        shifted.masked_fill_(shifted == -100, cfg.pad_token_id)

        emb_layer = get_emb_layer(real_model)
        model_dtype = next(model.parameters()).dtype
        # As above, the M2M/NLLB shared embedding is already scaled internally.
        dec_embeds = emb_layer(shifted).to(dtype=model_dtype)

        outputs = model(
            input_ids=inputs.get("input_ids"),
            attention_mask=inputs.get("attention_mask"),
            decoder_inputs_embeds=dec_embeds,
            labels=labels,
        )
        inputs["labels"] = labels
        del dec_embeds, shifted

        if self.label_smoother is not None:
            # ``shifted`` is already the decoder input (BOS, y_0, ...).  Seq2seq
            # logits at position t must therefore be compared to labels at t;
            # asking LabelSmoother to shift again trained an off-by-one objective.
            loss = self.label_smoother(outputs, labels, shift_labels=False)
        else:
            loss = outputs.loss if isinstance(outputs, dict) or hasattr(outputs, "loss") else outputs[0]

        if return_outputs:
            return loss, outputs
        return loss

    def training_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        num_items_in_batch: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Custom training step incorporating exact PaCA counterfactual attribution and CaGA SVD gradient correction.
        Executes correction precisely at gradient accumulation step boundaries to eliminate compounding micro-step multiplication
        and reduce SVD / Hessian computation overhead by gradient_accumulation_steps fold.
        Returns `loss.detach()` so native `_inner_training_loop` manages clean optimizer steps.
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        # Transformers sets Accelerator.sync_gradients immediately before it
        # calls training_step.  This is the sole reliable accumulation-boundary
        # signal: it also covers a short final accumulation group at the end of
        # each epoch.  A private global micro-step counter drifts whenever an
        # epoch has a non-divisible number of batches, causing PaCA/CaGA to be
        # applied before or after the optimiser step.
        is_accum_boundary = bool(getattr(self.accelerator, "sync_gradients", True))

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs, num_items_in_batch=num_items_in_batch)

        if self.args.n_gpu > 1:
            loss = loss.mean()
        if self.args.gradient_accumulation_steps > 1 and not self.is_deepspeed_enabled:
            # Match modern Trainer semantics, including a short last accumulation
            # group.  For CNI (388 batches at 10 examples/batch, GAS=12), the
            # final group contains four batches and must be divided by four.
            accumulation_size = getattr(
                self, "current_gradient_accumulation_steps", self.args.gradient_accumulation_steps
            )
            loss = loss / max(1, accumulation_size)

        if hasattr(self, "accelerator") and self.accelerator is not None:
            self.accelerator.backward(loss, retain_graph=is_accum_boundary)
        elif getattr(self, "do_grad_scaling", False) and hasattr(self, "scaler") and self.scaler is not None:
            self.scaler.scale(loss).backward(retain_graph=is_accum_boundary)
        else:
            loss.backward(retain_graph=is_accum_boundary)

        # Only compute and apply PaCA / CaGA correction on the accumulation step boundary across the fully accumulated gradient
        if is_accum_boundary:
            # 1. PaCA: optionally compute mean-one, parameter-level attribution scales.
            # The off mode is a clean control: it retains this model wrapper, data
            # pipeline, masking and LoRA modules, but applies raw adapter gradients.
            paca_active = self.enable_paca_correction and self.state.global_step >= self.paca_warmup_steps
            self.cur_E = compute_Et(loss, self.cur_p, self.args.device) if paca_active else []
            self.cur_grad = [p.grad.clone().detach().to(self.args.device) if p.grad is not None else torch.zeros_like(p) for p in self.cur_p]
            len_cur_grad = len(self.cur_grad)

            if is_main_process() and (self.state.global_step == 0 or self.state.global_step % 100 == 0):
                if self.cur_E:
                    scale_min = min(scale.min().item() for scale in self.cur_E)
                    scale_max = max(scale.max().item() for scale in self.cur_E)
                    scale_mean = sum(scale.float().mean().item() for scale in self.cur_E) / len(self.cur_E)
                    print(f"[PaCA] step={self.state.global_step} active | scale mean={scale_mean:.4f}, min={scale_min:.4e}, max={scale_max:.4e}")
                else:
                    print(f"[PaCA] step={self.state.global_step} inactive | raw LoRA gradients retained")

            # 2. CaGA: Cross-task gradient projection against prior tasks' gradient subspaces
            if self.pre_task_num > 0 and self.pre_grad is not None and len(self.pre_grad) > 0:
                self.proj = [torch.zeros_like(p, device=self.args.device) for p in self.cur_p]
                correlation_sum = torch.zeros((), device=self.args.device)
                correlation_terms = 0
                for i in range(self.pre_task_num):
                    if i in self.pre_grad and len(self.pre_grad[i]) == len_cur_grad:
                        for l in range(len_cur_grad):
                            cur_prj_old, correlation = lora_project_with_cached_basis(
                                self.pre_grad_bases[i][l], self.cur_grad[l], self.args.device
                            )
                            sign_proj = vector_or_matrix_cosine_similarity(self.pre_grad[i][l], cur_prj_old, self.args.device)
                            self.proj[l] += correlation * sign_proj
                            correlation_sum += correlation.detach()
                            correlation_terms += 1
                self.proj = [p / self.pre_task_num for p in self.proj]
                if is_main_process() and (self.state.global_step == 0 or self.state.global_step % 100 == 0):
                    mean_abs_projection = torch.stack(
                        [projection.float().abs().mean() for projection in self.proj]
                    ).mean().item()
                    mean_correlation = (correlation_sum / max(1, correlation_terms)).item()
                    print(
                        f"[CaGA] step={self.state.global_step} active | prior_tasks={self.pre_task_num} "
                        f"| mean projection ratio={mean_correlation:.4f} "
                        f"| mean |correction|={mean_abs_projection:.4e}"
                    )
            else:
                self.proj = []

            # 3. Apply gradient correction term G* strictly to isolated CaLoRA parameter tensors on accumulated gradients
            with torch.no_grad():
                for idx, p in enumerate(self.cur_p):
                    if p.grad is not None:
                        correction = 1.0
                        if self.pre_task_num > 0 and idx < len(self.proj):
                            correction = correction * (1 + self.proj[idx].detach())
                        if idx < len(self.cur_E):
                            correction = correction * self.cur_E[idx].detach()
                        p.grad.data = p.grad.data * correction

        # NOTE: Do not call self.optimizer.step(), self.lr_scheduler.step(), or model.zero_grad() here!
        # Native Seq2SeqTrainer._inner_training_loop handles gradient clipping and stepping right after this boundary step.
        return loss.detach()


# ═════════════════════════════════════════════════════════════════════════
# EMBEDDING LAYER RESOLUTION & SELECTIVE MASKING
# ═════════════════════════════════════════════════════════════════════════
def get_emb_layer(model: nn.Module) -> nn.Module:
    """Safely extracts input token embedding `nn.Embedding` layer across NLLBForConditionalGenerationWithCaLoRA or M2M100."""
    if hasattr(model, "get_input_embeddings"):
        return model.get_input_embeddings()
    elif hasattr(model, "model") and hasattr(model.model, "get_input_embeddings"):
        return model.model.get_input_embeddings()
    elif hasattr(model, "model") and hasattr(model.model, "model") and hasattr(model.model.model.encoder, "embed_tokens"):
        return model.model.model.encoder.embed_tokens
    else:
        raise AttributeError("Could not locate input embeddings layer on model.")


def initialize_new_language_embeddings(model: nn.Module, tokenizer: NllbTokenizer):
    """Initializes target rows (shp_Latn, agr_Latn, cni_Latn) as average of Quechua + Aymara."""
    print("\n---> Initializing target embeddings for new languages (shp, agr, cni) using (Quechua + Aymara)/2...")
    if hasattr(model, "resize_token_embeddings"):
        model.resize_token_embeddings(len(tokenizer))
    elif hasattr(model, "model") and hasattr(model.model, "resize_token_embeddings"):
        model.model.resize_token_embeddings(len(tokenizer))

    emb_layer = get_emb_layer(model)
    quy_id = tokenizer.convert_tokens_to_ids(LANG_TO_NLLB["quy"])
    ayr_id = tokenizer.convert_tokens_to_ids(LANG_TO_NLLB["ayr"])

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


def apply_embedding_masking(model: nn.Module, tokenizer: NllbTokenizer, active_langs: list[str]):
    """
    Applies selective embedding gradient masking for seen/active target languages only (Protocol Section 7.3).
    Enforces `requires_grad = True` on embeddings and registers a backward hook so ONLY the active seen
    target language tokens receive gradient updates while keeping future tokens inactive.
    """
    print(f"\n---> Applying selective embedding gradient mask for active seen language tokens: {active_langs}...")
    target_ids = [tokenizer.convert_tokens_to_ids(LANG_TO_NLLB[lang]) for lang in active_langs]

    emb_layer = get_emb_layer(model)
    emb_layer.weight.requires_grad = True # Enforce gradient susceptibility

    target_ids_tensor = torch.tensor(target_ids, dtype=torch.long, device=emb_layer.weight.device)

    def hook(grad):
        mask = torch.zeros_like(grad)
        mask[target_ids_tensor] = 1.0
        return grad * mask

    emb_layer.weight.register_hook(hook)
    print(f"     -> Gradient hook registered for active token IDs: {target_ids}")


# ═════════════════════════════════════════════════════════════════════════
# STATE PROPAGATION ACROSS SEQUENTIAL TASKS
# ═════════════════════════════════════════════════════════════════════════
def _legacy_load_task_indexed_calora_checkpoints(
    model: NLLBForConditionalGenerationWithCaLoRA,
    cur_task_idx: int,
    order_langs: list[str],
    output_dir: Path,
    device: torch.device
) -> Dict[int, list[torch.Tensor]]:
    """
    Propagates optimal checkpoint state when starting task `cur_task_idx` (where cur_task_idx > 0):
      1. Loads token embeddings (`embeddings.pt`) and prompt routing MLP (`trans_input`) from immediately preceding task `cur_task_idx - 1`.
      2. Populates frozen prior tasks' Q/K/V/O LoRA weights into their corresponding
         `previous_lora_weights_*[i]` modules.
      3. Populates frozen prior tasks' prompt routing keys (`prompt_key`) into row `i` of `previous_prompts_keys`.
      4. Loads prior tasks' CaLoRA parameter gradients (`lora_grad.pt`) into `previous_grad[i]` for CaGA SVD cross-task projection.
    """
    previous_grad = {}
    
    # 1. Load embeddings and trans_input from immediately preceding task
    prev_lang_immediate = order_langs[cur_task_idx - 1]
    prev_best_dir_immediate = output_dir / f"task_{cur_task_idx}_{prev_lang_immediate}" / "best_adapter"
    print(f"---> Propagating latest state from immediately preceding task [{prev_lang_immediate.upper()}] ({prev_best_dir_immediate.name}/):")
    
    emb_file = prev_best_dir_immediate / "embeddings.pt"
    if emb_file.exists():
        emb_data = torch.load(emb_file, map_location=device, weights_only=True)
        emb_layer = get_emb_layer(model)
        emb_layer.weight.data.copy_(emb_data)
        if hasattr(model.model.model, "shared"):
            model.model.model.shared.weight.data.copy_(emb_data)
        print(f"     ✓ Loaded embeddings.pt from [{prev_lang_immediate.upper()}]")
    else:
        print(f"     WARNING: Could not find embeddings.pt in {prev_best_dir_immediate}!")

    model_file_immediate = prev_best_dir_immediate / "calora_model.pt"
    if model_file_immediate.exists():
        state_immediate = torch.load(model_file_immediate, map_location=device, weights_only=True)
        trans_input_state = {k.replace("trans_input.", ""): v for k, v in state_immediate.items() if k.startswith("trans_input.")}
        if trans_input_state:
            model.trans_input.load_state_dict(trans_input_state)
            print(f"     ✓ Loaded prompt routing MLP (`trans_input`) from [{prev_lang_immediate.upper()}]")
    
    # 2. Loop through all prior tasks i = 0 .. cur_task_idx - 1 to load frozen CaLoRA blocks, keys, and gradients
    print(f"---> Loading frozen CaLoRA weights and gradients across {cur_task_idx} prior tasks:")
    for i in range(cur_task_idx):
        prev_lang_i = order_langs[i]
        prev_best_dir_i = output_dir / f"task_{i+1}_{prev_lang_i}" / "best_adapter"
        model_file_i = prev_best_dir_i / "calora_model.pt"
        grad_file_i = prev_best_dir_i / "lora_grad.pt"
        
        if not model_file_i.exists() or not grad_file_i.exists():
            print(f"     WARNING: Missing calora_model.pt or lora_grad.pt inside {prev_best_dir_i} for task {i+1} ({prev_lang_i.upper()})!")
            continue
            
        state_i = torch.load(model_file_i, map_location=device, weights_only=True)
        grad_i = torch.load(grad_file_i, map_location=device, weights_only=True)
        previous_grad[i] = [g.to(device) for g in grad_i]
        
        # Load prompt_key of task i into row i of previous_prompts_keys
        if "prompt_key" in state_i:
            model.previous_prompts_keys.data[i].copy_(state_i["prompt_key"].squeeze(0))
        
        # Load frozen Q/K/V/O adapters for each attention module. This legacy
        # task-indexed path is not used by the shared-adapter sequential run,
        # but keeping it aligned prevents silent Q/V-only restoration if it is
        # used for an inference ablation later.
        def copy_previous_attention_adapters(attn_module, prefix):
            for adapter_name in ("q", "k", "v", "o"):
                key_a = f"{prefix}.lora_{adapter_name}.lora_A.weight"
                key_b = f"{prefix}.lora_{adapter_name}.lora_B.weight"
                if key_a not in state_i or key_b not in state_i:
                    continue
                destination = getattr(attn_module, f"previous_lora_weights_{adapter_name}")[i]
                destination.lora_A.weight.data.copy_(state_i[key_a])
                destination.lora_B.weight.data.copy_(state_i[key_b])

        for j, layer in enumerate(model.model.model.encoder.layers):
            pfx = f"model.model.encoder.layers.{j}.self_attn"
            copy_previous_attention_adapters(layer.self_attn, pfx)
        
        for j, layer in enumerate(model.model.model.decoder.layers):
            pfx_self = f"model.model.decoder.layers.{j}.self_attn"
            copy_previous_attention_adapters(layer.self_attn, pfx_self)
                
            pfx_cross = f"model.model.decoder.layers.{j}.encoder_attn"
            copy_previous_attention_adapters(layer.encoder_attn, pfx_cross)
                
        print(f"     ✓ Loaded task {i+1} [{prev_lang_i.upper()}] LoRA blocks (`q`, `k`, `v`, `o`), `prompt_key`, and gradients ({len(grad_i)} tensors).")
        
    return previous_grad


# ═════════════════════════════════════════════════════════════════════════
# TASK LOGGING & BEST CHECKPOINT CALLBACK
# ═════════════════════════════════════════════════════════════════════════
def load_shared_calora_checkpoint(
    model: NLLBForConditionalGenerationWithCaLoRA,
    cur_task_idx: int,
    order_langs: list[str],
    output_dir: Path,
    device: torch.device,
) -> Dict[int, list[torch.Tensor]]:
    """Restore one shared adapter and load previous task gradients for CaGA.

    Every task uses ``task_id=0``, so its trainable Q/K/V/O LoRA modules
    modules have an identical state-dict layout. AGR consequently starts from
    the best CNI adapter and updates the same parameters used for CNI evaluation.
    Earlier gradients are CaGA memory, not frozen adapters.
    """
    previous_grad: Dict[int, list[torch.Tensor]] = {}
    prev_lang = order_langs[cur_task_idx - 1]
    prev_best_dir = output_dir / f"task_{cur_task_idx}_{prev_lang}" / "best_adapter"
    model_file = prev_best_dir / "calora_model.pt"
    print(f"---> Restoring shared best CaLoRA adapter from [{prev_lang.upper()}] ({prev_best_dir.name}/):")
    if not model_file.exists():
        raise FileNotFoundError(f"Missing shared CaLoRA checkpoint: {model_file}")

    state = torch.load(model_file, map_location=device, weights_only=True)
    try:
        model.load_state_dict(state, strict=True)
    except RuntimeError as exc:
        raise RuntimeError(
            "The preceding checkpoint uses the old task-indexed adapter architecture. "
            "Start a fresh run with --run_tag; do not resume it as shared-adapter CaLoRA."
        ) from exc
    print(f"     ✓ Restored shared LoRA weights and embeddings from [{prev_lang.upper()}]")

    expected_lora_parameters = [
        (name, parameter)
        for name, parameter in model.named_parameters()
        if parameter.requires_grad and any(f".lora_{target}." in name for target in ("q", "k", "v", "o"))
    ]
    expected_tensors = len(expected_lora_parameters)
    print(f"---> Loading CaGA gradient memory across {cur_task_idx} completed task(s):")
    for i in range(cur_task_idx):
        old_lang = order_langs[i]
        grad_file = output_dir / f"task_{i + 1}_{old_lang}" / "best_adapter" / "lora_grad.pt"
        if not grad_file.exists():
            raise FileNotFoundError(f"Missing CaGA gradient memory: {grad_file}")
        gradients = torch.load(grad_file, map_location=device, weights_only=True)
        if len(gradients) != expected_tensors:
            raise ValueError(
                f"Gradient-memory tensor count mismatch for Task {i + 1} ({old_lang}): "
                f"expected {expected_tensors}, found {len(gradients)}. Start a fresh shared-adapter run."
            )
        for tensor_idx, ((name, parameter), gradient) in enumerate(zip(expected_lora_parameters, gradients)):
            if gradient.shape != parameter.shape:
                raise ValueError(
                    f"Gradient-memory shape mismatch for Task {i + 1} ({old_lang}), tensor {tensor_idx} "
                    f"({name}): expected {tuple(parameter.shape)}, found {tuple(gradient.shape)}. "
                    "Start a fresh Q/K/V/O CaLoRA run."
                )
            if not torch.isfinite(gradient).all():
                raise FloatingPointError(
                    f"Non-finite CaGA gradient memory for Task {i + 1} ({old_lang}), tensor {tensor_idx} ({name})."
                )
        previous_grad[i] = [gradient.to(device) for gradient in gradients]
        print(
            f"     -> Verified {len(gradients)} Q/K/V/O CaGA tensors for Task {i + 1}: "
            "count, tensor order, shapes, and finite values are valid."
        )
        print(f"     ✓ Loaded Task {i + 1} [{old_lang.upper()}] gradient memory ({len(gradients)} LoRA tensors).")

    return previous_grad


class SequentialCaLoRATaskLogCallback(TrainerCallback):
    """
    Records metrics into training_log_X.csv / .json (`lang_code`_`task_order_idx`).
    Evaluates dev translations and saves optimal state:
      1. CaLoRA model state dictionary (`calora_model.pt`)
      2. Token embeddings table (`embeddings.pt`)
      3. CaLoRA parameter gradients (`lora_grad.pt`) from `self.trainer.cur_grad`
    so subsequent tasks start from exact highest chrF++ checkpoint with required CaGA projection vectors.
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
        self.trainer = None

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
                batch_size=self.gen_batch_size,
                # Shared-adapter CaLoRA has no task-specific routing at evaluation:
                # every language is measured through the same updated LoRA weights.
                target_task_idx=None,
            )

        print(f"   [Task Log {self.log_name}] Step: {step} | Epoch: {epoch} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dev chrF++: {chrf_score:.2f}")
        log_gpu_memory(f"Eval @ Step {step}", task_idx=self.task_order_idx, lang_code=self.lang_code, csv_path=self.output_dir / "gpu_memory_consumption.csv")

        row = [step, epoch, train_loss, val_loss, round(chrf_score, 2) if not math.isnan(chrf_score) else float("nan")]
        for p in [self.log_csv_path, self.task_csv_path]:
            with open(p, "a", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(row)

        record = {"step": step, "epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, f"{self.lang_code}_chrf": round(chrf_score, 2) if not math.isnan(chrf_score) else None}
        self.history.append(record)
        for p in [self.log_json_path, self.task_json_path]:
            with open(p, "w", encoding="utf-8") as f:
                json.dump(self.history, f, indent=2, ensure_ascii=False)

        # Track and save best CaLoRA checkpoint (`calora_model.pt`, `embeddings.pt`, `lora_grad.pt`)
        if not math.isnan(chrf_score) and chrf_score > self.best_chrf:
            self.best_chrf = chrf_score
            self.best_adapter_dir.mkdir(parents=True, exist_ok=True)
            real_m = getattr(self.model, "module", self.model)
            
            torch.save(real_m.state_dict(), self.best_adapter_dir / "calora_model.pt")
            self.tokenizer.save_pretrained(self.best_adapter_dir)

            emb_layer = get_emb_layer(real_m)
            torch.save(emb_layer.weight.data, self.best_adapter_dir / "embeddings.pt")

            if hasattr(self, "trainer") and self.trainer is not None and hasattr(self.trainer, "cur_grad") and self.trainer.cur_grad is not None:
                torch.save(self.trainer.cur_grad, self.best_adapter_dir / "lora_grad.pt")

            with open(self.best_adapter_dir / "best_metrics.json", "w", encoding="utf-8") as f:
                json.dump({"step": step, "epoch": epoch, "best_chrf": round(chrf_score, 2)}, f, indent=2)
            print(f"   ✓ [BEST ADAPTER] New highest chrF++ ({chrf_score:.2f}) at epoch {epoch}! Saved calora_model.pt + embeddings.pt + lora_grad.pt to {self.best_adapter_dir.name}/")

        return control


# ═════════════════════════════════════════════════════════════════════════
# DATA RESOLUTION & EVALUATION HELPERS
# ═════════════════════════════════════════════════════════════════════════
def resolve_paths(args):
    """Resolves data_in and models directories cleanly across Vast.ai, local, and Kaggle environments."""
    project_root = Path(__file__).resolve().parent.parent.parent

    if args.data_dir:
        data_dir = Path(args.data_dir)
    else:
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
    batch_size: int = 16,
    target_task_idx: Optional[int] = None,
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
        autocast_dtype = torch.bfloat16 if any(p.dtype == torch.bfloat16 for p in real_model.parameters()) else torch.float16
        with torch.no_grad(), torch.autocast("cuda", dtype=autocast_dtype, enabled=torch.cuda.is_available()):
            gen_tokens = real_model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=256,
                num_beams=num_beams,
                target_task_idx=target_task_idx,
            )
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)
        preds.extend([d.strip() for d in decoded])

    # SacreBLEU expects one *reference corpus* per reference stream.  With one
    # reference per source sentence that is ``[refs]``, not ``[[r] for r in
    # refs]``.  The latter shape silently evaluates only the first sentence in
    # sacrebleu 2.6, which makes checkpoint selection and reported chrF++
    # meaningless.
    score = chrf_calculator.corpus_score(preds, [refs]).score
    print(f"\n      [Debug Sample - {lang_code.upper()} | chrF++: {score:.2f}]")
    for s_idx in range(min(3, len(sources))):
        print(f"        SRC : {sources[s_idx]}")
        print(f"        REF : {refs[s_idx]}")
        print(f"        PRED: {preds[s_idx]}")
    print()
    return score


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(description="Sequential CaLoRA (r=128) Continual Learning for Experiment B")
    parser.add_argument("--order", type=str, default="order1", choices=["order1", "order2", "order3"], help="Sequential language order (order1: quy->cni, order2: cni->quy, order3: ayr->quy)")
    parser.add_argument("--lora_r", type=int, default=128, help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=256, help="LoRA alpha scaling (alpha=2*r=256 for exact parity with Experiment A & Protocol)")
    parser.add_argument("--lr", type=float, default=1e-4, help="CaLoRA learning rate (1e-4 recommended across sequential tasks)")
    parser.add_argument("--epochs_per_task", type=int, default=12, help="Number of epochs to train on each sequential task (if static_epochs is set)")
    parser.add_argument("--dynamic_epochs", action="store_true", default=True, help="Dynamically scale epochs per task based on dataset size to prevent overfitting")
    parser.add_argument("--static_epochs", action="store_true", default=False, help="Force exact --epochs_per_task for all languages (disables dynamic_epochs)")
    parser.add_argument("--batch_size", type=int, default=10, help="Per-device batch size (10 fits safely on 24GB GPUs; reduce to 6 if running on 16GB VRAM)")
    parser.add_argument("--grad_accum", type=int, default=12, help="Gradient accumulation steps (10*12=120 or 6*20=120 effective batch size)")
    parser.add_argument("--num_beams", type=int, default=4, help="Number of beams during generation & evaluation (protocol default 4/5)")
    parser.add_argument("--experiment_name", type=str, default="experiment_B", help="Top-level output folder")
    parser.add_argument("--run_tag", type=str, default=None, help="Optional isolated subfolder for a rerun, e.g. paca_off_cni_control")
    parser.add_argument("--data_dir", type=str, default=None, help="Explicit path to data directory containing train/ and dev/ folders")
    parser.add_argument("--output_dir", type=str, default=None, help="Explicit path to top-level models output directory")
    parser.add_argument("--start_task", type=int, default=1, help="Task index (1 to 5) to start training from if resuming across sessions")
    parser.add_argument("--end_task", type=int, default=5, help="Last task index to train (use 1 for the Task-1 diagnostic control)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed, applied before CaLoRA adapter construction for reproducible controls")
    parser.add_argument(
        "--disable_paca_correction",
        action="store_true",
        help="Use raw LoRA gradients. Recommended for the first diagnostic Task-1 control run.",
    )
    parser.add_argument(
        "--paca_warmup_steps",
        type=int,
        default=0,
        help="Optimizer steps with raw LoRA gradients before PaCA starts (default: 0; 50 is a safe optional warm-up).",
    )
    args = parser.parse_args()
    if not 1 <= args.start_task <= args.end_task <= len(ORDERS[args.order]):
        parser.error("Require 1 <= --start_task <= --end_task <= 5 for the selected order.")

    # This must precede model construction because CaLoRA's lora_A matrices
    # are randomly initialised by CaLoRALinear.__init__.
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

    method_tag = f"calora_sequential_r{args.lora_r}"
    lr_tag = f"lr{args.lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    output_dir = base_out / args.experiment_name / method_tag / args.order / lr_tag
    if args.run_tag:
        output_dir = output_dir / args.run_tag
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

    expected_steps_dict = compute_expected_steps(args.batch_size, args.grad_accum, args.epochs_per_task, dynamic_epochs=use_dynamic_epochs)
    eff_batch = args.batch_size * args.grad_accum

    print("════════════════════════════════════════════════════════════════════════")
    print(" EXPERIMENT B: SEQUENTIAL CaLoRA CONTINUAL LEARNING (Causal-Aware LoRA)")
    print(f" Order:        {args.order.upper()} -> {' -> '.join(order_langs)}")
    print(f" Data Dir:     {data_dir}")
    print(f" Output Dir:   {output_dir}")
    print(f" LoRA Rank:    r={args.lora_r}, alpha={args.lora_alpha}, targets=[q_proj, k_proj, v_proj, o_proj]")
    print(f" Precision:    {dtype_str}")
    print(f" Batching:     batch_size={args.batch_size} * grad_accum={args.grad_accum} = Effective Batch {eff_batch}")
    print(f" Seed:         {args.seed} (set before model and adapter construction)")
    print(f" Epoch Mode:   {'Dynamic Scaling (prevents overfitting on QUY/AYR)' if use_dynamic_epochs else f'Static ({args.epochs_per_task} epochs/task)'}")
    print(f" Num Beams:    {args.num_beams} (for evaluation & generation)")
    print(f" PaCA:         {'OFF (raw LoRA control)' if args.disable_paca_correction else 'ON (mean-one per-tensor scales)'} | warm-up={args.paca_warmup_steps} steps")
    print(f" Task range:   {args.start_task}..{args.end_task}")
    print(" Expected Dynamic Steps per Language:")
    for l_c, info in expected_steps_dict.items():
        print(f"   - {l_c.upper()} ({LANG_TO_NLLB[l_c]}): {info['epochs']} epochs | ~{info['total_steps']:,} total steps (~{info['steps_per_epoch']:,} steps/epoch | {info['pairs']:,} pairs)")
    print("════════════════════════════════════════════════════════════════════════\n")

    # 1. Load Tokenizer
    tok_path = MODEL_NAME
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

    new_tokens_needed = []
    for lang in NEW_LANGUAGES:
        t_str = LANG_TO_NLLB[lang]
        tid = tokenizer.convert_tokens_to_ids(t_str)
        if tid == tokenizer.unk_token_id or tid is None:
            new_tokens_needed.append(t_str)

    if new_tokens_needed:
        print(f"---> Adding new language tokens to vocabulary: {new_tokens_needed}")
        tokenizer.add_special_tokens({"additional_special_tokens": new_tokens_needed})

    # 2. Check Base Model Local Path
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

    load_dtype = torch.bfloat16 if use_bf16 else (torch.float16 if use_fp16 else torch.float32)

    # 3. Load Dev sets for Triangular Evaluation & Task Logging
    dev_data_dict = {}
    for lc in order_langs:
        pair = load_dev_raw(lc, data_dir)
        if pair:
            dev_data_dict[lc] = {"src": pair[0], "tgt": pair[1]}
        else:
            print(f"WARNING: Dev files missing for {lc} inside {data_dir / 'dev' / lc}")

    # A staged run (for example, Task 1 first and Tasks 2--5 later) must
    # preserve the already-computed triangular rows.  The shared adapter and
    # CaGA memory already resume from Task 1; without this reload, the final
    # BWT calculation would silently lose the Task-1 diagonal score.
    eval_matrix_path = output_dir / "triangular_eval_matrix.json"
    eval_matrix = {}
    if args.start_task > 1:
        if not eval_matrix_path.exists():
            raise FileNotFoundError(
                f"Cannot resume from Task {args.start_task}: missing prior triangular matrix at {eval_matrix_path}. "
                "Run the preceding task through its triangular evaluation first."
            )
        with open(eval_matrix_path, "r", encoding="utf-8") as f:
            eval_matrix = json.load(f)
        expected_prior_rows = order_langs[: args.start_task - 1]
        missing_prior_rows = [lang for lang in expected_prior_rows if lang not in eval_matrix]
        if missing_prior_rows:
            raise ValueError(
                f"Cannot resume from Task {args.start_task}: triangular matrix is missing prior row(s) "
                f"{missing_prior_rows}."
            )
        print(
            f"---> Resumed triangular evaluation matrix with {len(eval_matrix)} prior row(s): "
            f"{', '.join(eval_matrix.keys())}."
        )

    # 4. SEQUENTIAL CONTINUAL LEARNING LOOP ACROSS 5 TASKS
    for task_idx, lang_code in enumerate(order_langs):
        print("\n" + "█" * 70)
        print(f" TASK {task_idx + 1}/{len(order_langs)}: TRAINING CaLoRA ON [{lang_code.upper()}] ({LANG_TO_NLLB[lang_code]})")
        print("█" * 70)
        log_gpu_memory(f"Start Task {task_idx + 1} ({lang_code.upper()})", task_idx=task_idx + 1, lang_code=lang_code, csv_path=output_dir / "gpu_memory_consumption.csv")

        if task_idx + 1 < args.start_task:
            print(f"---> Skipping Task {task_idx + 1} ({lang_code.upper()}) because --start_task is set to {args.start_task}.")
            continue
        if task_idx + 1 > args.end_task:
            break

        # One shared CaLoRA adapter is updated sequentially, just like the LoRA
        # baseline. Task-specific gradient snapshots are CaGA memory only; they
        # are not separate frozen adapters.
        prompt_config = {
            "task_id": 0,
            "lora_r": args.lora_r,
            "lora_alpha": args.lora_alpha,
            "lora_dropout": 0.05,
            "trans_hidden_dim": 100,
            "attn_temperature": 1.0,
            "previous_lora_path": None,
            "previous_prompt_key_path": None
        }
        
        print("---> Instantiating NLLBForConditionalGenerationWithCaLoRA (shared adapter; task_id=0)...")
        model = NLLBForConditionalGenerationWithCaLoRA(model_load_path, prompt_config, torch_dtype=load_dtype)

        # Ensure token embeddings are sized for all special tokens
        initialize_new_language_embeddings(model, tokenizer)

        # Restore the immediately preceding best shared adapter; retain all prior
        # gradient snapshots for CaGA's Task-2+ correlation/affinity calculation.
        if task_idx > 0:
            previous_grad = load_shared_calora_checkpoint(model, task_idx, order_langs, output_dir, device)
        else:
            previous_grad = {}

        # Apply selective embedding gradient hook specifically for seen active languages (Protocol Section 7.3)
        active_seen_langs = order_langs[:task_idx + 1]
        apply_embedding_masking(model, tokenizer, active_seen_langs)

        model.to(device)

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

        task_order = [order_langs[i] for i in range(task_idx + 1)]

        training_args = Seq2SeqTrainingArguments(
            output_dir=str(task_out_dir / "hf"),
            max_steps=max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_ratio=0.06,
            lr_scheduler_type="cosine",
            eval_strategy=eval_strat if hasattr(Seq2SeqTrainingArguments, "eval_strategy") else eval_strat,
            save_strategy="no",               # Manually saved via SequentialCaLoRATaskLogCallback
            logging_steps=max(5, steps_per_epoch // 4),
            predict_with_generate=True,
            generation_num_beams=args.num_beams,
            bf16=use_bf16,
            fp16=use_fp16,
            optim="adamw_torch",
            max_grad_norm=1.0,
            label_smoothing_factor=0.1,       # CRITICAL: Parity with train_lora_sequential.py & Protocol
            weight_decay=0.0,                 # CRITICAL: Required by CaLoRA to prevent base vocabulary decay
            ignore_data_skip=True,
            group_by_length=False,
            dataloader_num_workers=4 if os.name != "nt" else 0,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True if os.name != "nt" else False,
            dataloader_prefetch_factor=2 if os.name != "nt" else None,
            tf32=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
            load_best_model_at_end=False,
            ddp_find_unused_parameters=False,
        )

        task_logger = SequentialCaLoRATaskLogCallback(
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

        trainer = CleanCaLoRATrainer(
            model=model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=dev_ds if has_eval else None,
            processing_class=tokenizer,
            tokenizer=tokenizer,
            data_collator=DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, label_pad_token_id=-100),
            callbacks=[task_logger],
            cur_task_id=task_idx,
            task_order=task_order,
            previous_grad=previous_grad,
            enable_paca_correction=not args.disable_paca_correction,
            paca_warmup_steps=args.paca_warmup_steps,
        )
        task_logger.trainer = trainer

        # Train on current language task via native transformers Seq2SeqTrainer loop
        trainer.train()
        log_gpu_memory(f"End Training Task {task_idx + 1}", task_idx=task_idx + 1, lang_code=lang_code, csv_path=output_dir / "gpu_memory_consumption.csv")

        # Save final epoch adapter (`calora_model.pt`, `embeddings.pt`, `lora_grad.pt`) to last_adapter/
        last_dir = task_out_dir / "last_adapter"
        real_m = getattr(model, "module", model)
        if is_main_process():
            last_dir.mkdir(parents=True, exist_ok=True)
            print(f"\n---> Saving final epoch (last_adapter) CaLoRA weights, embeddings.pt, and gradients to: {last_dir.name}/")
            torch.save(real_m.state_dict(), last_dir / "calora_model.pt")
            tokenizer.save_pretrained(last_dir)

            emb_layer = get_emb_layer(real_m)
            torch.save(emb_layer.weight.data, last_dir / "embeddings.pt")

            if hasattr(trainer, "cur_grad") and trainer.cur_grad is not None:
                torch.save(trainer.cur_grad, last_dir / "lora_grad.pt")

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

        # Before running Triangular Evaluation, reload BEST adapter of this task
        best_dir = task_out_dir / "best_adapter"
        if best_dir.exists():
            if is_main_process():
                print(f"---> Reloading BEST adapter + embeddings.pt from {best_dir.name}/ for Triangular Evaluation and subsequent tasks...")
            model_file = best_dir / "calora_model.pt"
            if model_file.exists():
                st_dict = torch.load(model_file, map_location=device, weights_only=True)
                real_m.load_state_dict(st_dict)

            emb_file = best_dir / "embeddings.pt"
            if emb_file.exists():
                emb_data = torch.load(emb_file, map_location=device, weights_only=True)
                get_emb_layer(model).weight.data.copy_(emb_data)
        else:
            if is_main_process():
                best_dir.mkdir(parents=True, exist_ok=True)
                torch.save(real_m.state_dict(), best_dir / "calora_model.pt")
                tokenizer.save_pretrained(best_dir)
                emb_layer = get_emb_layer(real_m)
                torch.save(emb_layer.weight.data, best_dir / "embeddings.pt")
                if hasattr(trainer, "cur_grad") and trainer.cur_grad is not None:
                    torch.save(trainer.cur_grad, best_dir / "lora_grad.pt")
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
                        batch_size=16,
                        # Use the single shared adapter for every past-language evaluation.
                        target_task_idx=None,
                    )
                    eval_matrix[lang_code][past_lang] = round(chrf, 2)
                    print(f"   Dev [{past_lang.upper()}]: chrF++ = {chrf:.2f}")

            with open(eval_matrix_path, "w", encoding="utf-8") as f:
                json.dump(eval_matrix, f, indent=2, ensure_ascii=False)
            model.train()
            log_gpu_memory(f"End Eval Task {task_idx + 1}", task_idx=task_idx + 1, lang_code=lang_code, csv_path=output_dir / "gpu_memory_consumption.csv")

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
        log_gpu_memory(f"Post-Cleanup Task {task_idx + 1}", task_idx=task_idx + 1, lang_code=lang_code, csv_path=output_dir / "gpu_memory_consumption.csv")

    # 5. Summary Report & Continual Learning Metrics (FM, AA, BWT)
    if is_main_process():
        print("\n" + "═" * 70)
        print(f" EXPERIMENT B (SEQUENTIAL CaLoRA r={args.lora_r} | {args.order.upper()}) FINAL REPORT")
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
        print(" SEQUENTIAL CaLoRA CONTINUAL LEARNING COMPLETE!")
        print("═" * 70)


if __name__ == "__main__":
    main()
