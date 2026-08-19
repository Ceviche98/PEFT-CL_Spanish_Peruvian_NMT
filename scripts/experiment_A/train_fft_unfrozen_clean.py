#!/usr/bin/env python3
r"""
train_fft_unfrozen_clean.py
============================
Clean Full Fine-Tuning (FFT) benchmark for NLLB-200-1.3B with UNFROZEN embeddings.

WHY THIS SCRIPT USES NATIVE Seq2SeqTrainer with Transformers 4.40.2
-------------------------------------------------------------------
We use `transformers==4.40.2` (the mature, highly stable release of the Transformers 4.x series).
Under `transformers==4.40.2`:
    1. `M2M100ForConditionalGeneration.forward()` and `DataCollatorForSeq2Seq` natively unpack
       keyword `labels` and `decoder_input_ids` without any dual-input collisions (`got multiple
       values for argument input_ids`) or `decoder_inputs_embeds` errors.
    2. Unlike older 4.33.x versions, `transformers==4.40.2` officially depends on `tokenizers~=0.19`,
       which ships instant, prebuilt Python 3.12 (`cp312`) binary wheels on PyPI (`.whl`).
    3. This eliminates all Rust compiler (`rustc`) requirements AND avoids `ImportError: tokenizers...`
       from Hugging Face's internal `dependency_versions_check.py`.

Enables 100% clean, native, and unpatched Hugging Face Seq2Seq training on modern hardware (RTX 5090 + Ryzen 9600X).
"""

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional
import numpy as np

# ── Environment setup (VAST.AI / remote GPU cluster) ──────────────────────
if os.path.exists("/workspace"):
    os.environ["HF_HOME"] = "/workspace/.cache/huggingface"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# CPU thread alignment: AMD Ryzen 5 9600X (6 physical cores / 12 logical SMT threads)
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
try:
    torch.set_num_threads(1)
except Exception:
    pass

import sacrebleu
from datasets import Dataset, concatenate_datasets, interleave_datasets
from transformers import (
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq,
    NllbTokenizer,
    Trainer,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
)
import transformers
from transformers.optimization import Adafactor
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import get_last_checkpoint
from transformers.modeling_outputs import Seq2SeqLMOutput


# ── Constants ─────────────────────────────────────────────────────────────
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


# ═════════════════════════════════════════════════════════════════════════
# CLEAN TRAINER — Native Seq2SeqTrainer for Transformers 4.33.3
# ═════════════════════════════════════════════════════════════════════════
class CleanFFTTrainer(Seq2SeqTrainer):
    """Native Seq2SeqTrainer with one global learning rate.

    Full fine-tuning is the control condition: every trainable parameter, including the
    tied input/output embedding matrix, receives exactly ``args.learning_rate``. PyTorch's
    ``model.parameters()`` emits a tied ``Parameter`` only once, avoiding the historical
    duplicate-update bug without assigning embeddings a special LR.
    """
    def create_optimizer(self):
        if self.optimizer is None:
            params = [p for p in self.model.parameters() if p.requires_grad]
            if len({id(p) for p in params}) != len(params):
                raise RuntimeError("Duplicate Parameter found in FFT optimizer input.")

            # Match the standard Transformers FFT convention: weight decay applies to
            # matrix weights (including the tied embedding/output matrix), but never to
            # biases or LayerNorm scales.  Both groups use the *same* Adafactor LR.
            decay_names = set(get_parameter_names(self.model, ALL_LAYERNORM_LAYERS))
            decay_names = {name for name in decay_names if not name.endswith(".bias")}
            decay_params, no_decay_params = [], []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                (decay_params if name in decay_names else no_decay_params).append(param)
            if len(decay_params) + len(no_decay_params) != len(params):
                raise RuntimeError("Optimizer groups do not cover every trainable FFT parameter.")

            tied = self.model.get_input_embeddings().weight is getattr(
                getattr(self.model, "lm_head", None), "weight", None
            )
            optimizer_name = self.args.optim
            print(
                f"\n[Optimizer] global-LR {optimizer_name}: {len(params)} unique tensors, "
                f"{sum(p.numel() for p in params):,} trainable params | "
                f"lr={self.args.learning_rate:g}, weight_decay={self.args.weight_decay:g} "
                f"({len(decay_params)} tensors; {len(no_decay_params)} bias/LayerNorm tensors at 0), "
                f"tied_input_output_embeddings={tied}"
            )
            groups = [
                {"params": decay_params, "weight_decay": self.args.weight_decay},
                {"params": no_decay_params, "weight_decay": 0.0},
            ]
            if optimizer_name == "adamw_torch":
                self.optimizer = torch.optim.AdamW(
                    groups,
                    lr=self.args.learning_rate,
                    betas=(self.args.adam_beta1, self.args.adam_beta2),
                    eps=self.args.adam_epsilon,
                )
            elif optimizer_name == "adafactor":
                self.optimizer = Adafactor(
                    groups,
                    lr=self.args.learning_rate,
                    scale_parameter=False,
                    relative_step=False,
                    clip_threshold=1.0,
                )
            else:
                raise ValueError(
                    f"Unsupported FFT optimizer {optimizer_name!r}; use 'adamw_torch' or 'adafactor'."
                )
        return self.optimizer


from transformers.models.m2m_100.modeling_m2m_100 import shift_tokens_right

class M2M100DataCollator(DataCollatorForSeq2Seq):
    """
    Data collator that explicitly guarantees `decoder_input_ids` are populated
    from `labels` before Trainer.compute_loss pops `labels` for label smoothing.
    Bypasses Hugging Face M2M100 missing prepare_decoder_input_ids_from_labels method.
    """
    def __call__(self, features, return_tensors=None):
        batch = super().__call__(features, return_tensors=return_tensors)
        if "decoder_input_ids" not in batch and "labels" in batch:
            batch["decoder_input_ids"] = shift_tokens_right(
                batch["labels"], self.model.config.pad_token_id, self.model.config.decoder_start_token_id
            )
        return batch




# ═════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═════════════════════════════════════════════════════════════════════════
def read_parallel_lines(src_path: Path, tgt_path: Path) -> tuple[list[str], list[str]]:
    """Read an LF-delimited parallel corpus without independently dropping either side.

    Do not use :meth:`str.splitlines` here.  Some corpora contain the C0 record/group
    separator characters (U+001C/U+001D) *inside* a sentence; ``splitlines`` treats
    them as line breaks and silently destroys the parallel alignment.  Corpus records
    are delimited by LF only.
    """
    def read_lf_records(path: Path) -> list[str]:
        rows = path.read_text(encoding="utf-8").split("\n")
        if rows and rows[-1] == "":  # normal final LF, not an extra corpus record
            rows.pop()
        return [row.rstrip("\r") for row in rows]

    src_raw = read_lf_records(src_path)
    tgt_raw = read_lf_records(tgt_path)
    if len(src_raw) != len(tgt_raw):
        raise ValueError(
            f"Parallel files have different line counts: {src_path} ({len(src_raw)}) vs "
            f"{tgt_path} ({len(tgt_raw)})"
        )

    pairs = [(src.strip(), tgt.strip()) for src, tgt in zip(src_raw, tgt_raw)
             if src.strip() and tgt.strip()]
    dropped = len(src_raw) - len(pairs)
    if dropped:
        print(f"[INFO] Dropped {dropped} blank parallel pair(s): {src_path.parent.name}")
    return [src for src, _ in pairs], [tgt for _, tgt in pairs]


def load_language_data(lang_code: str, data_dir: Path, tokenizer: NllbTokenizer,
                       max_length: int = 256) -> Optional[Dataset]:
    """Load and tokenize training data for a single language pair (es → lang)."""
    es_path = data_dir / "train" / lang_code / "train.filtered.es"
    lang_path = data_dir / "train" / lang_code / f"train.filtered.{lang_code}"

    if not es_path.exists() or not lang_path.exists():
        es_path = data_dir / "train" / lang_code / "train.es"
        lang_path = data_dir / "train" / lang_code / f"train.{lang_code}"
        if not es_path.exists() or not lang_path.exists():
            print(f"[SKIP] Data not found for {lang_code}")
            return None

    src_lines, tgt_lines = read_parallel_lines(es_path, lang_path)
    tgt_nllb = LANG_TO_NLLB[lang_code]

    def tokenize(batch):
        tokenizer.src_lang = SRC_LANG
        tokenizer.tgt_lang = tgt_nllb
        return tokenizer(
            batch["src"],
            text_target=batch["tgt"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    raw = Dataset.from_dict({"src": src_lines, "tgt": tgt_lines})
    tokenized = raw.map(tokenize, batched=True, batch_size=512, remove_columns=["src", "tgt"])
    return tokenized


def load_dev_raw(lang_code: str, data_dir: Path) -> Optional[tuple[list[str], list[str]]]:
    """Load raw dev set text lines for a language pair."""
    es_path = data_dir / "dev" / lang_code / "dev.es"
    lang_path = data_dir / "dev" / lang_code / f"dev.{lang_code}"
    if not es_path.exists() or not lang_path.exists():
        return None
    return read_parallel_lines(es_path, lang_path)


# ═════════════════════════════════════════════════════════════════════════
# EVALUATION CALLBACK
# ═════════════════════════════════════════════════════════════════════════
class MultiLangEvalCallback(TrainerCallback):
    """Computes per-language chrF++ on dev sets after each evaluation step."""

    def __init__(self, model, tokenizer, active_langs, data_dir, output_dir,
                 patience=5, gen_batch_size=16, max_new_tokens=256, num_beams=1):
        self.model = model
        self.tokenizer = tokenizer
        self.active_langs = active_langs
        self.output_dir = output_dir
        self.patience = patience
        self.gen_batch_size = gen_batch_size
        self.max_new_tokens = max_new_tokens
        self.num_beams = num_beams

        self.log_csv_path = output_dir / "training_log.csv"
        self.log_csv_path.parent.mkdir(parents=True, exist_ok=True)

        if not self.log_csv_path.exists():
            with open(self.log_csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(
                    ["step", "train_loss", "val_loss", "avg_chrf"]
                    + [f"{lc}_chrf" for lc in active_langs]
                )

        self.dev_data = {}
        for lc in active_langs:
            pair = load_dev_raw(lc, data_dir)
            if pair:
                self.dev_data[lc] = pair

        metrics_path = output_dir / "best_chrf_checkpoint" / "metrics.json"
        if metrics_path.exists():
            try:
                self.best_avg_chrf = float(json.loads(metrics_path.read_text(encoding="utf-8"))["avg_chrf"])
                print(f"[Evaluation] Resuming best chrF++ threshold: {self.best_avg_chrf:.2f}")
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f"[Evaluation] Could not restore best chrF++ ({exc}); starting fresh.")
                self.best_avg_chrf = -1.0
        else:
            self.best_avg_chrf = -1.0
        self.no_improve_evals = 0

    def _generate(self, model, src_lines, tgt_nllb_token):
        """Translate using exact native model.generate with beam search."""
        device = next(model.parameters()).device
        model.eval()
        forced_bos_id = self.tokenizer.convert_tokens_to_ids(tgt_nllb_token)
        translations = []

        for i in range(0, len(src_lines), self.gen_batch_size):
            batch = src_lines[i : i + self.gen_batch_size]
            self.tokenizer.src_lang = SRC_LANG
            inputs = self.tokenizer(
                batch, return_tensors="pt", padding=True,
                truncation=True, max_length=256,
            ).to(device)

            with torch.no_grad():
                generated_ids = model.generate(
                    **inputs,
                    forced_bos_token_id=forced_bos_id,
                    max_new_tokens=self.max_new_tokens,
                    num_beams=self.num_beams,
                )
            translations.extend(
                self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
            )

        return translations

    def on_evaluate(self, args, state, control, model=None, **kwargs):
        eval_model = model if model is not None else self.model
        step = state.global_step
        all_chrf = []
        lang_metrics = {}
        sample_translations = {}

        for lc, (src, ref) in self.dev_data.items():
            hyps = self._generate(eval_model, src, LANG_TO_NLLB[lc])
            chrf = sacrebleu.corpus_chrf(hyps, [ref], word_order=2).score
            all_chrf.append(chrf)
            lang_metrics[lc] = chrf
            sample_translations[lc] = (src[0], ref[0], hyps[0])

        avg_chrf = np.mean(all_chrf) if all_chrf else float("nan")

        train_loss = next(
            (x["loss"] for x in reversed(state.log_history) if "loss" in x),
            float("nan"),
        )
        val_loss = kwargs.get("metrics", {}).get("eval_loss", float("nan"))

        print(f"\nStep {step} Eval | Val Loss: {val_loss:.4f} | Avg chrF++: {avg_chrf:.2f}")
        scores_str = " | ".join(f"{lc.upper()}: {lang_metrics.get(lc, 0.0):.2f}" for lc in self.active_langs if lc in lang_metrics)
        print(f"Per-Language chrF++: {scores_str}")
        print("─" * 70)
        for lc in self.active_langs:
            if lc in sample_translations:
                s, r, h = sample_translations[lc]
                print(f"[{lc.upper()}] SRC: {s[:60]}...")
                print(f"      REF: {r[:60]}...")
                print(f"      HYP: {h[:60]}...")
        print("─" * 70)

        with open(self.log_csv_path, "a", newline="", encoding="utf-8") as f:
            row = [step, train_loss, val_loss, avg_chrf]
            row.extend([lang_metrics.get(lc, float("nan")) for lc in self.active_langs])
            csv.writer(f).writerow(row)

        if not math.isnan(avg_chrf) and avg_chrf > self.best_avg_chrf:
            self.best_avg_chrf = avg_chrf
            self.no_improve_evals = 0

            p = self.output_dir / "best_chrf_checkpoint"
            p.mkdir(parents=True, exist_ok=True)
            eval_model.save_pretrained(str(p))
            self.tokenizer.save_pretrained(str(p))

            emb_layer = eval_model.get_input_embeddings()
            torch.save(emb_layer.weight.data, p / "embeddings.pt")

            with open(p / "metrics.json", "w") as f:
                json.dump({"step": step, "avg_chrf": avg_chrf}, f)
            print("  ✓ Saved best chrF++ checkpoint (and explicit embeddings.pt).")
        else:
            self.no_improve_evals += 1

        if self.no_improve_evals >= self.patience:
            print(f"Early stopping triggered at step {step}!")
            control.should_training_stop = True

        # Trainer.evaluate() leaves the model in evaluation mode. Restore training mode only
        # after all generation and checkpoint operations have completed.
        eval_model.train()
        return control


class GroupGradNormCallback(TrainerCallback):
    """
    Logs the gradient norm of the embedding layer separately to detect localized blow-ups
    that get diluted in the aggregate 1.3B parameter grad_norm.
    """
    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step > 0 and state.global_step % args.logging_steps == 0 and model is not None:
            embed_weight = model.get_input_embeddings().weight
            if embed_weight.grad is not None:
                embed_norm = embed_weight.grad.detach().data.norm(2).item()
                if state.is_world_process_zero:
                    print(f"   [Step {state.global_step}] Embeddings Grad Norm: {embed_norm:.4f}")


# ═════════════════════════════════════════════════════════════════════════
# EMBEDDING INITIALIZATION
# ═════════════════════════════════════════════════════════════════════════
def initialize_embeddings(model, tokenizer):
    """Initialize new target language embeddings as (Quechua + Aymara) / 2 average."""
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


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Clean FFT Unfrozen Training for NLLB-200-1.3B (CleanFFTTrainer)"
    )
    parser.add_argument(
        "--lr", type=float, default=2e-5,
        help="One global learning rate for every trainable parameter (default: 2e-5).",
    )
    parser.add_argument(
        "--optimizer", choices=["adamw_torch", "adafactor"], default="adafactor",
        help="Optimizer for full fine-tuning (default: adafactor). Both use one global LR.",
    )
    parser.add_argument(
        "--max_steps", type=int, default=600000,
        help="Total training steps (shell script typically overrides this).",
    )
    parser.add_argument(
        "--experiment_name", type=str, default="experiment_A",
        help="Top-level folder under models/.",
    )
    parser.add_argument(
        "--patience", type=int, default=5,
        help="Early-stopping patience (evals without chrF++ improvement). "
             "Set to 999 to effectively disable.",
    )
    parser.add_argument(
        "--label_smoothing", type=float, default=0.1,
        help="Label smoothing factor for cross-entropy loss.",
    )
    parser.add_argument(
        "--warmup_ratio", type=float, default=0.1,
        help="Fraction of total optimizer steps used for linear warmup (default: 0.1).",
    )
    parser.add_argument(
        "--sampling_temperature", type=float, default=0.7,
        help="Exponent T in p(language) proportional to n_language**T (default: 0.7).",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.01,
        help="Weight decay for every FFT parameter, including the tied embedding matrix.",
    )
    parser.add_argument(
        "--batch_size", type=int, default=12,
        help="Per-device train batch size (default 12).",
    )
    parser.add_argument(
        "--grad_accum", type=int, default=10,
        help="Gradient accumulation steps (default 10; effective batch size = 120).",
    )
    parser.add_argument(
        "--num_workers", type=int, default=12,
        help="DataLoader CPU workers (default 12 for AMD EPYC 24 available cores).",
    )
    parser.add_argument(
        "--eval_num_beams", type=int, default=1,
        help="Beam width used for checkpoint selection (1 = greedy decoding).",
    )
    parser.add_argument(
        "--evals_per_epoch", type=int, default=1,
        help="How many generation evaluations to run per nominal epoch (default: 1).",
    )
    parser.add_argument(
        "--run_tag", type=str, default="global_adafactor_b12x10",
        help="Subdirectory for this hyperparameter configuration; prevents incompatible resume.",
    )
    args = parser.parse_args()
    if args.sampling_temperature <= 0:
        parser.error("--sampling_temperature must be greater than zero.")

    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data_in"
    method_tag = "fft_unfrozen"

    lr = args.lr
    lr_tag = f"lr{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")

    # A run tag prevents accidental resume with a different optimizer or effective batch size.
    output_dir = project_root / "models" / args.experiment_name / method_tag / lr_tag / args.run_tag
    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Tokenizer ─────────────────────────────────────────────────────
    tokenizer_path = project_root / "models/v0/tokenizer"
    tokenizer = NllbTokenizer.from_pretrained(str(tokenizer_path))

    # ── Model ─────────────────────────────────────────────────────────
    base_model_id = os.environ.get("BASE_MODEL_PATH", "facebook/nllb-200-1.3B")
    local_drive_path = project_root / "models" / "nllb-200-1.3B"
    if os.path.exists(local_drive_path) and os.path.isdir(local_drive_path):
        base_model_id = str(local_drive_path)
        print(f"[INFO] Local model detected at '{base_model_id}'.")

    # Preserve the exact executable configuration beside each run. Historical FFT logs
    # cannot otherwise establish which source revision or launcher arguments produced them.
    manifest_path = output_dir / "run_manifest.json"
    manifest = {
        "arguments": vars(args),
        "base_model": base_model_id,
        "tokenizer_path": str(tokenizer_path),
        "transformers_version": transformers.__version__,
        "torch_version": torch.__version__,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sampling_temperature_exponent": args.sampling_temperature,
    }
    if manifest_path.exists():
        old_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if old_manifest != manifest:
            raise RuntimeError(
                f"Existing run manifest differs from this invocation: {manifest_path}. "
                "Choose a new --run_tag rather than resuming incompatible state."
            )
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading '{base_model_id}' (bf16, low_cpu_mem_usage=True)…")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_id, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
    )
    print(f"Moving model to {device}...")
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    # Initialize new language embeddings
    initialize_embeddings(model, tokenizer)

    # Verify embeddings are fully unfrozen
    emb_layer = model.get_input_embeddings()
    emb_layer.weight.requires_grad = True
    if not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        model.prepare_decoder_input_ids_from_labels = lambda labels: shift_tokens_right(
            labels, model.config.pad_token_id, model.config.decoder_start_token_id
        )
    print("✓ Embedding table completely UNFROZEN (embed_tokens + shared). All 1.3B parameters trainable.")

    # ── Dataset preparation ───────────────────────────────────────────
    print(f"\nPreparing Datasets with Temperature T={args.sampling_temperature:g}…")
    all_ds = []
    lengths = []
    for lc in ALL_LANGS:
        ds = load_language_data(lc, data_dir, tokenizer, max_length=256)
        if ds:
            all_ds.append(ds)
            lengths.append(len(ds))

    lengths = np.array(lengths)
    T = args.sampling_temperature  # Temperature < 1 flattens distribution, upsampling low-resource langs
    probs = (lengths ** T) / np.sum(lengths ** T)
    print(f"Sampling Probabilities: {dict(zip(ALL_LANGS, [round(p, 3) for p in probs]))}")

    eff_batch = args.batch_size * args.grad_accum  # default 12 * 10 = 120
    _nominal_steps_per_epoch = max(100, int(sum(lengths) / eff_batch))
    effective_passes = (_nominal_steps_per_epoch * eff_batch * probs) / lengths
    print(
        "Effective passes per nominal epoch: "
        + str(dict(zip(ALL_LANGS, [round(x, 2) for x in effective_passes])))
    )
    samples_needed = args.max_steps * eff_batch * probs  # per language
    repeats_needed = (np.ceil(samples_needed / lengths).astype(int) + 1).tolist()
    print(f"Per-language repeats for {args.max_steps} steps: {dict(zip(ALL_LANGS, repeats_needed))}")

    train_ds = interleave_datasets(
        [concatenate_datasets([all_ds[i]] * repeats_needed[i]) for i in range(len(all_ds))],
        probabilities=probs, seed=42, stopping_strategy="first_exhausted",
    )

    # ── Evaluation dataset ────────────────────────────────────────────
    print("\nPreparing Evaluation Datasets...")
    dev_datasets = []
    for lc in ALL_LANGS:
        pair = load_dev_raw(lc, data_dir)
        if pair:
            es_lines, lang_lines = pair
            def tok(batch, _lc=lc):
                tokenizer.src_lang = SRC_LANG
                tokenizer.tgt_lang = LANG_TO_NLLB[_lc]
                return tokenizer(
                    batch["src"],
                    text_target=batch["tgt"],
                    truncation=True,
                    max_length=256,
                    padding=False,
                )
            ds_dev = Dataset.from_dict({"src": es_lines, "tgt": lang_lines})
            ds_dev = ds_dev.map(tok, batched=True, batch_size=512, remove_columns=["src", "tgt"])
            dev_datasets.append(ds_dev)
    eval_ds = concatenate_datasets(dev_datasets) if dev_datasets else None

    # ── Training configuration ────────────────────────────────────────
    _steps_per_epoch = _nominal_steps_per_epoch
    _eval_every = max(1, _steps_per_epoch // args.evals_per_epoch)
    _save_every = _eval_every
    _log_every = max(10, _steps_per_epoch // 30)  # ~30 log lines per epoch
    print(
        f"Steps per epoch: {_steps_per_epoch}  |  "
        f"Eval/save every: {_eval_every} steps ({args.evals_per_epoch}x per epoch)"
    )

    # Use evaluation_strategy for transformers <= 4.40 (like 4.33.3), and eval_strategy for >= 4.41
    eval_arg_key = "evaluation_strategy" if int(transformers.__version__.split(".")[0]) <= 4 and int(transformers.__version__.split(".")[1]) <= 40 else "eval_strategy"
    eval_kwargs = {eval_arg_key: "steps"}

    training_args = Seq2SeqTrainingArguments(
        output_dir=str(output_dir / "hf"),
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum, # Default effective batch = 120 (12 x 10)
        learning_rate=lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        eval_steps=_eval_every,
        logging_steps=_log_every,
        save_strategy="steps",
        save_steps=_save_every,
        save_total_limit=1,                   # Keep only latest checkpoint
        ignore_data_skip=True,                # Instant resume without data fast-forward
        gradient_checkpointing=False,         # No recomputation overhead
        dataloader_num_workers=args.num_workers,     # 12 background CPU workers for AMD EPYC 24 available cores
        dataloader_pin_memory=True,           # Instant DMA PCIe transfer
        bf16=True,
        optim=args.optimizer,
        max_grad_norm=1.0,
        label_smoothing_factor=args.label_smoothing, # Handled cleanly by Seq2SeqTrainer
        weight_decay=args.weight_decay,       # Handled via create_optimizer parameter groups
        **eval_kwargs,
    )

    # ── CleanFFTTrainer (Zero Monkey-Patches on Transformers Classes) ─
    trainer = CleanFFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=M2M100DataCollator(
            tokenizer=tokenizer, model=model,
            padding=True, label_pad_token_id=-100,
        ),
        callbacks=[
            MultiLangEvalCallback(
                model, tokenizer, ALL_LANGS, data_dir, output_dir,
                patience=args.patience,
                num_beams=args.eval_num_beams,
            ),
            GroupGradNormCallback(),
        ],
    )

    # ── Launch training ───────────────────────────────────────────────
    last_checkpoint = get_last_checkpoint(str(output_dir / "hf"))
    if last_checkpoint is not None:
        print(f"\n[RESUME] Found checkpoint at {last_checkpoint}. Resuming...")
        trainer.train(resume_from_checkpoint=last_checkpoint)
    else:
        print("\nStarting Clean FFT Unfrozen Training (CleanFFTTrainer) from scratch…")
        trainer.train()


if __name__ == "__main__":
    main()
