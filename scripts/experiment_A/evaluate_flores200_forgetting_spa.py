#!/usr/bin/env python3
"""
evaluate_flores200_forgetting_spa.py
====================================
Fast Catastrophic Forgetting Benchmark on FLORES-200 across ~200 NLLB Languages.
Focuses ONLY on Spanish -> X (spa_Latn -> X) direction to cut inference time in half
while maintaining 100% official published-grade accuracy (num_beams=4).

Evaluates 4 models without quantization (in bfloat16/float16):
  1. lora_r32:    models/experiment_A/final/lora_r32/lr1e-4/best_chrf_checkpoint
  2. lora_r128:   models/experiment_A/final/lora_r128/lr1e-4/best_chrf_checkpoint
  3. pissa_r128:  models/experiment_A/final/pissa_r128/lr1e-4/best_chrf_checkpoint
  4. base:        facebook/nllb-200-1.3B (evaluated last!)

Translation Direction:
  • Spanish to X (spa_Latn -> X) ONLY

Statistical Reporting Tiers (Mean, Median, Std, Q1, Q3, Min, Max):
  • Tier 1: All ~200 Target Languages (Overall)
  • Tier 2: Excluding Quechua (quy_Latn) & Aymara (ayr_Latn) — TRUE FORGETTING
  • Tier 3: Quechua & Aymara Only — FINE-TUNED RETENTION / GAIN

Usage (Optimized for NVIDIA RTX 5090 / 4090 / A100):
  python scripts/experiment_A/evaluate_flores200_forgetting_spa.py \
      --batch_size 96 --num_beams 4
"""

import argparse
import csv
import gc
import json
import math
import os
import sys
from pathlib import Path
from typing import Optional, Dict, List, Tuple

# Force UTF-8 output on Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# CRITICAL VAST.AI FIX: Prevent OpenMP/MKL/PyTorch thread thrashing and deadlock on Xeon/EPYC CPUs in Docker containers
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np
import sacrebleu
import torch
from tqdm import tqdm
from datasets import load_dataset, get_dataset_config_names
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer
from peft import PeftModel

# Limit PyTorch CPU threads to avoid cgroup lock contention on many-core host machines
try:
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)
except Exception:
    pass

# ---------------------------------------------------------------------------
# Configuration & Constants
# ---------------------------------------------------------------------------
BASE_MODEL_ID = "facebook/nllb-200-1.3B"
SRC_LANGS = ["spa_Latn"]  # ONLY SPANISH TO X DIRECTION FOR FAST INFERENCE
FINE_TUNED_LANGS = {"quy_Latn", "ayr_Latn"}  # NLLB codes for Quechua & Aymara

MODEL_CONFIGS = {
    "lora_r32": "models/experiment_A/final/lora_r32/lr1e-4/best_chrf_checkpoint",
    "lora_r128": "models/experiment_A/final/lora_r128/lr1e-4/best_chrf_checkpoint",
    "pissa_r128": "models/experiment_A/final/pissa_r128/lr1e-4/best_chrf_checkpoint",
    "fft_frozen": "models/experiment_A/fft_frozen/lr5e-6/best_chrf_checkpoint",
    "fft_unfrozen": "models/experiment_A/fft_unfrozen/lr5e-6/best_chrf_checkpoint",
    "base": None,  # Base model placed at the end!
}


# ---------------------------------------------------------------------------
# FLORES-200 Dataset Loader
# ---------------------------------------------------------------------------
def get_all_flores_languages() -> List[str]:
    """Discover all available language configurations in FLORES-200."""
    project_root = Path(__file__).resolve().parent.parent.parent
    local_devtest = project_root / "flores200" / "devtest"
    if local_devtest.exists():
        valid = [f.name.split(".")[0] for f in local_devtest.glob("*.devtest")]
        if len(valid) > 50:
            print(f"[INFO] Found {len(valid)} local FLORES-200 language files in: {local_devtest}")
            return sorted(valid)

    for repo in ["tomasmajercik/flores-parquet", "openchami/flores200", "Muennighoff/flores200", "facebook/flores"]:
        try:
            configs = get_dataset_config_names(repo)
            valid = [c for c in configs if c.endswith("_Latn") or c.endswith("_Cyrl") or c.endswith("_Arab") or c.endswith("_Deva") or c.endswith("_Hans") or c.endswith("_Hant") or "_" in c]
            if len(valid) > 50:
                print(f"[INFO] Discovered {len(valid)} languages in FLORES-200 repository: {repo}")
                return sorted(valid)
        except Exception as e:
            continue
    
    # Fallback curated list of common NLLB language codes if online config discovery fails
    print("[WARNING] Could not dynamically fetch FLORES-200 config names. Using fallback list.")
    return sorted(["eng_Latn", "spa_Latn", "fra_Latn", "deu_Latn", "por_Latn", "ita_Latn", "rus_Cyrl", "arb_Arab", "zho_Hans", "hin_Deva", "quy_Latn", "ayr_Latn", "cat_Latn", "nld_Latn", "pol_Latn", "tur_Latn", "vie_Latn", "kor_Hang", "jpn_Jpan", "swe_Latn"])


def load_flores_data(lang_code: str, split: str = "devtest") -> Optional[List[str]]:
    """Load sentences for a specific language code from FLORES-200."""
    project_root = Path(__file__).resolve().parent.parent.parent
    local_file = project_root / "flores200" / split / f"{lang_code}.{split}"
    if not local_file.exists():
        # Try checking .dev if split is validation or vice versa
        alt_split = "dev" if split == "validation" else split
        local_file = project_root / "flores200" / alt_split / f"{lang_code}.{alt_split}"

    if local_file.exists():
        with open(local_file, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]

    for repo in ["tomasmajercik/flores-parquet", "openchami/flores200", "Muennighoff/flores200", "facebook/flores"]:
        for sp in [split, "devtest", "validation", "test", "dev"]:
            try:
                ds = load_dataset(repo, lang_code, split=sp)
                for col in ["sentence", "text", "translation"]:
                    if col in ds.column_names:
                        return [str(s).strip() for s in ds[col]]
                return [str(row).strip() for row in ds]
            except Exception:
                continue
    return None


# ---------------------------------------------------------------------------
# Model Loading (No Quantization, Optimized for RTX 5090 / 3090 / CPU)
# ---------------------------------------------------------------------------
def load_eval_model(model_name: str, ckpt_path: Optional[str], device: str) -> Tuple[torch.nn.Module, NllbTokenizer]:
    print(f"\n[{model_name.upper()}] Loading model...")
    
    # Determine best dtype: bfloat16 if supported, else float16 on CUDA, float32 on CPU
    if device == "cuda":
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
            print("  Using dtype: torch.bfloat16")
        else:
            dtype = torch.float16
            print("  Using dtype: torch.float16")
    else:
        dtype = torch.float32
        print("  Using dtype: torch.float32 (CPU)")

    if ckpt_path and Path(ckpt_path).exists():
        print(f"  Loading tokenizer from checkpoint: {ckpt_path}")
        tokenizer = NllbTokenizer.from_pretrained(ckpt_path)
    else:
        print(f"  Loading base tokenizer: {BASE_MODEL_ID}")
        tokenizer = NllbTokenizer.from_pretrained(BASE_MODEL_ID)

    print(f"  Loading base model weights: {BASE_MODEL_ID}")
    print(f"  [NOTE] If this is the first time loading on this instance, HuggingFace is downloading a 5.4 GB checkpoint over the network. This can take 3-5 minutes before GPU VRAM spikes!")
    model = AutoModelForSeq2SeqLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=dtype,
        device_map=device if device == "cuda" else None,
    )
    if device == "cpu":
        model.to("cpu")

    # Resize embeddings if tokenizer vocabulary expanded
    model.resize_token_embeddings(len(tokenizer))

    if ckpt_path and Path(ckpt_path).exists():
        adapter_config = Path(ckpt_path) / "adapter_config.json"
        if adapter_config.exists():
            print(f"  Applying PEFT adapter from {ckpt_path}...")
            model = PeftModel.from_pretrained(model, ckpt_path)
            
            # Restore custom injected vocabulary embeddings if present
            embeddings_path = Path(ckpt_path) / "embeddings.pt"
            if embeddings_path.exists():
                print("  Recovering natively injected vocab embeddings (embeddings.pt)...")
                emb_layer = model.get_input_embeddings()
                loaded_emb = torch.load(embeddings_path, map_location=device).to(dtype=dtype)
                emb_layer.weight.data = loaded_emb
                if hasattr(model, "get_output_embeddings") and model.get_output_embeddings() is not None:
                    model.get_output_embeddings().weight.data = loaded_emb
        else:
            print("  [INFO] Standard full fine-tuned weights detected.")
            model = AutoModelForSeq2SeqLM.from_pretrained(
                ckpt_path,
                torch_dtype=dtype,
                device_map=device if device == "cuda" else None,
            )
            if device == "cpu":
                model.to("cpu")
                
    model.to(dtype=dtype, device=device)
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Translation Generation
# ---------------------------------------------------------------------------
def translate_batch(
    model,
    tokenizer: NllbTokenizer,
    src_lines: List[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 96,
    device: str = "cuda",
    num_beams: int = 4,
) -> List[str]:
    model.eval()
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    translations = []

    for i in tqdm(range(0, len(src_lines), batch_size), desc=f"    {src_lang[:3]}->{tgt_lang[:3]}", leave=False):
        batch = src_lines[i : i + batch_size]
        tokenizer.src_lang = src_lang
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=256,
                num_beams=num_beams,
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)

    return translations


# ---------------------------------------------------------------------------
# Statistical Calculation & Reporting
# ---------------------------------------------------------------------------
def compute_statistics(scores: List[float]) -> Dict[str, float]:
    if not scores:
        return {"mean": 0.0, "median": 0.0, "std": 0.0, "q1": 0.0, "q3": 0.0, "min": 0.0, "max": 0.0}
    arr = np.array(scores)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "std": float(np.std(arr)),
        "q1": float(np.percentile(arr, 25)),
        "q3": float(np.percentile(arr, 75)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


# ---------------------------------------------------------------------------
# Main Execution Flow
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="NLLB-200 FLORES-200 Catastrophic Forgetting Benchmark (Spanish->X Only)")
    parser.add_argument("--models", type=str, default="all", help="Comma-separated list of models to evaluate or 'all'")
    parser.add_argument("--languages", type=str, default="all", help="Comma-separated list of target language codes or 'all'")
    parser.add_argument("--max_sentences", type=int, default=1012, help="Max sentences per language (e.g. 2 for smoketest)")
    parser.add_argument("--batch_size", type=int, default=96, help="Batch size for inference (default 96 for RTX 5090)")
    parser.add_argument("--num_beams", type=int, default=4, help="Beam search size (default 4 for official NLLB paper accuracy)")
    parser.add_argument("--output_dir", type=str, default="data_out/experiment_A_forgetting_spa", help="Output directory")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing translation JSONL files if present")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent.parent
    out_dir = project_root / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*70}")
    print(f"  FLORES-200 Forgetting Benchmark (Spanish->X Only)")
    print(f"  Device        : {device.upper()}")
    print(f"  Max Sentences : {args.max_sentences}")
    print(f"  Batch Size    : {args.batch_size}")
    print(f"  Num Beams     : {args.num_beams}")
    print(f"  Output Dir    : {out_dir}")
    print(f"  Overwrite     : {args.overwrite}")
    print(f"{'='*70}")

    # 1. Discover or parse target languages
    if args.languages.lower() == "all":
        target_langs = get_all_flores_languages()
    else:
        target_langs = [l.strip() for l in args.languages.split(",") if l.strip()]

    print(f"\n[INFO] Evaluating across {len(target_langs)} target languages (Direction: spa_Latn -> X).")

    # 2. Determine models to evaluate
    if args.models.lower() == "all":
        selected_models = list(MODEL_CONFIGS.keys())
    else:
        selected_models = [m.strip() for m in args.models.split(",") if m.strip() in MODEL_CONFIGS]

    if not selected_models:
        print(f"\n[ERROR] No valid models found to evaluate! You specified --models '{args.models}', but MODEL_CONFIGS only contains: {list(MODEL_CONFIGS.keys())}")
        return

    # Pre-fetch Spanish source language sentences
    src_data: Dict[str, List[str]] = {}
    for src_code in SRC_LANGS:
        print(f"[INFO] Pre-fetching source language data for {src_code}...")
        data = load_flores_data(src_code)
        if data:
            src_data[src_code] = data[:args.max_sentences]
        else:
            print(f"[ERROR] Could not load source language {src_code}!")
            return

    # Master structure to hold all score results
    all_results: Dict[str, Dict[str, Dict[str, float]]] = {m: {"spa->X": {}} for m in selected_models}

    # 3. Iterate through models
    for model_name in selected_models:
        ckpt_rel = MODEL_CONFIGS[model_name]
        ckpt_path = str(project_root / ckpt_rel) if ckpt_rel else None

        if ckpt_rel and not Path(ckpt_path).exists():
            # Fallback: check if nested inside an extra subdirectory with the same name
            parent_dir = Path(ckpt_path).parent
            nested_path = parent_dir / parent_dir.name / "best_chrf_checkpoint"
            if nested_path.exists():
                ckpt_path = str(nested_path)
            else:
                print(f"\n[SKIP] Checkpoint not found for {model_name} at {ckpt_path}")
                continue

        export_file_path = out_dir / f"{model_name}_spa_translations.jsonl"

        # Check existing completed languages for resume or skip
        completed_langs = set()
        existing_records_by_lang: Dict[str, List[dict]] = {}
        if export_file_path.exists() and not args.overwrite and export_file_path.stat().st_size > 0:
            with open(export_file_path, "r", encoding="utf-8") as f_in:
                for line in f_in:
                    if not line.strip(): continue
                    try:
                        record = json.loads(line)
                        tgt_k = record.get("tgt_lang")
                        if tgt_k:
                            if tgt_k not in existing_records_by_lang:
                                existing_records_by_lang[tgt_k] = []
                            existing_records_by_lang[tgt_k].append(record)
                    except Exception:
                        continue

            for tgt_k, records in existing_records_by_lang.items():
                # Determine expected sentence count for this target language
                tgt_lines = load_flores_data(tgt_k)
                if tgt_lines:
                    min_len = min(len(src_data.get("spa_Latn", [])), len(tgt_lines[:args.max_sentences]))
                    if min_len > 0 and len(records) >= min_len:
                        completed_langs.add(tgt_k)
                        hyps = [r["hyp"] for r in records[:min_len]]
                        refs = [r["ref"] for r in records[:min_len]]
                        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2).score
                        bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
                        all_results[model_name]["spa->X"][tgt_k] = {
                            "chrf": round(chrf, 2),
                            "bleu": round(bleu, 2)
                        }

        langs_to_eval = [l for l in target_langs if l != "spa_Latn" and l not in completed_langs]

        if not langs_to_eval:
            print(f"\n[SKIP] All {len(completed_langs)} target languages already translated for {model_name} in {export_file_path}")
            continue

        if completed_langs:
            print(f"\n[RESUME] Found existing file for {model_name} with {len(completed_langs)} completed target languages. Resuming remaining {len(langs_to_eval)} languages...")
        else:
            print(f"\n[START] Generating translations for {model_name} across {len(langs_to_eval)} target languages...")

        model, tokenizer = load_eval_model(model_name, ckpt_path, device=device)

        file_mode = "a" if (export_file_path.exists() and not args.overwrite and len(completed_langs) > 0) else "w"
        if file_mode == "a" and any(k not in completed_langs for k in existing_records_by_lang.keys()):
            print("  [CLEANUP] Removing partial/interrupted records before resuming...")
            with open(export_file_path, "w", encoding="utf-8") as f_clean:
                for tgt_k in sorted(completed_langs):
                    for rec in existing_records_by_lang[tgt_k]:
                        f_clean.write(json.dumps(rec, ensure_ascii=False) + "\n")

        with open(export_file_path, file_mode, encoding="utf-8") as f_export:
            for tgt_code in tqdm(langs_to_eval, desc=f"Evaluating {model_name} (spa->X)"):
                tgt_lines = load_flores_data(tgt_code)
                if not tgt_lines:
                    continue
                tgt_lines = tgt_lines[:args.max_sentences]

                for src_code in SRC_LANGS:
                    if src_code == tgt_code:
                        continue
                    
                    direction = "spa->X"
                    src_lines = src_data[src_code]
                    min_len = min(len(src_lines), len(tgt_lines))
                    if min_len == 0:
                        continue

                    src_slice = src_lines[:min_len]
                    tgt_slice = tgt_lines[:min_len]

                    # Translate
                    hyps = translate_batch(
                        model, tokenizer, src_slice,
                        src_lang=src_code, tgt_lang=tgt_code,
                        batch_size=args.batch_size, device=device,
                        num_beams=args.num_beams,
                    )

                    # Compute sentence-level metrics
                    chrf = sacrebleu.corpus_chrf(hyps, [tgt_slice], word_order=2).score
                    bleu = sacrebleu.corpus_bleu(hyps, [tgt_slice]).score

                    all_results[model_name][direction][tgt_code] = {
                        "chrf": round(chrf, 2),
                        "bleu": round(bleu, 2)
                    }

                    # Write to JSONL export
                    for idx, (s_txt, r_txt, h_txt) in enumerate(zip(src_slice, tgt_slice, hyps)):
                        record = {
                            "model": model_name,
                            "direction": "spa->X",
                            "src_lang": src_code,
                            "tgt_lang": tgt_code,
                            "sentence_idx": idx,
                            "src": s_txt,
                            "ref": r_txt,
                            "hyp": h_txt
                        }
                        f_export.write(json.dumps(record, ensure_ascii=False) + "\n")
                    f_export.flush()

        # Free memory between models
        del model
        del tokenizer
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # 4. Compute Summary Report across Tiers
    print(f"\n{'='*70}")
    print("  Calculating Catastrophic Forgetting Summary Statistics (spa->X)...")
    print(f"{'='*70}")

    summary_rows = []
    
    for model_name in selected_models:
        direction = "spa->X"
        lang_scores_chrf = {k: v["chrf"] for k, v in all_results[model_name][direction].items()}
        lang_scores_bleu = {k: v["bleu"] for k, v in all_results[model_name][direction].items()}

        # Tier 1: All Languages
        t1_c = compute_statistics(list(lang_scores_chrf.values()))
        t1_b = compute_statistics(list(lang_scores_bleu.values()))

        # Tier 2: Excluding Quechua & Aymara (TRUE FORGETTING)
        t2_c_vals = [v for k, v in lang_scores_chrf.items() if k not in FINE_TUNED_LANGS]
        t2_b_vals = [v for k, v in lang_scores_bleu.items() if k not in FINE_TUNED_LANGS]
        t2_c = compute_statistics(t2_c_vals)
        t2_b = compute_statistics(t2_b_vals)

        # Tier 3: Quechua & Aymara Only
        t3_c_vals = [v for k, v in lang_scores_chrf.items() if k in FINE_TUNED_LANGS]
        t3_b_vals = [v for k, v in lang_scores_bleu.items() if k in FINE_TUNED_LANGS]
        t3_c = compute_statistics(t3_c_vals)
        t3_b = compute_statistics(t3_b_vals)

        summary_rows.append({
            "model": model_name,
            "direction": direction,
            "tier1_chrf_mean": round(t1_c["mean"], 2),
            "tier1_chrf_median": round(t1_c["median"], 2),
            "tier1_chrf_std": round(t1_c["std"], 2),
            "tier1_chrf_q1": round(t1_c["q1"], 2),
            "tier1_chrf_q3": round(t1_c["q3"], 2),
            "tier2_chrf_mean": round(t2_c["mean"], 2),
            "tier2_chrf_median": round(t2_c["median"], 2),
            "tier2_chrf_std": round(t2_c["std"], 2),
            "tier2_chrf_q1": round(t2_c["q1"], 2),
            "tier2_chrf_q3": round(t2_c["q3"], 2),
            "tier3_chrf_mean": round(t3_c["mean"], 2),
            "tier3_chrf_median": round(t3_c["median"], 2),
            "tier3_chrf_std": round(t3_c["std"], 2),
        })

    if not summary_rows:
        print("\n[ERROR] No summary statistics generated! Exiting report generation.")
        return

    # 5. Save Summary CSV & Markdown Table
    csv_path = out_dir / "forgetting_benchmark_summary_spa.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f_csv:
        writer = csv.DictWriter(f_csv, fieldnames=summary_rows[0].keys())
        writer.writeheader()
        writer.writerows(summary_rows)

    md_path = out_dir / "forgetting_benchmark_report_spa.md"
    with open(md_path, "w", encoding="utf-8") as f_md:
        f_md.write("# FLORES-200 Catastrophic Forgetting Benchmark Report (Spanish->X Only)\n\n")
        f_md.write("## chrF++ Summary Results (word_order=2)\n\n")
        f_md.write("| Model | Direction | All Mean ± Std | All Median (Q1-Q3) | **True Forgetting Mean ± Std** (excl. quy/ayr) | **True Forgetting Median** | quy/ayr Mean |\n")
        f_md.write("| :--- | :--- | :--- | :--- | :--- | :--- | :--- |\n")
        for r in summary_rows:
            all_str = f"{r['tier1_chrf_mean']} ± {r['tier1_chrf_std']}"
            all_med = f"{r['tier1_chrf_median']} ({r['tier1_chrf_q1']}-{r['tier1_chrf_q3']})"
            tf_str = f"**{r['tier2_chrf_mean']} ± {r['tier2_chrf_std']}**"
            tf_med = f"**{r['tier2_chrf_median']}**"
            qa_str = f"{r['tier3_chrf_mean']}"
            f_md.write(f"| {r['model']} | {r['direction']} | {all_str} | {all_med} | {tf_str} | {tf_med} | {qa_str} |\n")

    print(f"\n[SUCCESS] Benchmark summary saved to:\n  • CSV : {csv_path}\n  • MD  : {md_path}")
    print("\nSample Preview Table:")
    with open(md_path, "r", encoding="utf-8") as f_md:
        print(f_md.read())


if __name__ == "__main__":
    main()
