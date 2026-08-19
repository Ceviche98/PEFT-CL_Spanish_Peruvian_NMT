#!/usr/bin/env python3
"""
evaluate_lora.py
================
Step 5 of the NLLB-200 LoRA pipeline.

Loads a trained LoRA adapter, runs batch inference on a test/dev file,
and reports BLEU and chrF++ (word_order=2) scores — the same metrics
computed by the existing scripts/evaluate.py, extended with model inference.

Compatible with all three saved checkpoints:
    models/v0/checkpoints/best_agr_chrf/
    models/v0/checkpoints/best_avg_chrf/
    models/v0/checkpoints/final/

Usage:
    # Evaluate best Awajún model on dev set
    python scripts/evaluate_lora.py \\
        --model_path models/v0/checkpoints/best_agr_chrf \\
        --source_file data_in/dev/agr/dev.es \\
        --reference_file data_in/dev/agr/dev.agr \\
        --src_lang spa_Latn \\
        --tgt_lang agr_Latn

    # Evaluate on a custom test file, save output
    python scripts/evaluate_lora.py \\
        --model_path models/v0/checkpoints/best_avg_chrf \\
        --source_file data_in/test/quy/test.es \\
        --reference_file data_in/test/quy/test.quy \\
        --src_lang spa_Latn \\
        --tgt_lang quy_Latn \\
        --output_file evaluation/v0/quy_test_translations.txt \\
        --detailed_output

HUMAN REVIEW GATE  ►  Verify that:
    • Translations are printed / saved and look linguistically plausible
    • BLEU and chrF++ scores are printed in the same format as evaluate.py
    • Score report saved to evaluation/v0/<lang_code>_scores.txt
"""

import argparse
import datetime
from pathlib import Path
from typing import Optional

import sacrebleu
import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSeq2SeqLM, NllbTokenizer


# ---------------------------------------------------------------------------
# Score report (mirrors existing evaluate.py API)
# ---------------------------------------------------------------------------

def calculate_score_report(
    sys_lines: list[str],
    ref_lines: list[str],
    score_only: bool = False,
    lang_code: str = "?",
) -> dict:
    """
    Compute chrF++ and BLEU; print in the same format as scripts/evaluate.py.
    Returns a dict with numeric scores for downstream use.
    """
    chrf = sacrebleu.corpus_chrf(sys_lines, [ref_lines], word_order=2)
    bleu = sacrebleu.corpus_bleu(sys_lines, [ref_lines])
    prefix = "BLEU = " if score_only else ""

    print("#### Score Report ####")
    print(f"  Language  : {lang_code}")
    print(f"  Sentences : {len(sys_lines)}")
    print(chrf)
    print(f"{prefix}{bleu.format(score_only=score_only)}")

    return {"bleu": round(bleu.score, 2), "chrf": round(chrf.score, 2)}


# ---------------------------------------------------------------------------
# Model loading (LoRA adapter on top of base NLLB model)
# ---------------------------------------------------------------------------

def load_lora_model(model_path: str, device: str = "auto") -> tuple:
    """
    Load a PEFT LoRA adapter from disk.
    Reads the base model name from the PeftConfig stored in model_path.
    Returns (model, tokenizer).
    """
    print(f"\nLoading LoRA adapter from {model_path} …")
    peft_config = PeftConfig.from_pretrained(model_path)
    base_model_id = peft_config.base_model_name_or_path
    print(f"  Base model : {base_model_id}")

    # 1. Load tokenizer from the adapter directory FIRST
    tokenizer = NllbTokenizer.from_pretrained(model_path)
    print(f"  Vocabulary : {len(tokenizer):,} tokens")

    # 2. Load base model in fp16
    base_model = AutoModelForSeq2SeqLM.from_pretrained(
        base_model_id,
        torch_dtype=torch.float16 if device != "cpu" else torch.float32,
        device_map=device,
    )

    # 3. CRITICAL FIX: Resize the base model's embeddings to match the new tokenizer
    base_model.resize_token_embeddings(len(tokenizer))

    # 4. Now it is safe to load the PEFT model containing the updated weights
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()

    return model, tokenizer


# ---------------------------------------------------------------------------
# Batch inference
# ---------------------------------------------------------------------------

def translate_file(
    model,
    tokenizer: NllbTokenizer,
    src_lines: list[str],
    src_lang: str,
    tgt_lang: str,
    batch_size: int = 8,
    max_new_tokens: int = 256,
    num_beams: int = 4,
    device: str = "cuda",
) -> list[str]:
    """Generate translations for all src_lines in batches."""
    forced_bos_id = tokenizer.convert_tokens_to_ids(tgt_lang)
    translations = []
    total = len(src_lines)

    for i in range(0, total, batch_size):
        batch = src_lines[i : i + batch_size]
        pct = (i + len(batch)) / total * 100
        print(f"\r  Translating… {i + len(batch)}/{total} ({pct:.0f}%)", end="", flush=True)

        tokenizer.src_lang = src_lang
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                forced_bos_token_id=forced_bos_id,
                max_new_tokens=max_new_tokens,
                num_beams=num_beams,
                repetition_penalty=1.2,          # Adds a mathematical penalty if it repeats words
                no_repeat_ngram_size=3,          # Completely bans repeating any 3-word phrase
            )
        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        translations.extend(decoded)

    print()   # newline after progress
    return translations


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained NLLB-200 LoRA model with BLEU + chrF++"
    )
    parser.add_argument("--model_path", required=True,
                        help="Path to LoRA adapter directory (e.g. models/v0/checkpoints/best_agr_chrf)")
    parser.add_argument("--source_file", "--src", required=True,
                        help="File with Spanish source sentences (one per line)")
    parser.add_argument("--reference_file", "--ref", required=True,
                        help="File with reference translations (one per line)")
    parser.add_argument("--src_lang", default="spa_Latn",
                        help="NLLB source language code (default: spa_Latn)")
    parser.add_argument("--tgt_lang", required=True,
                        help="NLLB target language code (e.g. agr_Latn, quy_Latn)")
    parser.add_argument("--output_file", default=None,
                        help="Optional: file to save generated translations")
    parser.add_argument("--scores_dir", default=None,
                        help="Directory to save score report. Default: <project>/evaluation/v0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--num_beams", type=int, default=4)
    parser.add_argument("--detailed_output", action="store_true",
                        help="Print full sacrebleu details (mirrors evaluate.py flag)")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    scores_dir = Path(args.scores_dir) if args.scores_dir else project_root / "evaluation" / "v0"
    scores_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*60}")
    print(f"  NLLB-200 LoRA Pipeline — Step 5: Evaluate")
    print(f"  Source lang : {args.src_lang}  →  {args.tgt_lang}")
    print(f"  Device      : {device}")
    print(f"{'='*60}")

    # ------------------------------------------------------------------
    # 1. Load model
    # ------------------------------------------------------------------
    model, tokenizer = load_lora_model(args.model_path, device=device)

    # ------------------------------------------------------------------
    # 2. Load source and reference files (matching evaluate.py filtering)
    # ------------------------------------------------------------------
    no_translations: list[int] = []
    gold_lines: list[str] = []
    with open(args.reference_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if len(line.strip()) == 0:
                no_translations.append(i)
                continue
            gold_lines.append(line.strip())

    src_lines_raw: list[str] = []
    with open(args.source_file, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i in no_translations:
                continue
            src_lines_raw.append(line.strip())

    print(f"\n  Source sentences : {len(src_lines_raw)}")
    print(f"  References       : {len(gold_lines)}")
    if len(src_lines_raw) != len(gold_lines):
        print(f"[WARN] Line count mismatch — truncating to {min(len(src_lines_raw), len(gold_lines))}")
        n = min(len(src_lines_raw), len(gold_lines))
        src_lines_raw = src_lines_raw[:n]
        gold_lines = gold_lines[:n]

    # ------------------------------------------------------------------
    # 3. Generate translations
    # ------------------------------------------------------------------
    print(f"\nGenerating translations …")
    hypotheses = translate_file(
        model, tokenizer, src_lines_raw,
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
        num_beams=args.num_beams,
        device=device,
    )

    # ------------------------------------------------------------------
    # 4. Save translations
    # ------------------------------------------------------------------
    lang_code = args.tgt_lang.split("_")[0]  # e.g. "agr" from "agr_Latn"
    if args.output_file:
        out_path = Path(args.output_file)
    else:
        out_path = scores_dir / f"{lang_code}_translations.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(hypotheses) + "\n")
    print(f"  Translations saved → {out_path}")

    # ------------------------------------------------------------------
    # 5. Score report (same format as evaluate.py)
    # ------------------------------------------------------------------
    print()
    scores = calculate_score_report(
        sys_lines=hypotheses,
        ref_lines=gold_lines,
        score_only=not args.detailed_output,
        lang_code=args.tgt_lang,
    )

    # Save score report to text file
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    score_file = scores_dir / f"{lang_code}_scores_{ts}.txt"
    with open(score_file, "w", encoding="utf-8") as f:
        f.write(f"Language  : {args.tgt_lang}\n")
        f.write(f"Model     : {args.model_path}\n")
        f.write(f"Source    : {args.source_file}\n")
        f.write(f"Reference : {args.reference_file}\n")
        f.write(f"Timestamp : {ts}\n\n")
        f.write(f"BLEU   : {scores['bleu']}\n")
        f.write(f"chrF++ : {scores['chrf']}\n")
    print(f"\n  Score report saved → {score_file}")
    print(f"\n  ► HUMAN REVIEW GATE: verify scores and inspect {out_path} for translation quality.")


if __name__ == "__main__":
    main()
