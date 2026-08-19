#!/usr/bin/env python3
import os
import sys
from pathlib import Path
from transformers import NllbTokenizer

LANG_TO_NLLB = {
    "quy": "quy_Latn",
    "ayr": "ayr_Latn",
    "shp": "shp_Latn",
    "agr": "agr_Latn",
    "cni": "cni_Latn",
}
SRC_LANG = "spa_Latn"
MAX_LEN = 256

def filter_and_count():
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data_in" / "train"
    out_file = project_root / "data_in" / "summary_counts.txt"
    
    tokenizer_path = project_root / "models/v0/tokenizer"
    if tokenizer_path.exists():
        tokenizer = NllbTokenizer.from_pretrained(str(tokenizer_path))
        print("Loaded modified tokenizer.")
    else:
        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B")
        print("Loaded base tokenizer facebook/nllb-200-1.3B.")

    summary_lines = []
    summary_lines.append(f"{'Language':<10} | {'Original Pairs':<15} | {'Filtered (<=256)':<15} | {'Retained %':<10}")
    summary_lines.append("-" * 60)

    total_orig = 0
    total_filt = 0

    for lang, nllb_code in LANG_TO_NLLB.items():
        lang_dir = data_dir / lang
        es_path = lang_dir / "train.es"
        tgt_path = lang_dir / f"train.{lang}"
        
        if not es_path.exists() or not tgt_path.exists():
            print(f"[SKIP] Missing train data for {lang}")
            continue
            
        with open(es_path, "r", encoding="utf-8") as f:
            es_lines = [line.strip() for line in f]
        with open(tgt_path, "r", encoding="utf-8") as f:
            tgt_lines = [line.strip() for line in f]

        orig_len = min(len(es_lines), len(tgt_lines))
        total_orig += orig_len
        
        filtered_es = []
        filtered_tgt = []
        
        # Batch tokenization for speed
        BATCH_SIZE = 1000
        for i in range(0, orig_len, BATCH_SIZE):
            es_batch = es_lines[i:i+BATCH_SIZE]
            tgt_batch = tgt_lines[i:i+BATCH_SIZE]
            
            tokenizer.src_lang = SRC_LANG
            es_tokens = tokenizer(es_batch, add_special_tokens=True)["input_ids"]
            
            tokenizer.src_lang = nllb_code
            tgt_tokens = tokenizer(tgt_batch, add_special_tokens=True)["input_ids"]
            
            for src_t, tgt_t, src_text, tgt_text in zip(es_tokens, tgt_tokens, es_batch, tgt_batch):
                if len(src_t) <= MAX_LEN and len(tgt_t) <= MAX_LEN:
                    filtered_es.append(src_text)
                    filtered_tgt.append(tgt_text)
                    
        filt_len = len(filtered_es)
        total_filt += filt_len
        ratio = (filt_len / orig_len * 100) if orig_len > 0 else 0
        
        summary_lines.append(f"{lang:<10} | {orig_len:<15} | {filt_len:<15} | {ratio:.2f}%")
        
        # Saving filtered
        with open(lang_dir / "train.filtered.es", "w", encoding="utf-8") as f:
            f.write("\n".join(filtered_es) + "\n")
        with open(lang_dir / f"train.filtered.{lang}", "w", encoding="utf-8") as f:
            f.write("\n".join(filtered_tgt) + "\n")

        print(f"Processed {lang}: {filt_len}/{orig_len} pairs kept.")

    summary_lines.append("-" * 60)
    total_ratio = (total_filt / total_orig * 100) if total_orig > 0 else 0
    summary_lines.append(f"{'TOTAL':<10} | {total_orig:<15} | {total_filt:<15} | {total_ratio:.2f}%")

    with open(out_file, "w", encoding="utf-8") as f:
        f.write("\n".join(summary_lines) + "\n")
        
    print(f"\nSummary written to {out_file}")

if __name__ == "__main__":
    filter_and_count()
