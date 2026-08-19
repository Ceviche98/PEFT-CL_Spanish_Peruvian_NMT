#!/usr/bin/env python3
import argparse
from pathlib import Path
from transformers import NllbTokenizer

LANG_TO_NLLB = {
    "quy": "quy_Latn",
    "ayr": "ayr_Latn",
    "shp": "shp_Latn",
    "agr": "agr_Latn",
    "cni": "cni_Latn",
}

def analyze_fertility(args):
    project_root = Path(__file__).resolve().parent.parent.parent
    data_dir = project_root / "data_in" / "train"
    
    tokenizer_path = project_root / "models/v0/tokenizer"
    if tokenizer_path.exists():
        tokenizer = NllbTokenizer.from_pretrained(str(tokenizer_path))
    else:
        tokenizer = NllbTokenizer.from_pretrained("facebook/nllb-200-1.3B")

    experiment_A_dir = project_root / "scripts" / "experiment_A"
    output_file = experiment_A_dir / "token_fertility.txt"
    
    lines_out = []
    header = f"{'Language':<10} | {'Sentences':<10} | {'Total Words':<15} | {'Total Subwords':<15} | {'Fertility':<10}"
    lines_out.append(header)
    lines_out.append("-" * 65)
    
    print(header)
    print("-" * 65)

    for lang, nllb_code in LANG_TO_NLLB.items():
        tgt_path = data_dir / lang / f"train.{lang}"
        if not tgt_path.exists():
            continue
            
        with open(tgt_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
            
        if not lines:
            continue
            
        total_words = 0
        total_subwords = 0
        
        tokenizer.src_lang = nllb_code
        BATCH_SIZE = 1000
        for i in range(0, len(lines), BATCH_SIZE):
            batch = lines[i:i+BATCH_SIZE]
            
            for sentence in batch:
                words = sentence.split()
                total_words += len(words)
                
            encodings = tokenizer(batch, add_special_tokens=False)["input_ids"]
            for token_list in encodings:
                total_subwords += len(token_list)

        fertility = total_subwords / total_words if total_words > 0 else 0
        
        res = f"{lang:<10} | {len(lines):<10} | {total_words:<15} | {total_subwords:<15} | {fertility:.3f}"
        lines_out.append(res)
        print(res)
        
    with open(output_file, "w", encoding="utf-8") as out_f:
        out_f.write("\n".join(lines_out) + "\n")
    print(f"\nFertility report written to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    args = parser.parse_args()
    analyze_fertility(args)
