#!/usr/bin/env python3
"""
modify_tokenizer.py
===================
Step 2 of the NLLB-200 LoRA pipeline.

Loads the facebook/nllb-200-3.3B tokenizer, adds language tokens for
languages NOT natively present in the NLLB-200 vocabulary, and saves the
modified tokenizer to disk.

NLLB-200 natively supports (among others):
    quy_Latn  — Southern Quechua
    ayr_Latn  — Central Aymara
    grn_Latn  — Guaraní

The following 3 tokens are NEW and will be added:
    shp_Latn  — Shipibo-Konibo
    agr_Latn  — Awajún
    cni_Latn  — Ashaninka

NOTE: Embedding resizing (model.resize_token_embeddings()) is performed
inside train_lora.py after the base model is loaded, so that the new
embedding rows can be initialized from the mean of existing rows while
the full model weights are in memory.

Usage:
    python scripts/modify_tokenizer.py
    python scripts/modify_tokenizer.py --output_dir models/v0/tokenizer

HUMAN REVIEW GATE  ►  After running, verify:
    • models/v0/tokenizer/tokenizer_config.json  exists
    • The file contains all 3 new tokens in additional_special_tokens
    • Running:  python -c "from transformers import NllbTokenizer; \\
        t = NllbTokenizer.from_pretrained('models/v0/tokenizer'); \\
        print('agr_Latn' in t.additional_special_tokens)"
      prints True
"""

import argparse
import json
from pathlib import Path

from transformers import NllbTokenizer


# ---------------------------------------------------------------------------
# New language tokens to add to the NLLB-200 vocabulary
# ---------------------------------------------------------------------------

NEW_LANGUAGE_TOKENS = [
    "shp_Latn",  # Shipibo-Konibo
    "agr_Latn",  # Awajún
    "cni_Latn",  # Ashaninka
]

# Tokens confirmed present natively in NLLB-200 (for reference / logging)
NATIVE_NLLB_TOKENS = [
    "quy_Latn",  # Southern Quechua
    "ayr_Latn",  # Central Aymara
    "grn_Latn",  # Guaraní
    "spa_Latn",  # Spanish
    "eng_Latn",  # English
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add AmericasNLP language tokens to the NLLB-200 tokenizer"
    )
    parser.add_argument(
        "--base_model",
        type=str,
        default="facebook/nllb-200-3.3B",
        help="HuggingFace model ID for the base tokenizer",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Where to save the modified tokenizer. Default: <project_root>/models/v0/tokenizer",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    output_dir = Path(args.output_dir) if args.output_dir else project_root / "models" / "v0" / "tokenizer"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  NLLB-200 LoRA Pipeline — Step 2: Modify Tokenizer")
    print(f"  Base model : {args.base_model}")
    print(f"  Output dir : {output_dir}")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # 1. Load base tokenizer
    # ------------------------------------------------------------------
    print(f"Loading tokenizer from '{args.base_model}' …")
    tokenizer = NllbTokenizer.from_pretrained(args.base_model)
    original_vocab_size = len(tokenizer)
    print(f"  Original vocabulary size: {original_vocab_size:,}")

    # ------------------------------------------------------------------
    # 2. Verify which native tokens already exist
    # ------------------------------------------------------------------
    print("\nVerifying native NLLB-200 language tokens:")
    all_special = set(tokenizer.additional_special_tokens)
    for tok in NATIVE_NLLB_TOKENS:
        present = tok in all_special
        status = "✓ present" if present else "✗ MISSING"
        print(f"  {tok:<18} {status}")

    # ------------------------------------------------------------------
    # 3. Add new language tokens
    # ------------------------------------------------------------------
    print("\nAdding new language tokens:")
    truly_new = [tok for tok in NEW_LANGUAGE_TOKENS if tok not in all_special]
    already_present = [tok for tok in NEW_LANGUAGE_TOKENS if tok in all_special]

    if already_present:
        print(f"  [SKIP] Already in vocab: {already_present}")

    if truly_new:
        num_added = tokenizer.add_special_tokens(
            {"additional_special_tokens": tokenizer.additional_special_tokens + truly_new}
        )
        print(f"  [OK] Added {num_added} new tokens:")
        for tok in truly_new:
            token_id = tokenizer.convert_tokens_to_ids(tok)
            print(f"       {tok:<18} → ID {token_id}")
    else:
        print("  All new tokens already present — nothing to add.")

    new_vocab_size = len(tokenizer)
    print(f"\n  Vocabulary size: {original_vocab_size:,} → {new_vocab_size:,}")

    # ------------------------------------------------------------------
    # 4. Save modified tokenizer
    # ------------------------------------------------------------------
    print(f"\nSaving tokenizer to {output_dir} …")
    tokenizer.save_pretrained(str(output_dir))

    # Also save a human-readable token map for reference
    token_map = {
        tok: tokenizer.convert_tokens_to_ids(tok)
        for tok in NATIVE_NLLB_TOKENS + NEW_LANGUAGE_TOKENS
    }
    token_map_path = output_dir / "lang_token_ids.json"
    with open(token_map_path, "w", encoding="utf-8") as f:
        json.dump(token_map, f, indent=2, ensure_ascii=False)
    print(f"  Language token ID map saved → {token_map_path.name}")

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    print(f"\n{'='*60}")
    print(f"  Done. Summary of language token IDs:")
    for lang, token_id in token_map.items():
        tag = "[native]" if lang in NATIVE_NLLB_TOKENS else "[new]   "
        print(f"    {tag} {lang:<18} ID = {token_id}")
    print(f"\n  ► HUMAN REVIEW GATE: check output at {output_dir}")
    print()


if __name__ == "__main__":
    main()
