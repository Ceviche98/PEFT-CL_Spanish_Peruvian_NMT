#!/usr/bin/env bash
# =============================================================================
# run_setup.sh
# =============================================================================
# PHASE 0 — Environment, Data Download, Preprocessing & Tokenizer Setup
#
# This script performs every step required before any training run:
#   1. Install Python dependencies
#   2. Download AmericasNLP corpora (git clone + zip downloads)
#   3. Preprocess & normalize data → data_in/train/ and data_in/dev/
#   4. Add shp_Latn / agr_Latn / cni_Latn tokens to the NLLB-200 tokenizer
#   5. Filter sentence pairs exceeding 256 tokens & summarise dataset counts
#   6. (Optional) Measure token fertility for the morphological analysis
#
# Usage:
#   bash run_setup.sh
#
# After this script completes, you can run run_experiment_A_train.sh or
# run_experiment_B_calora.sh.
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# ---------------------------------------------------------------------------
# Hugging Face authentication (needed to download the base model later)
# Replace with your personal token from https://huggingface.co/settings/tokens
export HF_TOKEN="YOUR_HF_TOKEN_HERE"
# ---------------------------------------------------------------------------

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# ── Step 1: Install dependencies ────────────────────────────────────────────
echo ""
echo "=== [1/6] Installing Python dependencies ==="
pip install -U pip "setuptools<70.0.0" wheel
pip install --prefer-binary -r requirements.txt

# ── Step 2: Download raw corpora ────────────────────────────────────────────
echo ""
echo "=== [2/6] Downloading AmericasNLP corpora ==="
# Clones americasnlp2025, americasnlp2021-st, REPUcs-AmericasNLP2021
# and downloads the NLLB-Seed zip archive into data_in/raw/
python scripts/download_raw_data.py

# ── Step 3: Preprocess & normalize ─────────────────────────────────────────
echo ""
echo "=== [3/6] Preprocessing data (all 5 languages) ==="
# Applies Unicode NFC + MosesPunctNormalizer, removes blank-pair lines,
# and writes aligned train/dev splits to data_in/train/ and data_in/dev/
python scripts/preprocess_data.py --languages specific \
    --lang_list quy,ayr,shp,agr,cni

# ── Step 4: Modify tokenizer ────────────────────────────────────────────────
echo ""
echo "=== [4/6] Adding new language tokens to NLLB-200 tokenizer ==="
# Adds shp_Latn, agr_Latn, cni_Latn to facebook/nllb-200-1.3B tokenizer
# Output: models/v0/tokenizer/
python scripts/modify_tokenizer.py

# ── Step 5: Filter & count ──────────────────────────────────────────────────
echo ""
echo "=== [5/6] Filtering sentence pairs (max 256 tokens) ==="
# Writes data_in/train/<lang>/train.filtered.<lang|es> and
# produces data_in/summary_counts.txt with retention statistics
python scripts/experiment_A/filter_and_count.py

# ── Step 6: Token fertility ─────────────────────────────────────────────────
echo ""
echo "=== [6/6] Computing token fertility (morphological complexity) ==="
# Measures Fertility = Subwords / Words for each language
# Saves results to scripts/experiment_A/token_fertility.txt
python scripts/experiment_A/token_fertilizer.py

echo ""
echo "========================================================================"
echo " Setup complete. Next steps:"
echo "   • (Optional) Download the NLLB-200-1.3B base model locally:"
echo "       python scripts/download_nllb_model.py"
echo "   • Run Experiment A training:"
echo "       bash run_experiment_A_train.sh"
echo "========================================================================"
