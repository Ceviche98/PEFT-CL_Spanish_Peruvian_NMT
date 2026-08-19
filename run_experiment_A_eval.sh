#!/usr/bin/env bash
# =============================================================================
# run_experiment_A_eval.sh
# =============================================================================
# PHASE 2 — Experiment A: Evaluation
#
# Runs all Experiment A evaluation scripts in order:
#   1. Beam-4 evaluation on 5-language dev sets (chrF++ + BLEU)
#   2. FLORES-200 catastrophic forgetting benchmark (Spanish→X, ~200 languages)
#   3. Embedding integrity verification (checks trained embeddings vs base model)
#
# Prerequisites:
#   • bash run_experiment_A_train.sh  (trained checkpoints must exist)
#   • flores200/ directory must exist. Download from:
#       https://github.com/facebookresearch/flores/tree/main
#     and place the dev/ and devtest/ folders under flores200/
#
# Tunable environment overrides:
#   BATCH_SIZE (default 16)  NUM_BEAMS (default 4)  MODELS_DIR  DATA_DIR
#
# Usage:
#   bash run_experiment_A_eval.sh
#   BATCH_SIZE=8 bash run_experiment_A_eval.sh   # reduce if OOM
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_XET_HIGH_PERFORMANCE=0
export HF_HUB_ENABLE_HF_TRANSFER=0

BATCH_SIZE="${BATCH_SIZE:-16}"
NUM_BEAMS="${NUM_BEAMS:-4}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-256}"
MODELS_DIR="${MODELS_DIR:-$(pwd)/models}"
DATA_DIR="${DATA_DIR:-$(pwd)/data_in}"

# ── Install / confirm dependencies ──────────────────────────────────────────
echo "=== Installing dependencies ==="
pip install -U pip "setuptools<70.0.0" wheel
pip install --prefer-binary -r requirements.txt

# ── 1. Beam-4 evaluation on AmericasNLP dev sets ────────────────────────────
echo ""
echo "========================================================================"
echo " [1/3] Beam-4 evaluation — 5-language dev sets (chrF++ + BLEU)"
echo "       Models: lora_r32  lora_r128  lora_r128_qkvo  pissa_r128"
echo "========================================================================"
python scripts/experiment_A/evaluate_all_experiment_a_beam4.py \
    --models_dir "${MODELS_DIR}" \
    --data_dir   "${DATA_DIR}" \
    --output_dir "${MODELS_DIR}/experiment_A/final/experiment_A_beam4_evaluation" \
    --batch_size "${BATCH_SIZE}" \
    --num_beams  "${NUM_BEAMS}" \
    --max_new_tokens "${MAX_NEW_TOKENS}" \
    --decoder_scale_mode auto \
    --models lora_r32 lora_r128 lora_r128_qkvo pissa_r128

# ── 2. FLORES-200 forgetting benchmark ──────────────────────────────────────
echo ""
echo "========================================================================"
echo " [2/3] FLORES-200 catastrophic forgetting benchmark (spa→X)"
echo "       Requires flores200/ directory — see README §Prerequisites"
echo "========================================================================"
python scripts/experiment_A/evaluate_flores200_forgetting_spa.py \
    --batch_size "${BATCH_SIZE}" \
    --num_beams  "${NUM_BEAMS}"

# ── 3. Embedding integrity check ────────────────────────────────────────────
echo ""
echo "========================================================================"
echo " [3/3] Embedding integrity verification for Experiment A models"
echo "========================================================================"
python scripts/experiment_A/check_final_embeddings.py

echo ""
echo "========================================================================"
echo " Experiment A evaluation complete."
echo " Results written to:  models/experiment_A/final/"
echo "                      data_out/experiment_A_forgetting_spa/"
echo "========================================================================"
