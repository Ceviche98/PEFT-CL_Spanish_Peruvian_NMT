#!/usr/bin/env bash
# =============================================================================
# run_experiment_A_train.sh
# =============================================================================
# PHASE 1 — Experiment A: PEFT Benchmark Training
#
# Runs all Experiment A training methods sequentially on a single GPU:
#   A) LoRA benchmarks (r=32 / r=128 / r=128 QKVO)
#   B) PiSSA (r=128)
#   C) Full Fine-Tuning — Unfrozen embeddings  [uses transformers==4.40.2]
#   D) A-Bridge QKVO v3 (r=32 and r=128 joint multilingual)
#
# Prerequisites:
#   • bash run_setup.sh  (data and tokenizer must exist)
#   • NLLB-200-1.3B model available (locally or via HuggingFace hub)
#
# Tunable environment overrides (all have sane defaults):
#   BATCH_SIZE, GRAD_ACCUM, LR, MAX_STEPS, PATIENCE, EXP
#
# Hardware target: Single NVIDIA GPU (24 GB+ recommended for FFT)
#
# Usage:
#   bash run_experiment_A_train.sh          # run everything
#   SKIP_FFT=1 bash run_experiment_A_train.sh  # skip FFT (low VRAM)
#
# Resuming from a checkpoint:
#   Each training script auto-detects the last HuggingFace checkpoint under
#   models/<EXP>/<method>/lr<LR>/hf/ via get_last_checkpoint().
#   Simply re-run the same command — training resumes automatically.
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_XET_HIGH_PERFORMANCE=0
export HF_HUB_ENABLE_HF_TRANSFER=0

# ── Hyper-parameters ────────────────────────────────────────────────────────
# Experiment A benchmarks (LoRA / PiSSA)
EXP="${EXP:-experiment_A}"
BATCH_SIZE="${BATCH_SIZE:-8}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"          # Effective batch = 16
# 489 034 filtered pairs / eff-batch 16 / 20 joint-data passes ≈ 611 300 steps
MAX_STEPS="${MAX_STEPS:-611300}"
PATIENCE="${PATIENCE:-5}"

# A-Bridge QKVO v3
BRIDGE_BATCH="${BRIDGE_BATCH:-8}"
BRIDGE_GRAD="${BRIDGE_GRAD:-2}"        # Must keep product == 16
BRIDGE_STEPS="${BRIDGE_STEPS:-611280}"
BRIDGE_PATIENCE="${BRIDGE_PATIENCE:-5}"
BRIDGE_LR="${BRIDGE_LR:-1e-4}"
BRIDGE_EXP="${BRIDGE_EXP:-experiment_A_bridge_qkvo_v3}"

SKIP_FFT="${SKIP_FFT:-0}"

# ── Install / confirm dependencies ──────────────────────────────────────────
echo "=== Installing dependencies ==="
pip install -U pip "setuptools<70.0.0" wheel
pip install --prefer-binary -r requirements.txt

# ── A) LoRA benchmarks ──────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo " [A] LoRA r=32  (Q, V projections)"
echo "========================================================================"
python scripts/experiment_A/train_benchmarks.py \
    --method lora --lora_r 32 \
    --lr 1e-4 --max_steps "${MAX_STEPS}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH_SIZE}" --grad_accum "${GRAD_ACCUM}" \
    --experiment_name "${EXP}"

echo ""
echo "========================================================================"
echo " [A] LoRA r=128  (Q, V projections)"
echo "========================================================================"
python scripts/experiment_A/train_benchmarks.py \
    --method lora --lora_r 128 \
    --lr 1e-4 --max_steps "${MAX_STEPS}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH_SIZE}" --grad_accum "${GRAD_ACCUM}" \
    --experiment_name "${EXP}"

echo ""
echo "========================================================================"
echo " [A] LoRA r=128 QKVO  (Q, K, V, O projections)"
echo "========================================================================"
python scripts/experiment_A/train_benchmarks.py \
    --method lora --lora_r 128 --target_modules q_proj k_proj v_proj o_proj \
    --lr 1e-4 --max_steps "${MAX_STEPS}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH_SIZE}" --grad_accum "${GRAD_ACCUM}" \
    --experiment_name "${EXP}"

# ── B) PiSSA ────────────────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo " [B] PiSSA r=128"
echo "========================================================================"
python scripts/experiment_A/train_benchmarks_pissa.py \
    --lora_r 128 \
    --lr 1e-4 --max_steps "${MAX_STEPS}" \
    --patience "${PATIENCE}" \
    --batch_size "${BATCH_SIZE}" --grad_accum "${GRAD_ACCUM}" \
    --experiment_name "${EXP}"

# ── C) FFT Unfrozen ─────────────────────────────────────────────────────────
if [[ "${SKIP_FFT}" != "1" ]]; then
    echo ""
    echo "========================================================================"
    echo " [C] Full Fine-Tuning — Unfrozen embeddings"
    echo "     NOTE: Requires transformers==4.40.2 (pinned for this run)"
    echo "========================================================================"
    # Temporarily install the pinned transformers version for FFT
    pip install "transformers==4.40.2" --quiet

    FFT_STEPS=20375   # 5 passes at effective batch 120 (12×10)
    python scripts/experiment_A/train_fft_unfrozen_clean.py \
        --lr 1e-5 \
        --optimizer adafactor \
        --max_steps "${FFT_STEPS}" \
        --experiment_name "${EXP}" \
        --patience 99 \
        --weight_decay 0.001 \
        --sampling_temperature 0.7 \
        --label_smoothing 0.1 \
        --batch_size 12 --grad_accum 10 \
        --num_workers 12 \
        --eval_num_beams 1 \
        --run_tag "adafactor_lr1e-5_wd1e-3_t0.7_b12x10_5ep"

    # Restore full dependencies for subsequent scripts
    pip install --prefer-binary -r requirements.txt --quiet
else
    echo ""
    echo "[C] FFT skipped (SKIP_FFT=1)"
fi

# ── D) A-Bridge QKVO v3 ─────────────────────────────────────────────────────
echo ""
echo "========================================================================"
echo " [D] A-Bridge QKVO v3 — ranks 32 and 128 (joint multilingual control)"
echo "========================================================================"
for RANK in 32 128; do
    echo "=== A-Bridge v3 r=${RANK} ==="
    python scripts/experiment_A/train_abridge_qkvo_v3.py \
        --lora_r "${RANK}" \
        --lr "${BRIDGE_LR}" \
        --max_steps "${BRIDGE_STEPS}" \
        --patience "${BRIDGE_PATIENCE}" \
        --batch_size "${BRIDGE_BATCH}" \
        --grad_accum "${BRIDGE_GRAD}" \
        --experiment_name "${BRIDGE_EXP}"
done

echo ""
echo "========================================================================"
echo " Experiment A training complete."
echo " Checkpoints saved under:  models/${EXP}/"
echo " Next step: bash run_experiment_A_eval.sh"
echo "========================================================================"
