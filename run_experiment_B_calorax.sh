#!/usr/bin/env bash
# =============================================================================
# run_experiment_B_calorax.sh
# =============================================================================
# PHASE 3b — Experiment B: Sequential CaLoRA-X
#
# CaLoRA-X extends standard CaLoRA with:
#   • Compact representative gradient memory (multi-snapshot SVD bases)
#   • Continuous conflict-gated soft projection (λ-attenuated)
# rather than the single boundary-gradient hard projection used by CaLoRA.
#
# Prerequisites:
#   • bash run_setup.sh  (data + tokenizer ready)
#   • run_experiment_B_calora.sh should have completed (or at least order2)
#     since CaLoRA-X was primarily evaluated on order2.
#
# Tunable environment overrides:
#   ORDER (default order2)
#   BATCH_SIZE (default 6)   GRAD_ACCUM (default 20)   Effective batch = 120
#   LR (default 1e-4)   LORA_R (default 128)   LORA_ALPHA (default 256)
#   SEED (default 42)   RUN_TAG   PACA_OFF   PACA_WARMUP_STEPS
#   X_MEMORY_SAMPLES (default 4)      X_MEMORY_START_FRACTION (default 0.4)
#   X_MEMORY_RANK (default 8)         X_ATTENUATION (default 0.5)
#   X_MIN_SCALE (default 0.1)
#
# Resuming from a checkpoint:
#   CaLoRA-X saves per-task adapters and representative memories under:
#       models/experiment_B/<run_tag>/task_<N>_<lang>/
#   To resume from task N: START_TASK=N bash run_experiment_B_calorax.sh
#
# Usage:
#   bash run_experiment_B_calorax.sh
#   RUN_TAG=calorax_order2_v1 SEED=42 bash run_experiment_B_calorax.sh
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_XET_HIGH_PERFORMANCE=0
export HF_HUB_ENABLE_HF_TRANSFER=0

# ── Hyper-parameters ────────────────────────────────────────────────────────
ORDER="${ORDER:-order2}"
BATCH_SIZE="${BATCH_SIZE:-6}"
GRAD_ACCUM="${GRAD_ACCUM:-20}"    # Effective batch = 120
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"
SEED="${SEED:-42}"
START_TASK="${START_TASK:-1}"
END_TASK="${END_TASK:-5}"
PACA_OFF="${PACA_OFF:-0}"
PACA_WARMUP_STEPS="${PACA_WARMUP_STEPS:-0}"
RUN_TAG="${RUN_TAG:-calorax_qkvo_r${LORA_R}_${ORDER}_seed${SEED}}"

# CaLoRA-X memory / projection parameters
X_MEMORY_SAMPLES="${X_MEMORY_SAMPLES:-4}"
X_MEMORY_START_FRACTION="${X_MEMORY_START_FRACTION:-0.4}"
X_MEMORY_RANK="${X_MEMORY_RANK:-8}"
X_ATTENUATION="${X_ATTENUATION:-0.5}"
X_MIN_SCALE="${X_MIN_SCALE:-0.1}"

# ── Install / confirm dependencies ──────────────────────────────────────────
echo "=== Installing dependencies ==="
pip install -U pip "setuptools<70.0.0" wheel
pip install --prefer-binary -r requirements.txt

# ── Build extra args ─────────────────────────────────────────────────────────
EXTRA_ARGS=(
    --paca_warmup_steps "${PACA_WARMUP_STEPS}"
    --x_memory_samples  "${X_MEMORY_SAMPLES}"
    --x_memory_start_fraction "${X_MEMORY_START_FRACTION}"
    --x_memory_rank     "${X_MEMORY_RANK}"
    --x_attenuation     "${X_ATTENUATION}"
    --x_min_scale       "${X_MIN_SCALE}"
)
[[ "${PACA_OFF}" == "1" ]] && EXTRA_ARGS+=(--disable_paca_correction)
[[ -n "${RUN_TAG}" ]]      && EXTRA_ARGS+=(--run_tag "${RUN_TAG}")

echo ""
echo "========================================================================"
echo " Experiment B: Sequential CaLoRA-X (Q/K/V/O | r=${LORA_R})"
echo " Order: ${ORDER} | Tasks: ${START_TASK}..${END_TASK} | Seed: ${SEED}"
echo " Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
echo " Memory: ${X_MEMORY_SAMPLES} snapshots after ${X_MEMORY_START_FRACTION}; rank=${X_MEMORY_RANK}"
echo " Soft projection: lambda=${X_ATTENUATION}; min scale=${X_MIN_SCALE}"
echo " PaCA: $([[ "${PACA_OFF}" == "1" ]] && echo OFF || echo ON) | Run tag: ${RUN_TAG}"
echo "========================================================================"

python scripts/experiment_B/train_calora_x_sequential.py \
    --order "${ORDER}" \
    --start_task "${START_TASK}" \
    --end_task   "${END_TASK}" \
    --lora_r     "${LORA_R}" \
    --lora_alpha "${LORA_ALPHA}" \
    --lr         "${LR}" \
    --batch_size "${BATCH_SIZE}" \
    --grad_accum "${GRAD_ACCUM}" \
    --seed       "${SEED}" \
    --experiment_name "experiment_B" \
    "${EXTRA_ARGS[@]}"

# ── BWT / FM / AA ────────────────────────────────────────────────────────────
MATRIX="models/experiment_B/${RUN_TAG}/triangular_eval_matrix.json"
if [[ -f "${MATRIX}" ]]; then
    echo ""
    echo "--- BWT / FM / AA for CaLoRA-X (${ORDER}) ---"
    python scripts/experiment_B/calculate_bwt_from_matrix.py \
        "${MATRIX}" --order "${ORDER}"
fi

echo ""
echo "========================================================================"
echo " Experiment B (CaLoRA-X) complete."
echo " Checkpoints saved under: models/experiment_B/${RUN_TAG}/"
echo "========================================================================"
