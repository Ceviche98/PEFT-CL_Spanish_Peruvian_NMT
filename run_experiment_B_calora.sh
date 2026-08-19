#!/usr/bin/env bash
# =============================================================================
# run_experiment_B_calora.sh
# =============================================================================
# PHASE 3 — Experiment B: Sequential Continual Learning (LoRA + CaLoRA)
#
# Runs the paired LoRA vs CaLoRA sequential comparison across the 5 Peruvian
# Indigenous languages for two language orderings:
#
#   order1 (Andino → Amazónico): quy → ayr → shp → agr → cni
#   order2 (Amazónico → Andino): cni → agr → shp → ayr → quy
#
# For each ordering, both plain sequential LoRA (baseline) and sequential
# CaLoRA (CaGA + PaCA) are run so that the Triangular Evaluation Matrix,
# Forgetting Measure (FM), and Backward Transfer (BWT) can be computed.
#
# After each training run, calculate_bwt_from_matrix.py is called
# automatically to print AA / FM / BWT from the saved triangular_eval_matrix.json.
#
# Prerequisites:
#   • bash run_setup.sh  (data + tokenizer ready)
#   • NLLB-200-1.3B base model available (locally or via HuggingFace hub)
#
# Tunable environment overrides (all have sane defaults):
#   BATCH_SIZE (default 10)   GRAD_ACCUM (default 12)   Effective batch = 120
#   LR (default 1e-4)   LORA_R (default 128)   LORA_ALPHA (default 256)
#   SEED (default 42)
#   PACA_OFF=1  →  disables PaCA correction (raw-gradient diagnostic control)
#   START_TASK=N  →  resume from task N (see "Resuming from a checkpoint" below)
#   END_TASK=N    →  stop after task N
#   RUN_TAG       →  string appended to output directory to avoid overwriting
#
# Resuming from a checkpoint:
#   Each language task saves its best adapter checkpoint under:
#       models/experiment_B/<run_tag>/task_<N>_<lang>/best_checkpoint/
#   To resume from task 3 (e.g. after a crash on shp):
#       START_TASK=3 bash run_experiment_B_calora.sh
#   The script will load the task-2 adapter as its starting point.
#
# Usage:
#   bash run_experiment_B_calora.sh           # order1 + order2
#   ORDER=order1 bash run_experiment_B_calora.sh   # one order only
# =============================================================================

set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

export HF_TOKEN="YOUR_HF_TOKEN_HERE"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_XET_HIGH_PERFORMANCE=0
export HF_HUB_ENABLE_HF_TRANSFER=0

# ── Hyper-parameters ────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-10}"
GRAD_ACCUM="${GRAD_ACCUM:-12}"    # Effective batch = 120
LR="${LR:-1e-4}"
LORA_R="${LORA_R:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"
SEED="${SEED:-42}"
START_TASK="${START_TASK:-1}"
END_TASK="${END_TASK:-5}"
PACA_OFF="${PACA_OFF:-0}"
PACA_WARMUP_STEPS="${PACA_WARMUP_STEPS:-0}"
RUN_TAG="${RUN_TAG:-}"

# Which orderings to run (space-separated, e.g. "order1" or "order1 order2")
ORDERS="${ORDER:-order1 order2}"

# ── Install / confirm dependencies ──────────────────────────────────────────
echo "=== Installing dependencies ==="
pip install -U pip "setuptools<70.0.0" wheel
pip install --prefer-binary -r requirements.txt

# ── Helper: build PACA / run-tag args ───────────────────────────────────────
build_paca_args() {
    local args=(--paca_warmup_steps "${PACA_WARMUP_STEPS}")
    [[ "${PACA_OFF}" == "1" ]] && args+=(--disable_paca_correction)
    [[ -n "${RUN_TAG}" ]]      && args+=(--run_tag "${RUN_TAG}")
    echo "${args[@]}"
}

# ── Run for each ordering ────────────────────────────────────────────────────
for ORDER in ${ORDERS}; do
    echo ""
    echo "========================================================================"
    echo " Experiment B — Order: ${ORDER}"
    echo " LoRA Rank: ${LORA_R} | Alpha: ${LORA_ALPHA} | LR: ${LR}"
    echo " Batch: ${BATCH_SIZE} x ${GRAD_ACCUM} = $((BATCH_SIZE * GRAD_ACCUM))"
    echo " Tasks: ${START_TASK}..${END_TASK} | Seed: ${SEED}"
    echo "========================================================================"

    LORA_TAG="${RUN_TAG:-lora_qkvo_r${LORA_R}_${ORDER}_seed${SEED}}"
    CALORA_TAG="${RUN_TAG:-calora_qkvo_r${LORA_R}_${ORDER}_seed${SEED}}"

    # ── 1/2: Plain sequential LoRA baseline ──────────────────────────────────
    echo ""
    echo "--- [1/2] Sequential LoRA baseline (${ORDER}) ---"
    python scripts/experiment_B/train_lora_sequential.py \
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
        --run_tag "${LORA_TAG}"

    # Compute BWT / FM / AA from the triangular evaluation matrix
    LORA_MATRIX="models/experiment_B/${LORA_TAG}/triangular_eval_matrix.json"
    if [[ -f "${LORA_MATRIX}" ]]; then
        echo ""
        echo "--- BWT / FM / AA for Sequential LoRA (${ORDER}) ---"
        python scripts/experiment_B/calculate_bwt_from_matrix.py \
            "${LORA_MATRIX}" --order "${ORDER}"
    fi

    # ── 2/2: Sequential CaLoRA ───────────────────────────────────────────────
    echo ""
    echo "--- [2/2] Sequential CaLoRA (${ORDER}) ---"
    PACA_ARGS=($(build_paca_args))
    python scripts/experiment_B/train_calora_sequential.py \
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
        --run_tag "${CALORA_TAG}" \
        "${PACA_ARGS[@]}"

    CALORA_MATRIX="models/experiment_B/${CALORA_TAG}/triangular_eval_matrix.json"
    if [[ -f "${CALORA_MATRIX}" ]]; then
        echo ""
        echo "--- BWT / FM / AA for Sequential CaLoRA (${ORDER}) ---"
        python scripts/experiment_B/calculate_bwt_from_matrix.py \
            "${CALORA_MATRIX}" --order "${ORDER}"
    fi

    echo ""
    echo "=== ${ORDER} complete ==="
done

echo ""
echo "========================================================================"
echo " Experiment B (CaLoRA) complete."
echo " Checkpoints saved under: models/experiment_B/"
echo " To compute BWT/FM/AA manually on any run:"
echo "   python scripts/experiment_B/calculate_bwt_from_matrix.py \\"
echo "       models/experiment_B/<run_tag>/triangular_eval_matrix.json"
echo "========================================================================"
