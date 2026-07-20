#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

EXP_NAME="${EXP_NAME:-}"
if [[ $# -gt 0 && "$1" != *=* ]]; then
    EXP_NAME="$1"
    shift
fi

# Select GPUs with GPUS=0,2 (comma-separated indices); overrides CUDA_VISIBLE_DEVICES.
GPUS="${GPUS:-4,5,6,7}"
export CUDA_VISIBLE_DEVICES="$GPUS"

OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
EXP_DIR="${EXP_DIR:-${OUTPUT_DIR}/${RUN_ID}_${EXP_NAME}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${BATCH_SIZE:-64}}"
MAX_STEPS="${MAX_STEPS:-200000}"
SAVE_EVERY_STEP="${SAVE_EVERY_STEP:-5000}"

count_visible_gpus() {
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
        echo 0
        return
    fi

    local count=0
    local IFS=','
    local gpu
    read -ra gpus <<< "$CUDA_VISIBLE_DEVICES"
    for gpu in "${gpus[@]}"; do
        gpu="${gpu//[[:space:]]/}"
        if [[ -n "$gpu" && "$gpu" != "-1" ]]; then
            count=$((count + 1))
        fi
    done
    echo "$count"
}

if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]] && command -v nvidia-smi >/dev/null 2>&1; then
    if GPU_LIST="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | tr -d ' ' | paste -sd, -)" && [[ -n "$GPU_LIST" ]]; then
        CUDA_VISIBLE_DEVICES="$GPU_LIST"
        export CUDA_VISIBLE_DEVICES
    fi
fi

GPU_COUNT="$(count_visible_gpus)"
if [[ "$GPU_COUNT" -eq 0 && -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
    GPU_COUNT=1
fi

GPU_ARG="gpu=0"
if [[ "$GPU_COUNT" -eq 0 ]]; then
    GPU_ARG="gpu=null"
fi

if ! [[ "$GLOBAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
    echo "BATCH_SIZE/GLOBAL_BATCH_SIZE must be a positive integer, got '${GLOBAL_BATCH_SIZE}'." >&2
    exit 1
fi

BATCH_DIVISOR="$GPU_COUNT"
if [[ "$BATCH_DIVISOR" -lt 1 ]]; then
    BATCH_DIVISOR=1
fi

if [[ $((GLOBAL_BATCH_SIZE % BATCH_DIVISOR)) -ne 0 ]]; then
    echo "Global batch size ${GLOBAL_BATCH_SIZE} is not divisible by ${BATCH_DIVISOR} GPU process(es)." >&2
    echo "Set BATCH_SIZE/GLOBAL_BATCH_SIZE to a divisible value, or restrict CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi
PER_GPU_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / BATCH_DIVISOR))

if [[ "$GPU_COUNT" -gt 1 ]]; then
    LAUNCH=(torchrun --standalone --nproc_per_node="$GPU_COUNT")
else
    LAUNCH=(python)
fi

echo "Launching training with ${GPU_COUNT} GPU process(es). CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-unset}"
echo "Global batch size: ${GLOBAL_BATCH_SIZE}; per-GPU batch size: ${PER_GPU_BATCH_SIZE}"
echo "Experiment directory: ${EXP_DIR}"

cd "$REPO_ROOT"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${LAUNCH[@]}" pipeline/step8_3d_training/train_no_diff.py \
            hydra/job_logging=none \
            hydra/hydra_logging=none \
            exp_dir="${EXP_DIR}" \
            platform=TensorBoard \
            task=contact_gen \
            task.train.batch_size="${PER_GPU_BATCH_SIZE}" \
            task.train.max_steps="${MAX_STEPS}" \
            task.train.save_every_step="${SAVE_EVERY_STEP}" \
            task.dataset.num_points=8192 \
            model=cdm \
            "$GPU_ARG" \
            "$@"
