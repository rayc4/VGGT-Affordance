#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

usage() {
    cat >&2 <<EOF
Usage:
  $0 <experiment-name> <processed-sam2-dir> [hydra overrides...]
  EXP_NAME=<experiment-name> $0 <processed-sam2-dir> [hydra overrides...]
  EXP_NAME=<experiment-name> TASA_PROCESSED_SAM2_DIR=<dir> $0 [hydra overrides...]
EOF
}

EXP_NAME="${EXP_NAME:-}"
TASA_PROCESSED_SAM2_DIR="${TASA_PROCESSED_SAM2_DIR:-scenefun3d/processed_sam2}"

# Preserve the original two-positional-argument interface. When exactly one
# leading positional value is supplied, fill whichever required value was not
# provided through the environment.
if [[ $# -ge 2 && "$1" != *=* && "$2" != *=* ]]; then
    EXP_NAME="$1"
    TASA_PROCESSED_SAM2_DIR="$2"
    shift 2
elif [[ $# -ge 1 && "$1" != *=* ]]; then
    if [[ -n "$EXP_NAME" && -z "$TASA_PROCESSED_SAM2_DIR" ]]; then
        TASA_PROCESSED_SAM2_DIR="$1"
    elif [[ -z "$EXP_NAME" && -n "$TASA_PROCESSED_SAM2_DIR" ]]; then
        EXP_NAME="$1"
    else
        echo "Cannot determine whether '$1' is the experiment name or processed dataset directory." >&2
        usage
        exit 1
    fi
    shift
fi

if [[ -z "$EXP_NAME" || -z "$TASA_PROCESSED_SAM2_DIR" ]]; then
    echo "Both EXP_NAME and TASA_PROCESSED_SAM2_DIR are required." >&2
    usage
    exit 1
fi
if [[ $# -gt 0 && "$1" != *=* ]]; then
    echo "Unexpected positional argument: $1" >&2
    usage
    exit 1
fi
export TASA_PROCESSED_SAM2_DIR

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required to activate the 'pt' environment." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "Conda activation script not found: ${CONDA_SH}" >&2
    exit 1
fi
# shellcheck source=/dev/null
set +u
if ! source "$CONDA_SH"; then
    set -u
    echo "Unable to initialize Conda from ${CONDA_SH}." >&2
    exit 1
fi
if conda activate pt; then
    CONDA_ACTIVATED=1
else
    CONDA_ACTIVATED=0
fi
set -u
if [[ "$CONDA_ACTIVATED" -ne 1 ]]; then
    echo "Unable to activate the Conda environment 'pt'." >&2
    exit 1
fi
echo "Conda environment: ${CONDA_DEFAULT_ENV}"

# Select GPUs with GPUS=0,2 (comma-separated indices); overrides CUDA_VISIBLE_DEVICES.
GPUS="${GPUS:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES="$GPUS"

OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
EXP_DIR="${EXP_DIR:-${OUTPUT_DIR}/${RUN_ID}_${EXP_NAME}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-${BATCH_SIZE:-64}}"
MAX_STEPS="${MAX_STEPS:-50000}"

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
echo "Processed dataset: ${TASA_PROCESSED_SAM2_DIR}"

cd "$REPO_ROOT"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
"${LAUNCH[@]}" pipeline/step8_3d_training/train_no_diff.py \
            hydra/job_logging=none \
            hydra/hydra_logging=none \
            exp_name="${EXP_NAME}" \
            exp_dir="${EXP_DIR}" \
            platform=TensorBoard \
            task=contact_gen \
            task.train.batch_size="${PER_GPU_BATCH_SIZE}" \
            task.train.max_steps="${MAX_STEPS}" \
            task.dataset.num_points=8192 \
            model=cdm \
            "$GPU_ARG" \
            "$@"
