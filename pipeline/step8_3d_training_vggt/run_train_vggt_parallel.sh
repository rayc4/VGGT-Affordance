#!/usr/bin/env bash
# Launch VGGT-conditioned Step 8 training with one DDP process per GPU.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  pipeline/step8_3d_training_vggt/run_train_vggt_parallel.sh [--dry-run] \
    [<experiment-name> <processed-sam2-dir>] [hydra overrides...]

Examples:
  GPUS=0,1,2,3 pipeline/step8_3d_training_vggt/run_train_vggt_parallel.sh \
    vggt-gated scenefun3d/processed_sam2

  GPUS="0 2" GLOBAL_BATCH_SIZE=32 EXP_NAME=vggt_two_gpu \
    pipeline/step8_3d_training_vggt/run_train_vggt_parallel.sh \
      task.train.max_steps=25000

Environment overrides:
  GPUS                    Comma/space-separated GPU IDs. Defaults to
                          CUDA_VISIBLE_DEVICES, then all nvidia-smi GPUs.
  ENV_NAME                Conda environment (default: pt).
  EXP_NAME                Experiment name (default: vggt_refine).
  OUTPUT_DIR              Output root (default: outputs).
  EXP_DIR                 Exact shared run directory. Defaults to
                          <OUTPUT_DIR>/<timestamp>_<EXP_NAME>.
  GLOBAL_BATCH_SIZE       Train batch across all GPU processes (default: 64).
  GLOBAL_VAL_BATCH_SIZE   Validation batch across all processes (default: 32).
  NUM_WORKERS             DataLoader workers per process (default: 4).
  VAL_NUM_WORKERS         Validation workers per process (default: 2).
  DATA_ROOT               Dataset root (default: scenefun3d).
  PROCESSED_SAM2_DIR      Processed frame tree (default:
                          <DATA_ROOT>/processed_sam2).
  VGGT_FEAT_ROOT          Feature-tree root (default:
                          scenefun3d/processed_sam2/vggt_features).
EOF
}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

ENV_EXP_NAME="${EXP_NAME:-}"
ENV_PROCESSED_SAM2_DIR="${PROCESSED_SAM2_DIR:-scenefun3d/processed_sam2}"
POSITIONAL_EXP_NAME=""
POSITIONAL_PROCESSED_SAM2_DIR=""

# Accept the same <experiment-name> <processed-sam2-dir> prefix as
# scripts/train.sh. Values containing '=' are Hydra overrides and remain in
# "$@" for forwarding below.
if [[ $# -ge 2 && "$1" != *=* && "$2" != *=* ]]; then
    POSITIONAL_EXP_NAME="$1"
    POSITIONAL_PROCESSED_SAM2_DIR="$2"
    shift 2
elif [[ $# -ge 1 && "$1" != *=* ]]; then
    if [[ -n "$ENV_EXP_NAME" && -z "$ENV_PROCESSED_SAM2_DIR" ]]; then
        POSITIONAL_PROCESSED_SAM2_DIR="$1"
    elif [[ -z "$ENV_EXP_NAME" && -n "$ENV_PROCESSED_SAM2_DIR" ]]; then
        POSITIONAL_EXP_NAME="$1"
    else
        echo "ERROR: positional experiment name and processed directory must be supplied together." >&2
        usage >&2
        exit 2
    fi
    shift
fi
if [[ $# -gt 0 && "$1" != *=* ]]; then
    echo "ERROR: unexpected positional argument: $1" >&2
    usage >&2
    exit 2
fi

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
SCRIPT="pipeline/step8_3d_training_vggt/train_vggt.py"
ENV_NAME="${ENV_NAME:-pt}"
EXP_NAME="${POSITIONAL_EXP_NAME:-${ENV_EXP_NAME:-vggt_refine}}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
EXP_DIR="${EXP_DIR:-${OUTPUT_DIR}/${RUN_ID}_${EXP_NAME}}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
GLOBAL_VAL_BATCH_SIZE="${GLOBAL_VAL_BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-4}"
VAL_NUM_WORKERS="${VAL_NUM_WORKERS:-2}"
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
PROCESSED_SAM2_DIR="${POSITIONAL_PROCESSED_SAM2_DIR:-${ENV_PROCESSED_SAM2_DIR:-${DATA_ROOT}/processed_sam2}}"
VGGT_FEAT_ROOT="${VGGT_FEAT_ROOT:-${PROCESSED_SAM2_DIR}/vggt_features}"
GPUS="${GPUS-0,1,2,3}"

cd "$REPO_ROOT"

[[ -f "$SCRIPT" ]] || {
    echo "ERROR: training entry point not found: $SCRIPT" >&2
    exit 1
}
for path in \
    "$PROCESSED_SAM2_DIR/train" \
    "$PROCESSED_SAM2_DIR/val" \
    "$VGGT_FEAT_ROOT/train" \
    "$VGGT_FEAT_ROOT/val"; do
    [[ -d "$path" ]] || {
        echo "ERROR: required dataset directory not found: $path" >&2
        exit 1
    }
done

# Resolve GPU IDs in priority order: GPUS, inherited visibility, nvidia-smi.
if [[ -n "$GPUS" ]]; then
    GPU_SPEC="$GPUS"
elif [[ -n "${CUDA_VISIBLE_DEVICES:-}" && "${CUDA_VISIBLE_DEVICES:-}" != "-1" ]]; then
    GPU_SPEC="$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
    GPU_SPEC="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)"
else
    GPU_SPEC=""
fi
GPU_SPEC="${GPU_SPEC//,/ }"
GPU_SPEC="${GPU_SPEC//$'\n'/ }"
GPU_SPEC="${GPU_SPEC//$'\r'/ }"
read -r -a GPU_ARR <<< "$GPU_SPEC"
NPROC_PER_NODE="${#GPU_ARR[@]}"
if [[ "$NPROC_PER_NODE" -eq 0 ]]; then
    echo "ERROR: no GPUs selected; set GPUS or CUDA_VISIBLE_DEVICES." >&2
    exit 1
fi

for value_name in GLOBAL_BATCH_SIZE GLOBAL_VAL_BATCH_SIZE; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $value_name must be a positive integer (got '$value')." >&2
        exit 1
    fi
done
for value_name in NUM_WORKERS VAL_NUM_WORKERS; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[0-9]+$ ]]; then
        echo "ERROR: $value_name must be a non-negative integer (got '$value')." >&2
        exit 1
    fi
done
if (( GLOBAL_BATCH_SIZE % NPROC_PER_NODE != 0 )); then
    echo "ERROR: GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE is not divisible by $NPROC_PER_NODE processes." >&2
    exit 1
fi
if (( GLOBAL_VAL_BATCH_SIZE % NPROC_PER_NODE != 0 )); then
    echo "ERROR: GLOBAL_VAL_BATCH_SIZE=$GLOBAL_VAL_BATCH_SIZE is not divisible by $NPROC_PER_NODE processes." >&2
    exit 1
fi

PER_GPU_BATCH_SIZE=$((GLOBAL_BATCH_SIZE / NPROC_PER_NODE))
PER_GPU_VAL_BATCH_SIZE=$((GLOBAL_VAL_BATCH_SIZE / NPROC_PER_NODE))
VISIBLE_GPUS="$(IFS=,; echo "${GPU_ARR[*]}")"
export CUDA_VISIBLE_DEVICES="$VISIBLE_GPUS"
export PYTHONNOUSERSITE=1

if [[ "$NPROC_PER_NODE" -gt 1 ]]; then
    LAUNCH=(torchrun --standalone --nproc_per_node="$NPROC_PER_NODE")
else
    LAUNCH=(python)
fi

TRAIN_ARGS=(
    "$SCRIPT"
    "exp_name=$EXP_NAME"
    "exp_dir=$EXP_DIR"
    "task.dataset.root_dir=$DATA_ROOT"
    "task.dataset.processed_sam2_dir=$PROCESSED_SAM2_DIR"
    "task.dataset.vggt_feat_root=$VGGT_FEAT_ROOT"
    "task.train.batch_size=$PER_GPU_BATCH_SIZE"
    "task.train.val_batch_size=$PER_GPU_VAL_BATCH_SIZE"
    "task.train.num_workers=$NUM_WORKERS"
    "task.train.val_num_workers=$VAL_NUM_WORKERS"
    "$@"
)

echo "VGGT Step 8 distributed training"
echo "  GPUs:             $VISIBLE_GPUS ($NPROC_PER_NODE process(es))"
echo "  train batch:      $GLOBAL_BATCH_SIZE global / $PER_GPU_BATCH_SIZE per GPU"
echo "  validation batch: $GLOBAL_VAL_BATCH_SIZE global / $PER_GPU_VAL_BATCH_SIZE per GPU"
echo "  dataset:          $PROCESSED_SAM2_DIR"
echo "  features:         $VGGT_FEAT_ROOT"
echo "  output:           $EXP_DIR"

if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] CUDA_VISIBLE_DEVICES=%q ' "$CUDA_VISIBLE_DEVICES"
    printf '%q ' "${LAUNCH[@]}" "${TRAIN_ARGS[@]}"
    printf '\n'
    exit 0
fi

command -v conda >/dev/null 2>&1 || {
    echo "ERROR: conda is required to activate '$ENV_NAME'." >&2
    exit 1
}
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

command -v "${LAUNCH[0]}" >/dev/null 2>&1 || {
    echo "ERROR: ${LAUNCH[0]} is unavailable in conda environment '$ENV_NAME'." >&2
    exit 1
}

echo "  environment:      $CONDA_DEFAULT_ENV ($(command -v python))"
PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${LAUNCH[@]}" "${TRAIN_ARGS[@]}"
