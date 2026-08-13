#!/usr/bin/env bash
# Controlled early-fusion VGGT training initialized from a mask-only base.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  pipeline/step8_3d_training_vggt/run_vggt_early_fusion_from_base.sh \
    [--dry-run] [hydra overrides...]

Environment:
  BASE_CKPT             Mask-only checkpoint (defaults to July 29 best-mAP)
  GPUS                  GPUs passed to the DDP launcher
  EXP_NAME              Run name (default: weighted-vggt-early-fusion)
  EXP_DIR               Exact output directory
  OUTPUT_DIR            Output root when EXP_DIR is unset (default: outputs)
  PROCESSED_SAM2_DIR    Processed train/val tree
  VGGT_FEAT_ROOT        Mirrored VGGT feature tree
  VGGT_FEAT_NAME        Aggregated feature file (default: vggt_feat.npy)
  MODEL_CONFIG          Hydra model preset (default:
                        cdm_vggt_early_fusion_controlled)
  MAX_STEPS             Fixed optimization budget (default: 5800)
  BASE_LR               PointTransformer/contact-head LR (default: 1e-5)
  VGGT_LR_MULTIPLIER    New VGGT-path LR multiplier (default: 10)
  BASE_FREEZE_STEPS     VGGT-only warmup before joint training (default: 725)
  EXPECTED_TRAIN_FRAMES Strict train cardinality (default: 9257)
  EXPECTED_VAL_FRAMES   Strict validation cardinality (default: 5619)
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
if [[ $# -gt 0 && "$1" != *=* ]]; then
    echo "ERROR: unexpected positional argument: $1" >&2
    usage >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
LAUNCHER="$REPO_ROOT/pipeline/step8_3d_training_vggt/run_train_vggt_parallel.sh"
BASE_CKPT="${BASE_CKPT:-$REPO_ROOT/outputs/2026-07-29_20-29-16_base/ckpt/mask_refinement_model_best.pt}"
VGGT_FEAT_NAME="${VGGT_FEAT_NAME:-vggt_feat.npy}"
MODEL_CONFIG="${MODEL_CONFIG:-cdm_vggt_early_fusion_controlled}"
MAX_STEPS="${MAX_STEPS:-5800}"
BASE_LR="${BASE_LR:-1e-5}"
VGGT_LR_MULTIPLIER="${VGGT_LR_MULTIPLIER:-10}"
BASE_FREEZE_STEPS="${BASE_FREEZE_STEPS:-725}"
EXPECTED_TRAIN_FRAMES="${EXPECTED_TRAIN_FRAMES:-9257}"
EXPECTED_VAL_FRAMES="${EXPECTED_VAL_FRAMES:-5619}"

[[ -f "$BASE_CKPT" ]] || {
    echo "ERROR: base checkpoint not found: $BASE_CKPT" >&2
    exit 1
}
BASE_CKPT="$(realpath "$BASE_CKPT")"
for value_name in MAX_STEPS EXPECTED_TRAIN_FRAMES EXPECTED_VAL_FRAMES; do
    value="${!value_name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: $value_name must be a positive integer (got '$value')." >&2
        exit 1
    }
done
[[ "$BASE_FREEZE_STEPS" =~ ^[0-9]+$ ]] || {
    echo "ERROR: BASE_FREEZE_STEPS must be a non-negative integer." >&2
    exit 1
}

export EXP_NAME="${EXP_NAME:-weighted-vggt-early-fusion}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs}"
RUN_ID="${RUN_ID:-$(date +%Y-%m-%d_%H-%M-%S)}"
export EXP_DIR="${EXP_DIR:-$OUTPUT_DIR/${RUN_ID}_${EXP_NAME}}"
if [[ "$DRY_RUN" -eq 0 && -d "$EXP_DIR" \
      && -n "$(find "$EXP_DIR" -mindepth 1 -print -quit)" ]]; then
    echo "ERROR: refusing to reuse non-empty EXP_DIR: $EXP_DIR" >&2
    exit 1
fi

dry_run_arg=()
if [[ "$DRY_RUN" -eq 1 ]]; then
    dry_run_arg=(--dry-run)
fi

"$LAUNCHER" "${dry_run_arg[@]}" \
    "model=$MODEL_CONFIG" \
    "task.train.init_ckpt=$BASE_CKPT" \
    task.train.validate_before_training=true \
    "task.train.max_steps=$MAX_STEPS" \
    task.train.early_stopping_patience=0 \
    "task.train.lr=$BASE_LR" \
    "+task.train.lr_multipliers.vggt=$VGGT_LR_MULTIPLIER" \
    "task.train.base_freeze_steps=$BASE_FREEZE_STEPS" \
    "task.dataset.expected_train_frames=$EXPECTED_TRAIN_FRAMES" \
    "task.dataset.expected_val_frames=$EXPECTED_VAL_FRAMES" \
    "task.dataset.vggt_feat_name=$VGGT_FEAT_NAME" \
    "$@"
