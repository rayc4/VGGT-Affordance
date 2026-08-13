#!/usr/bin/env bash
# Train only the confidence-conditioned VGGT residual adapter on a fixed base.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  pipeline/step8_3d_training_vggt/run_vggt_adapter_from_base.sh \
    [--dry-run] [hydra overrides...]

Environment:
  BASE_CKPT             Mask-only initialization checkpoint (defaults to the
                        2026-07-29 base best checkpoint)
  GPUS                  GPU IDs passed to the distributed launcher
  EXP_NAME              Run name (default: uniform-vggt-conf-frozen)
  EXP_DIR               Exact output directory
  OUTPUT_DIR            Output root when EXP_DIR is unset (default: outputs)
  PROCESSED_SAM2_DIR    Processed train/val tree
  VGGT_FEAT_ROOT        Mirrored VGGT feature tree
  VGGT_FEATURE_NAME     Per-point feature file (default: vggt_feat_uniform.npy)
  VGGT_CONF_NAME        Per-point confidence file (default: vggt_conf.npy)
  VGGT_VIEW_COUNT_NAME  Per-point view-count file (default: vggt_view_count.npy)
  MODEL_CONFIG          Hydra model config (default: cdm_vggt_adapter_frozen)
  ADAPTER_MAX_STEPS     Fixed adapter budget (default: 5800)
  EXPECTED_TRAIN_FRAMES Strict post-filter train cardinality (default: 9257)
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
ADAPTER_MAX_STEPS="${ADAPTER_MAX_STEPS:-5800}"
EXPECTED_TRAIN_FRAMES="${EXPECTED_TRAIN_FRAMES:-9257}"
EXPECTED_VAL_FRAMES="${EXPECTED_VAL_FRAMES:-5619}"
VGGT_FEATURE_NAME="${VGGT_FEATURE_NAME:-vggt_feat_uniform.npy}"
VGGT_CONF_NAME="${VGGT_CONF_NAME:-vggt_conf.npy}"
VGGT_VIEW_COUNT_NAME="${VGGT_VIEW_COUNT_NAME:-vggt_view_count.npy}"
MODEL_CONFIG="${MODEL_CONFIG:-cdm_vggt_adapter_frozen}"

[[ -f "$BASE_CKPT" ]] || {
    echo "ERROR: base checkpoint not found: $BASE_CKPT" >&2
    exit 1
}
BASE_CKPT="$(realpath "$BASE_CKPT")"

for value_name in ADAPTER_MAX_STEPS EXPECTED_TRAIN_FRAMES EXPECTED_VAL_FRAMES; do
    value="${!value_name}"
    if ! [[ "$value" =~ ^[1-9][0-9]*$ ]]; then
        echo "ERROR: $value_name must be a positive integer (got '$value')." >&2
        exit 1
    fi
done

export EXP_NAME="${EXP_NAME:-uniform-vggt-conf-frozen}"
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
    "task.train.max_steps=$ADAPTER_MAX_STEPS" \
    task.train.early_stopping_patience=0 \
    "task.dataset.expected_train_frames=$EXPECTED_TRAIN_FRAMES" \
    "task.dataset.expected_val_frames=$EXPECTED_VAL_FRAMES" \
    "task.dataset.vggt_feat_name=$VGGT_FEATURE_NAME" \
    "task.dataset.vggt_conf_name=$VGGT_CONF_NAME" \
    "task.dataset.vggt_view_count_name=$VGGT_VIEW_COUNT_NAME" \
    "$@"
