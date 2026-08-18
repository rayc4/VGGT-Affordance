#!/usr/bin/env bash
# Fully frozen-base additive-stem ablation on the Step 2b-filtered dataset.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    cat <<'EOF'
Usage:
  VGGT_LR=1e-5 \
    pipeline/step8_3d_training_vggt/run_vggt_dpt_source_additive_stem_frozen_base_2bfilter.sh \
    [--dry-run] [hydra overrides...]

Runs the source-DPT additive-stem model on the Step 2b-filtered data while
keeping the initialized PointTransformer and contact head frozen for all
14,900 optimization steps. Outputs go to a fresh timestamped directory.

Environment overrides:
  VGGT_LR          Adapter learning rate (default: 3e-5)
  GPUS             GPU IDs (default: 4,5,6,7)
  EXP_NAME         Experiment name; controls the timestamped output suffix
  EXP_DIR          Optional exact output directory (must be empty or absent)
  CHECKPOINT_METRIC Validation metric used for mask_refinement_model_best.pt
                    (default: mIoU)
EOF
    exit 0
fi

export BASE_CKPT="${BASE_CKPT:-$REPO_ROOT/outputs/2026-08-17_16-20-04_base-map-v1-2bfilter/ckpt/mask_refinement_model011920.pt}"
export PROCESSED_SAM2_DIR="${PROCESSED_SAM2_DIR:-$REPO_ROOT/scenefun3d/processed_sam2_2bfilter}"
export VGGT_FEAT_ROOT="${VGGT_FEAT_ROOT:-$PROCESSED_SAM2_DIR/vggt_features}"

export GPUS="${GPUS:-0,1,2,3}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export GLOBAL_VAL_BATCH_SIZE="${GLOBAL_VAL_BATCH_SIZE:-16}"
export EXPECTED_TRAIN_FRAMES="${EXPECTED_TRAIN_FRAMES:-47665}"
export EXPECTED_VAL_FRAMES="${EXPECTED_VAL_FRAMES:-25795}"

export MAX_STEPS="${MAX_STEPS:-14900}"
export BASE_LR="${BASE_LR:-3e-6}"
export VGGT_LR="${VGGT_LR:-3e-5}"
export FULLY_FREEZE_BASE=1
export CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-mIoU}"
export EXP_NAME="${EXP_NAME:-source-dpt-additive-stem-frozen-base-2bfilter-vggtlr-${VGGT_LR}}"

exec "$SCRIPT_DIR/run_vggt_dpt_source_additive_stem_joint_from_base.sh" "$@"
