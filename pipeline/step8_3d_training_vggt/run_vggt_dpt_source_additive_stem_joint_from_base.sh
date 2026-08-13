#!/usr/bin/env bash
# Staged source-DPT additive-stem fine-tuning from the mask-only base.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

dry_run_arg=()
if [[ "${1:-}" == "--dry-run" ]]; then
    dry_run_arg=(--dry-run)
    shift
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    exec "$SCRIPT_DIR/run_vggt_early_fusion_from_base.sh" "$1"
fi

export BASE_CKPT="${BASE_CKPT:-$REPO_ROOT/outputs/2026-08-04_21-46-06_base-map-v1/ckpt/mask_refinement_model_best.pt}"
export MODEL_CONFIG="${MODEL_CONFIG:-cdm_vggt_additive_stem_dpt_joint}"
export EXP_NAME="${EXP_NAME:-source-dpt-additive-stem-joint}"
export VGGT_FEAT_NAME="${VGGT_FEAT_NAME:-vggt_dpt_source_feat.npy}"

## Match the frozen source-DPT comparison while giving the base path a 10x
## smaller learning rate after an adapter-only warmup.
export MAX_STEPS="${MAX_STEPS:-2900}"
export BASE_LR="${BASE_LR:-3e-6}"
export VGGT_LR_MULTIPLIER="${VGGT_LR_MULTIPLIER:-10}"
export BASE_FREEZE_STEPS="${BASE_FREEZE_STEPS:-725}"
export GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-32}"
export GLOBAL_VAL_BATCH_SIZE="${GLOBAL_VAL_BATCH_SIZE:-16}"
export CHECKPOINT_METRIC="${CHECKPOINT_METRIC:-mAP}"

exec "$SCRIPT_DIR/run_vggt_early_fusion_from_base.sh" \
    "${dry_run_arg[@]}" \
    task.dataset.vggt_conf_name=vggt_source_conf.npy \
    task.dataset.vggt_view_count_name=vggt_source_view_count.npy \
    "task.train.lr_anneal_steps=$MAX_STEPS" \
    "task.train.early_stopping_metric=$CHECKPOINT_METRIC" \
    "$@"
