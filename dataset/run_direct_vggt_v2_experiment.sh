#!/usr/bin/env bash
# Pixel-correct Qwen grounding with targeted rescue, preserving the exact
# original processed_sam2 leaf format for unchanged downstream training.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-scenefun3d/preprocessing_experiments/direct_vggt_v2}"
export DIRECT_ANCHOR_ROOT="${DIRECT_ANCHOR_ROOT:-$EXPERIMENT_ROOT/anchors}"
# CLIP cache contents depend only on frame sampling and CLIP configuration, so
# the compatible v1 cache is reused instead of recomputing image embeddings.
export CLIP_CACHE_ROOT="${CLIP_CACHE_ROOT:-scenefun3d/preprocessing_experiments/direct_vggt_v1/clip_cache}"
export PROCESSED_ROOT="${PROCESSED_ROOT:-scenefun3d/processed_sam2_vggt_direct_v2}"
export DIRECT_EXTRA_ARGS="${DIRECT_EXTRA_ARGS:---local-files-only --grounding-coordinate-mode qwen-resized-pixels --max-candidates 8 --max-per-video 8 --primary-candidates 4 --rescue-on-primary-abstention --rescue-success-target 1}"
export DIRECT_VAL_MIN_COVERAGE="${DIRECT_VAL_MIN_COVERAGE:-0.78}"
export DIRECT_TRAIN_MIN_COVERAGE="${DIRECT_TRAIN_MIN_COVERAGE:-0.65}"
export EXP_NAME="${EXP_NAME:-direct_vggt_legacy_v2}"

exec "$SCRIPT_DIR/run_direct_vggt_experiment.sh" "$@"
