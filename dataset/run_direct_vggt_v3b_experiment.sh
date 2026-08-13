#!/usr/bin/env bash
# Pure v3b: v3 structured retrieval and marked-point verification around the
# less restrictive, description-only v2 grounding prompt. The experiment,
# CLIP cache, anchors, logs, and exact-format output all use isolated roots.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-scenefun3d/preprocessing_experiments/direct_vggt_v3b}"
export DIRECT_ANCHOR_ROOT="${DIRECT_ANCHOR_ROOT:-$EXPERIMENT_ROOT/anchors}"
export CLIP_CACHE_ROOT="${CLIP_CACHE_ROOT:-$EXPERIMENT_ROOT/clip_cache}"
# Fail closed instead of building a cache implicitly. Populate the isolated
# split cache explicitly before running this wrapper.
export DIRECT_STAGE="${DIRECT_STAGE:-anchors}"
export PROCESSED_ROOT="${PROCESSED_ROOT:-scenefun3d/processed_sam2_vggt_direct_v3b}"
export DIRECT_EXTRA_ARGS="${DIRECT_EXTRA_ARGS:---local-files-only --v3-mode --v3-description-only-grounding --v3-conflict-filter --grounding-coordinate-mode qwen-resized-pixels --max-candidates 12 --max-per-video 12 --verified-anchor-target 2}"
export VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:---local-files-only --sam-local-files-only --component-selection anchor-consensus --crop-semantic-center --crop-semantic-radius 0.15 --crop-semantic-neighborhood-points 2048 --crop-semantic-neighborhood-radius 0.25 --max-published-views 6 --min-published-view-component-f1 0.10}"
export DIRECT_VAL_MIN_COVERAGE="${DIRECT_VAL_MIN_COVERAGE:-0.78}"
export DIRECT_TRAIN_MIN_COVERAGE="${DIRECT_TRAIN_MIN_COVERAGE:-0.70}"
export DIRECT_VAL_MIN_ALIGNMENT="${DIRECT_VAL_MIN_ALIGNMENT:-0.70}"
export DIRECT_ALIGNMENT_MAX_DISTANCE_M="${DIRECT_ALIGNMENT_MAX_DISTANCE_M:-0.20}"
export DIRECT_ALIGNMENT_MAX_ANCHORS="${DIRECT_ALIGNMENT_MAX_ANCHORS:-3}"
export EXP_NAME="${EXP_NAME:-direct_vggt_legacy_v3b}"

exec "$SCRIPT_DIR/run_direct_vggt_experiment.sh" "$@"
