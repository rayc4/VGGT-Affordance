#!/usr/bin/env bash
# Pure v3c: fresh 2MP role-aware direct anchors with one bounded verifier
# relocation.  No earlier anchors, processed samples, logs, or cache files are
# consumed; the exact six-file legacy output remains the downstream contract.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-scenefun3d/preprocessing_experiments/direct_vggt_v3c}"
export DIRECT_ANCHOR_ROOT="${DIRECT_ANCHOR_ROOT:-$EXPERIMENT_ROOT/anchors}"
export CLIP_CACHE_ROOT="${CLIP_CACHE_ROOT:-$EXPERIMENT_ROOT/clip_cache}"
# Build the small, model-independent CLIP cache under the fresh v3c root.
export DIRECT_STAGE="${DIRECT_STAGE:-all}"
export PROCESSED_ROOT="${PROCESSED_ROOT:-scenefun3d/processed_sam2_vggt_direct_v3c}"
export DIRECT_EXTRA_ARGS="${DIRECT_EXTRA_ARGS:---local-files-only --v3-mode --v3-description-only-grounding --v3-role-aware-targeting --v3-verifier-relocation --v3-conflict-filter --grounding-coordinate-mode qwen-resized-pixels --qwen-max-pixels 2007040 --verification-max-new-tokens 128 --max-candidates 12 --max-per-video 12 --verified-anchor-target 2}"
export VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:---local-files-only --sam-local-files-only --component-selection anchor-consensus --crop-semantic-center --crop-semantic-radius 0.15 --crop-semantic-neighborhood-points 2048 --crop-semantic-neighborhood-radius 0.25 --max-published-views 6 --min-published-view-component-f1 0.10}"
export DIRECT_VAL_MIN_COVERAGE="${DIRECT_VAL_MIN_COVERAGE:-0.78}"
export DIRECT_TRAIN_MIN_COVERAGE="${DIRECT_TRAIN_MIN_COVERAGE:-0.70}"
export DIRECT_VAL_MIN_ALIGNMENT="${DIRECT_VAL_MIN_ALIGNMENT:-0.70}"
export DIRECT_ALIGNMENT_MAX_DISTANCE_M="${DIRECT_ALIGNMENT_MAX_DISTANCE_M:-0.20}"
export DIRECT_ALIGNMENT_MAX_ANCHORS="${DIRECT_ALIGNMENT_MAX_ANCHORS:-3}"
export EXP_NAME="${EXP_NAME:-direct_vggt_legacy_v3c}"

exec "$SCRIPT_DIR/run_direct_vggt_experiment.sh" "$@"
