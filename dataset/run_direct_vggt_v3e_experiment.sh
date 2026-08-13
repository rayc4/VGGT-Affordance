#!/usr/bin/env bash
# Validation-only v3e experiment:
# fresh role-aware proposals -> prediction-only VGGT/3D consensus -> exact
# legacy-format preprocessing. No original or earlier experiment artifacts are
# consumed, and every quality gate fails closed before the next expensive step.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: dataset/run_direct_vggt_v3e_experiment.sh [--dry-run]

Runs validation only. Important environment overrides:
  GPUS, N_WORKERS, PREPROCESS_ENV, DATA_ROOT, EXPERIMENT_ROOT,
  PROPOSAL_ROOT, CLIP_CACHE_ROOT, CONSENSUS_ROOT, PROCESSED_ROOT,
  DIRECT_EXTRA_ARGS, CONSENSUS_EXTRA_ARGS, VGGT_EXTRA_ARGS.
EOF
}

DRY_RUN="${DRY_RUN:-0}"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $arg" >&2; usage; exit 2 ;;
  esac
done
case "${DRY_RUN,,}" in
  1|true|yes) DRY_RUN=1 ;;
  0|false|no) DRY_RUN=0 ;;
  *) echo "ERROR: DRY_RUN must be boolean (got '$DRY_RUN')." >&2; exit 2 ;;
esac

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PREPROCESS_ENV="${PREPROCESS_ENV:-tasa}"
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-scenefun3d/preprocessing_experiments/direct_vggt_v3e}"
PROPOSAL_ROOT="${PROPOSAL_ROOT:-$EXPERIMENT_ROOT/proposals}"
CLIP_CACHE_ROOT="${CLIP_CACHE_ROOT:-$EXPERIMENT_ROOT/clip_cache}"
CONSENSUS_ROOT="${CONSENSUS_ROOT:-$EXPERIMENT_ROOT/consensus}"
PROCESSED_ROOT="${PROCESSED_ROOT:-scenefun3d/processed_sam2_vggt_direct_v3e}"
GPUS="${GPUS-}"
N_WORKERS="${N_WORKERS-}"
DIRECT_EXTRA_ARGS="${DIRECT_EXTRA_ARGS:---local-files-only --v3-mode --v3-role-aware-targeting --v3-target-surface-grounding --v3-using-instrument-targeting --v3-contact-region-grounding --v3-conflict-filter --grounding-coordinate-mode qwen-resized-pixels --qwen-max-pixels 2007040 --max-candidates 18 --max-per-video 18 --verified-anchor-target 6}"
CONSENSUS_EXTRA_ARGS="${CONSENSUS_EXTRA_ARGS:---local-files-only --max-proposals 6 --max-anchors 3 --cluster-radius 0.12 --min-video-support 2 --context-frames 8 --context-window-seconds 3.0}"
VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:---local-files-only --sam-local-files-only --component-selection anchor-consensus --crop-semantic-center --crop-semantic-radius 0.15 --crop-semantic-neighborhood-points 2048 --crop-semantic-neighborhood-radius 0.25 --max-published-views 6 --min-published-view-component-f1 0.10}"
DIRECT_MIN_COVERAGE="${DIRECT_MIN_COVERAGE:-0.78}"
ALIGNMENT_MIN_COVERAGE="${ALIGNMENT_MIN_COVERAGE:-0.70}"
ALIGNMENT_MAX_DISTANCE_M="${ALIGNMENT_MAX_DISTANCE_M:-0.20}"
ALIGNMENT_MAX_ANCHORS="${ALIGNMENT_MAX_ANCHORS:-3}"
FINAL_MIN_COVERAGE="${FINAL_MIN_COVERAGE:-0.7258}"

cd "$REPO_ROOT"

run_command() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

runner_args=()
if [ "$DRY_RUN" -eq 1 ]; then runner_args+=(--dry-run); fi

echo "v3e validation: proposals=$PROPOSAL_ROOT/val"
echo "v3e validation: consensus=$CONSENSUS_ROOT/val"
echo "v3e validation: legacy=$PROCESSED_ROOT/val"
echo "gates: direct=$DIRECT_MIN_COVERAGE alignment=$ALIGNMENT_MIN_COVERAGE within ${ALIGNMENT_MAX_DISTANCE_M}m final=$FINAL_MIN_COVERAGE"

run_command env \
  REPO_ROOT="$REPO_ROOT" \
  ENV_NAME="$PREPROCESS_ENV" \
  SPLIT=val \
  GPUS="$GPUS" \
  N_WORKERS="$N_WORKERS" \
  DATA_ROOT="$DATA_ROOT" \
  OUTPUT_ROOT="$PROPOSAL_ROOT" \
  CACHE_ROOT="$CLIP_CACHE_ROOT" \
  STAGE=all \
  EXTRA_ARGS="$DIRECT_EXTRA_ARGS" \
  LOG_DIR="$EXPERIMENT_ROOT/logs/proposals/val" \
  bash dataset/run_direct_anchors_parallel.sh "${runner_args[@]}"

run_command env \
  REPO_ROOT="$REPO_ROOT" \
  ENV_NAME="$PREPROCESS_ENV" \
  SPLIT=val \
  GPUS="$GPUS" \
  N_WORKERS="$N_WORKERS" \
  DATA_ROOT="$DATA_ROOT" \
  PROPOSAL_ROOT="$PROPOSAL_ROOT/val" \
  OUTPUT_ROOT="$CONSENSUS_ROOT/val" \
  EXTRA_ARGS="$CONSENSUS_EXTRA_ARGS" \
  LOG_DIR="$EXPERIMENT_ROOT/logs/consensus/val" \
  bash dataset/run_v3e_consensus_parallel.sh "${runner_args[@]}"

run_command conda run --no-capture-output -n "$PREPROCESS_ENV" \
  python -m pipeline.step1_direct_anchors.audit \
  --data-root "$DATA_ROOT" \
  --root "$CONSENSUS_ROOT" \
  --split val \
  --min-description-coverage "$DIRECT_MIN_COVERAGE"

run_command conda run --no-capture-output -n "$PREPROCESS_ENV" \
  python -m pipeline.step4_vggt_legacy.semantic_alignment_audit \
  --data-root "$DATA_ROOT" \
  --anchor-root "$CONSENSUS_ROOT/val" \
  --split val \
  --max-distance-m "$ALIGNMENT_MAX_DISTANCE_M" \
  --max-anchors "$ALIGNMENT_MAX_ANCHORS" \
  --min-aligned-coverage "$ALIGNMENT_MIN_COVERAGE"

run_command env \
  REPO_ROOT="$REPO_ROOT" \
  ENV_NAME="$PREPROCESS_ENV" \
  SPLIT=val \
  GPUS="$GPUS" \
  N_WORKERS="$N_WORKERS" \
  DATA_ROOT="$DATA_ROOT" \
  ANCHOR_ROOT="$CONSENSUS_ROOT/val" \
  NO_POINT_ROOT=1 \
  SAVE_DIR="$PROCESSED_ROOT" \
  EXTRA_ARGS="$VGGT_EXTRA_ARGS" \
  LOG_DIR="$EXPERIMENT_ROOT/logs/vggt_legacy/val" \
  bash dataset/run_preprocess_vggt_legacy_parallel.sh "${runner_args[@]}"

run_command conda run --no-capture-output -n "$PREPROCESS_ENV" \
  python -m pipeline.step4_vggt_legacy.audit \
  --data-root "$DATA_ROOT" \
  --root "$PROCESSED_ROOT" \
  --split val \
  --min-description-coverage "$FINAL_MIN_COVERAGE" \
  --require-hardlink-dedup
