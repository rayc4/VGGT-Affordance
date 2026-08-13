#!/usr/bin/env bash
# End-to-end, exact-legacy-format direct-anchor experiment.
#
# Validation is generated and audited before any train preprocessing starts.
# The audit gates match the original usable-description coverage:
#   val:   323 / 445  = 0.725842... (gate 0.7258)
#   train: 1506 / 2598 = 0.579676... (gate 0.57967)
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: dataset/run_direct_vggt_experiment.sh [--dry-run] [--preprocess-only] [--validation-only]

The runner auto-selects every visible GPU and executes:
  val direct anchors -> val VGGT legacy output -> val coverage gate ->
  train direct anchors -> train VGGT legacy output -> train coverage gate ->
  unchanged downstream training.

Important environment overrides:
  GPUS, N_WORKERS, PREPROCESS_ENV, DATA_ROOT, EXPERIMENT_ROOT,
  DIRECT_ANCHOR_ROOT, CLIP_CACHE_ROOT, PROCESSED_ROOT,
  DIRECT_STAGE, DIRECT_EXTRA_ARGS, VGGT_EXTRA_ARGS, DIRECT_VAL_MIN_COVERAGE,
  DIRECT_TRAIN_MIN_COVERAGE, DIRECT_VAL_MIN_ALIGNMENT,
  DIRECT_ALIGNMENT_MAX_DISTANCE_M, DIRECT_ALIGNMENT_MAX_ANCHORS,
  GLOBAL_BATCH_SIZE, MAX_STEPS, EXP_NAME, TRAIN_OVERRIDES

DIRECT_VAL_MIN_COVERAGE and DIRECT_TRAIN_MIN_COVERAGE default to 0, which
disables the pre-VGGT direct-anchor audit and preserves the v1 workflow.
DIRECT_STAGE accepts all (default) or anchors. Use anchors only after the
split-specific CLIP cache has been populated under CLIP_CACHE_ROOT.
EOF
}

DRY_RUN="${DRY_RUN:-0}"
PREPROCESS_ONLY="${PREPROCESS_ONLY:-0}"
VALIDATION_ONLY="${VALIDATION_ONLY:-0}"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --preprocess-only) PREPROCESS_ONLY=1 ;;
    --validation-only) VALIDATION_ONLY=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $arg" >&2; usage; exit 2 ;;
  esac
done
for boolean_name in DRY_RUN PREPROCESS_ONLY VALIDATION_ONLY; do
  boolean_value="${!boolean_name}"
  case "${boolean_value,,}" in
    1|true|yes) printf -v "$boolean_name" '%s' 1 ;;
    0|false|no) printf -v "$boolean_name" '%s' 0 ;;
    *) echo "ERROR: $boolean_name must be boolean (got '$boolean_value')." >&2; exit 2 ;;
  esac
done

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PREPROCESS_ENV="${PREPROCESS_ENV:-tasa}"
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-scenefun3d/preprocessing_experiments/direct_vggt_v1}"
DIRECT_ANCHOR_ROOT="${DIRECT_ANCHOR_ROOT:-$EXPERIMENT_ROOT/anchors}"
CLIP_CACHE_ROOT="${CLIP_CACHE_ROOT:-$EXPERIMENT_ROOT/clip_cache}"
PROCESSED_ROOT="${PROCESSED_ROOT:-scenefun3d/processed_sam2_vggt_direct_v1}"
DIRECT_STAGE="${DIRECT_STAGE:-all}"
DIRECT_EXTRA_ARGS="${DIRECT_EXTRA_ARGS:---local-files-only}"
VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:---local-files-only --sam-local-files-only --component-selection anchor-consensus --crop-semantic-center --crop-semantic-radius 0.15 --max-published-views 6 --min-published-view-component-f1 0.10}"
DIRECT_VAL_MIN_COVERAGE="${DIRECT_VAL_MIN_COVERAGE:-0}"
DIRECT_TRAIN_MIN_COVERAGE="${DIRECT_TRAIN_MIN_COVERAGE:-0}"
DIRECT_VAL_MIN_ALIGNMENT="${DIRECT_VAL_MIN_ALIGNMENT:-0}"
DIRECT_ALIGNMENT_MAX_DISTANCE_M="${DIRECT_ALIGNMENT_MAX_DISTANCE_M:-0.20}"
DIRECT_ALIGNMENT_MAX_ANCHORS="${DIRECT_ALIGNMENT_MAX_ANCHORS:-3}"
EXP_NAME="${EXP_NAME:-direct_vggt_legacy_v1}"
GLOBAL_BATCH_SIZE="${GLOBAL_BATCH_SIZE:-64}"
MAX_STEPS="${MAX_STEPS:-50000}"
TRAIN_OVERRIDES="${TRAIN_OVERRIDES:-}"
GPUS="${GPUS-}"
N_WORKERS="${N_WORKERS-}"

case "$DIRECT_STAGE" in
  all|anchors) ;;
  *) echo "ERROR: DIRECT_STAGE must be all or anchors (got '$DIRECT_STAGE')." >&2; exit 2 ;;
esac

cd "$REPO_ROOT"

if [ -n "$GPUS" ]; then
  GPU_SPEC="$GPUS"
elif [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
  GPU_SPEC="$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  if ! GPU_SPEC="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)"; then
    GPU_SPEC=""
  fi
else
  GPU_SPEC=""
fi
if [ "$GPU_SPEC" = "-1" ]; then
  GPU_SPEC=""
fi
GPU_SPEC="${GPU_SPEC//,/ }"
GPU_SPEC="${GPU_SPEC//$'\n'/ }"
GPU_SPEC="${GPU_SPEC//$'\r'/ }"
read -r -a GPU_ARR <<< "$GPU_SPEC"
GPU_COUNT="${#GPU_ARR[@]}"

declare -A SEEN_GPUS=()
for gpu in "${GPU_ARR[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ || "$gpu" =~ ^GPU-[[:alnum:]-]+$ || "$gpu" =~ ^MIG-[[:alnum:]_./-]+$ ]]; then
    echo "ERROR: invalid GPU selection '$gpu'; use CUDA indices or GPU/MIG UUIDs." >&2
    exit 2
  fi
  if [ "${SEEN_GPUS[$gpu]+present}" = present ]; then
    echo "ERROR: duplicate GPU selection '$gpu'." >&2
    exit 2
  fi
  SEEN_GPUS["$gpu"]=1
done
if [ "$DRY_RUN" -eq 0 ] && [ "$GPU_COUNT" -eq 0 ]; then
  echo "ERROR: the quality experiment requires CUDA; set GPUS or CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi
if [ "$GPU_COUNT" -eq 0 ]; then
  GPU_ARR=(0)
  GPU_COUNT=1
fi
if [ -z "$N_WORKERS" ]; then
  N_WORKERS="$GPU_COUNT"
fi
if ! [[ "$N_WORKERS" =~ ^[1-9][0-9]*$ ]] || [ "$N_WORKERS" -gt "$GPU_COUNT" ]; then
  echo "ERROR: N_WORKERS must be in [1, $GPU_COUNT] (got '$N_WORKERS')." >&2
  exit 2
fi
GPU_CSV="$(IFS=,; echo "${GPU_ARR[*]}")"

if ! [[ "$GLOBAL_BATCH_SIZE" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: GLOBAL_BATCH_SIZE must be a positive integer." >&2
  exit 2
fi
if ! [[ "$MAX_STEPS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: MAX_STEPS must be a positive integer." >&2
  exit 2
fi
for coverage_name in DIRECT_VAL_MIN_COVERAGE DIRECT_TRAIN_MIN_COVERAGE DIRECT_VAL_MIN_ALIGNMENT; do
  coverage_value="${!coverage_name}"
  if ! [[ "$coverage_value" =~ ^(0([.][0-9]+)?|1([.]0+)?)$ ]]; then
    echo "ERROR: $coverage_name must lie in [0, 1] (got '$coverage_value')." >&2
    exit 2
  fi
done
if ! [[ "$DIRECT_ALIGNMENT_MAX_DISTANCE_M" =~ ^([0-9]+([.][0-9]*)?|[.][0-9]+)$ ]] || \
   ! awk -v value="$DIRECT_ALIGNMENT_MAX_DISTANCE_M" \
     'BEGIN { exit !(value > 0) }'; then
  echo "ERROR: DIRECT_ALIGNMENT_MAX_DISTANCE_M must be positive." >&2
  exit 2
fi
if ! [[ "$DIRECT_ALIGNMENT_MAX_ANCHORS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: DIRECT_ALIGNMENT_MAX_ANCHORS must be a positive integer." >&2
  exit 2
fi
if [ "$PREPROCESS_ONLY" -eq 0 ] && [ "$VALIDATION_ONLY" -eq 0 ] && \
   [ $((GLOBAL_BATCH_SIZE % GPU_COUNT)) -ne 0 ]; then
  echo "ERROR: GLOBAL_BATCH_SIZE=$GLOBAL_BATCH_SIZE is not divisible by $GPU_COUNT GPUs." >&2
  echo "Set GLOBAL_BATCH_SIZE to preserve a valid equal per-GPU batch." >&2
  exit 2
fi

echo "Direct VGGT experiment: GPUs=[$GPU_CSV], workers=$N_WORKERS"
echo "Direct anchors: $DIRECT_ANCHOR_ROOT/{val,train}"
echo "Direct-anchor stage: $DIRECT_STAGE"
echo "Legacy-format samples: $PROCESSED_ROOT/{val,train}"
echo "Validation must pass 0.7258 coverage before train preprocessing starts."
echo "Pre-VGGT direct-anchor gates: val=$DIRECT_VAL_MIN_COVERAGE train=$DIRECT_TRAIN_MIN_COVERAGE (0 disables)"
echo "Pre-VGGT semantic-alignment gate: val=$DIRECT_VAL_MIN_ALIGNMENT within ${DIRECT_ALIGNMENT_MAX_DISTANCE_M}m (0 disables)"

run_command() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

run_direct_anchors() {
  local split="$1"
  if [ "$DRY_RUN" -eq 0 ] && [ "$DIRECT_STAGE" = anchors ] && \
     [ ! -f "$CLIP_CACHE_ROOT/$split/.clip_cache_config.json" ]; then
    echo "ERROR: DIRECT_STAGE=anchors requires a populated $split cache at" >&2
    echo "  $CLIP_CACHE_ROOT/$split" >&2
    echo "Run the cache stage or populate the isolated cache before retrying." >&2
    return 1
  fi
  run_command env \
    REPO_ROOT="$REPO_ROOT" \
    ENV_NAME="$PREPROCESS_ENV" \
    SPLIT="$split" \
    GPUS="$GPU_CSV" \
    N_WORKERS="$N_WORKERS" \
    DATA_ROOT="$DATA_ROOT" \
    OUTPUT_ROOT="$DIRECT_ANCHOR_ROOT" \
    CACHE_ROOT="$CLIP_CACHE_ROOT" \
    STAGE="$DIRECT_STAGE" \
    EXTRA_ARGS="$DIRECT_EXTRA_ARGS" \
    LOG_DIR="$EXPERIMENT_ROOT/logs/direct_anchors/$split" \
    bash dataset/run_direct_anchors_parallel.sh
}

run_vggt_legacy() {
  local split="$1"
  run_command env \
    REPO_ROOT="$REPO_ROOT" \
    ENV_NAME="$PREPROCESS_ENV" \
    SPLIT="$split" \
    GPUS="$GPU_CSV" \
    N_WORKERS="$N_WORKERS" \
    DATA_ROOT="$DATA_ROOT" \
    ANCHOR_ROOT="$DIRECT_ANCHOR_ROOT/$split" \
    NO_POINT_ROOT=1 \
    SAVE_DIR="$PROCESSED_ROOT" \
    EXTRA_ARGS="$VGGT_EXTRA_ARGS" \
    LOG_DIR="$EXPERIMENT_ROOT/logs/vggt_legacy/$split" \
    bash dataset/run_preprocess_vggt_legacy_parallel.sh
}

run_direct_audit() {
  local split="$1"
  local coverage="$2"
  run_command conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step1_direct_anchors.audit \
    --data-root "$DATA_ROOT" \
    --root "$DIRECT_ANCHOR_ROOT" \
    --split "$split" \
    --min-description-coverage "$coverage"
}

run_alignment_audit() {
  local split="$1"
  local coverage="$2"
  run_command conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step4_vggt_legacy.semantic_alignment_audit \
    --data-root "$DATA_ROOT" \
    --anchor-root "$DIRECT_ANCHOR_ROOT/$split" \
    --split "$split" \
    --max-distance-m "$DIRECT_ALIGNMENT_MAX_DISTANCE_M" \
    --max-anchors "$DIRECT_ALIGNMENT_MAX_ANCHORS" \
    --min-aligned-coverage "$coverage"
}

run_audit() {
  local split="$1"
  local coverage="$2"
  run_command conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step4_vggt_legacy.audit \
    --data-root "$DATA_ROOT" \
    --root "$PROCESSED_ROOT" \
    --split "$split" \
    --min-description-coverage "$coverage" \
    --require-hardlink-dedup
}

direct_audit_enabled() {
  local coverage="$1"
  [[ ! "$coverage" =~ ^0([.]0+)?$ ]]
}

# The order is intentional: do not spend time on train or start downstream
# training unless replacement-quality validation coverage is already present.
run_direct_anchors val
if direct_audit_enabled "$DIRECT_VAL_MIN_COVERAGE"; then
  run_direct_audit val "$DIRECT_VAL_MIN_COVERAGE"
fi
if direct_audit_enabled "$DIRECT_VAL_MIN_ALIGNMENT"; then
  run_alignment_audit val "$DIRECT_VAL_MIN_ALIGNMENT"
fi
run_vggt_legacy val
run_audit val 0.7258

if [ "$VALIDATION_ONLY" -eq 1 ]; then
  exit 0
fi

run_direct_anchors train
if direct_audit_enabled "$DIRECT_TRAIN_MIN_COVERAGE"; then
  run_direct_audit train "$DIRECT_TRAIN_MIN_COVERAGE"
fi
run_vggt_legacy train
run_audit train 0.57967

if [ "$PREPROCESS_ONLY" -eq 0 ]; then
  # Match the repository convention that extra Hydra overrides are simple
  # whitespace-delimited tokens.
  # shellcheck disable=SC2206
  TRAIN_EXTRA_ARR=($TRAIN_OVERRIDES)
  run_command env \
    GPUS="$GPU_CSV" \
    GLOBAL_BATCH_SIZE="$GLOBAL_BATCH_SIZE" \
    MAX_STEPS="$MAX_STEPS" \
    bash scripts/train.sh "$EXP_NAME" "$PROCESSED_ROOT" "${TRAIN_EXTRA_ARR[@]}"
fi
