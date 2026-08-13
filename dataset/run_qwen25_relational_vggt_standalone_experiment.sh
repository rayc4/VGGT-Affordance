#!/usr/bin/env bash
# Strict Qwen2.5 relational anchors -> VGGT propagation -> standalone six-file
# dataset. Every benchmark description is eligible; no original processed leaf,
# repair plan, or baseline-first union participates in selection or publication.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: dataset/run_qwen25_relational_vggt_standalone_experiment.sh [--dry-run] [--with-train]

Validation always runs first. It processes every benchmark description into a
fresh relational-only processed root. Strict grounding may abstain, so coverage
is measured by the aggregate audit rather than enforced independently per GPU.

Options:
  --dry-run     Print every preprocessing and audit command.
  --with-train  Process train only after the standalone validation audit passes.
  -h, --help    Show this help.

Safe runtime overrides:
  GPUS, N_WORKERS, PREPROCESS_ENV, DIRECT_TUNING_ARGS, VGGT_EXTRA_ARGS,
  VAL_MIN_COVERAGE, TRAIN_MIN_COVERAGE.

After --with-train succeeds, train with the unchanged loader:
  GPUS=0,1,2,3,4,5,6,7 GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 \
    bash scripts/train.sh qwen25_relational_vggt_standalone_v2 \
    scenefun3d/processed_sam2_qwen25_relational_vggt_standalone_v2
EOF
}

DRY_RUN="${DRY_RUN:-0}"
WITH_TRAIN="${WITH_TRAIN:-0}"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    --with-train) WITH_TRAIN=1 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown option: $arg" >&2; usage; exit 2 ;;
  esac
done

normalize_boolean() {
  local name="$1" value="${!1}"
  case "${value,,}" in
    1|true|yes) printf -v "$name" '%s' 1 ;;
    0|false|no) printf -v "$name" '%s' 0 ;;
    *) echo "ERROR: $name must be boolean (got '$value')." >&2; exit 2 ;;
  esac
}
normalize_boolean DRY_RUN
normalize_boolean WITH_TRAIN

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PREPROCESS_ENV="${PREPROCESS_ENV:-tasa}"
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-scenefun3d/preprocessing_experiments/qwen25_relational_vggt_standalone_v2}"
ANCHOR_ROOT="${ANCHOR_ROOT:-$EXPERIMENT_ROOT/anchors}"
CACHE_ROOT="${CACHE_ROOT:-$EXPERIMENT_ROOT/clip_cache}"
LOG_ROOT="${LOG_ROOT:-$EXPERIMENT_ROOT/logs}"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-$EXPERIMENT_ROOT/diagnostics}"
PROCESSED_ROOT="${PROCESSED_ROOT:-scenefun3d/processed_sam2_qwen25_relational_vggt_standalone_v2}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
N_WORKERS="${N_WORKERS:-8}"

# No replacement-quality coverage is assumed before the full validation pilot.
# The legacy audit independently requires at least one leaf, validates every
# leaf, and reports measured description coverage against all 445 descriptions.
VAL_MIN_COVERAGE="${VAL_MIN_COVERAGE:-0.0}"
TRAIN_MIN_COVERAGE="${TRAIN_MIN_COVERAGE:-0.0}"
DIRECT_TUNING_ARGS="${DIRECT_TUNING_ARGS:-}"
VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:---local-files-only --sam-local-files-only}"

# V2 couples each external relation with its reference, permits empty relation
# lists, repairs one malformed signature, verifies full-frame identity plus a
# point zoom, and filters semantic seeds geometrically before VGGT inference.
REQUIRED_DIRECT_ARGS="--local-files-only --v3-mode --v3-role-aware-targeting --v5-coupled-relational-grounding --v3-conflict-filter --grounding-coordinate-mode qwen-resized-pixels --qwen-model Qwen/Qwen2.5-VL-7B-Instruct --qwen-max-pixels 2007040 --frame-stride 3 --signature-max-new-tokens 192 --qwen-max-new-tokens 256 --verification-max-new-tokens 192 --retrieval-full-weight 0.70 --retrieval-parent-weight 0.20 --retrieval-component-weight 0.10 --max-candidates 18 --max-per-video 6 --verified-anchor-target 4"
REQUIRED_VGGT_ARGS="--max-anchors 3 --pretrack-semantic-lift-gate --pretrack-candidate-pool 12 --pretrack-max-pixel-distance 12 --pretrack-consensus-radius 0.20 --pretrack-min-independent-videos 2 --pretrack-single-anchor-min-confidence 0.85 --context-frames 8 --context-window-seconds 3.0 --component-selection anchor-consensus --independent-anchor-video-votes --crop-semantic-center --crop-semantic-radius 0.15 --crop-semantic-neighborhood-points 2048 --crop-semantic-neighborhood-radius 0.25 --max-published-views 6 --min-published-view-component-f1 0.10"

cd "$REPO_ROOT"
command -v realpath >/dev/null 2>&1 || {
  echo "ERROR: realpath is required for output isolation checks." >&2
  exit 1
}
command -v awk >/dev/null 2>&1 || {
  echo "ERROR: awk is required to validate coverage thresholds." >&2
  exit 1
}

resolve_path() { realpath -m "$1"; }
path_overlaps() {
  local left="$1" right="$2"
  [[ "$left" == "$right" || "$left" == "$right/"* || "$right" == "$left/"* ]]
}

CANONICAL_DATA_ROOT="$(resolve_path scenefun3d)"
RESOLVED_DATA_ROOT="$(resolve_path "$DATA_ROOT")"
RESOLVED_EXPERIMENT_ROOT="$(resolve_path "$EXPERIMENT_ROOT")"
RESOLVED_ANCHOR_ROOT="$(resolve_path "$ANCHOR_ROOT")"
RESOLVED_CACHE_ROOT="$(resolve_path "$CACHE_ROOT")"
RESOLVED_LOG_ROOT="$(resolve_path "$LOG_ROOT")"
RESOLVED_DIAGNOSTICS_ROOT="$(resolve_path "$DIAGNOSTICS_ROOT")"
RESOLVED_PROCESSED_ROOT="$(resolve_path "$PROCESSED_ROOT")"
RESOLVED_EXPERIMENT_BASE="$(resolve_path scenefun3d/preprocessing_experiments)"
RESOLVED_ORIGINAL_ROOT="$(resolve_path scenefun3d/processed_sam2)"

if [ "$RESOLVED_DATA_ROOT" != "$CANONICAL_DATA_ROOT" ]; then
  echo "ERROR: DATA_ROOT is pinned to $CANONICAL_DATA_ROOT" >&2
  exit 2
fi
if [ "$(dirname "$RESOLVED_EXPERIMENT_ROOT")" != "$RESOLVED_EXPERIMENT_BASE" ]; then
  echo "ERROR: EXPERIMENT_ROOT must be a direct child of $RESOLVED_EXPERIMENT_BASE" >&2
  exit 2
fi
case "$(basename "$RESOLVED_EXPERIMENT_ROOT")" in
  qwen25_relational_vggt_standalone_*) ;;
  *) echo "ERROR: EXPERIMENT_ROOT must use the qwen25_relational_vggt_standalone namespace." >&2; exit 2 ;;
esac
if [ "$(dirname "$RESOLVED_PROCESSED_ROOT")" != "$CANONICAL_DATA_ROOT" ]; then
  echo "ERROR: PROCESSED_ROOT must be a direct child of DATA_ROOT." >&2
  exit 2
fi
case "$(basename "$RESOLVED_PROCESSED_ROOT")" in
  processed_sam2_qwen25_relational_vggt_standalone_*) ;;
  *) echo "ERROR: PROCESSED_ROOT must use the processed_sam2_qwen25_relational_vggt_standalone namespace." >&2; exit 2 ;;
esac

for named_child in \
  "ANCHOR_ROOT:$RESOLVED_ANCHOR_ROOT" \
  "CACHE_ROOT:$RESOLVED_CACHE_ROOT" \
  "LOG_ROOT:$RESOLVED_LOG_ROOT" \
  "DIAGNOSTICS_ROOT:$RESOLVED_DIAGNOSTICS_ROOT"; do
  child_name="${named_child%%:*}"
  child_path="${named_child#*:}"
  case "$child_path" in
    "$RESOLVED_EXPERIMENT_ROOT"/*) ;;
    *) echo "ERROR: $child_name must be contained by EXPERIMENT_ROOT." >&2; exit 2 ;;
  esac
done
if path_overlaps "$RESOLVED_ANCHOR_ROOT" "$RESOLVED_CACHE_ROOT"; then
  echo "ERROR: ANCHOR_ROOT and CACHE_ROOT must be distinct and non-nested." >&2
  exit 2
fi
if path_overlaps "$RESOLVED_EXPERIMENT_ROOT" "$RESOLVED_PROCESSED_ROOT"; then
  echo "ERROR: experiment and processed roots must be distinct and non-nested." >&2
  exit 2
fi
if path_overlaps "$RESOLVED_EXPERIMENT_ROOT" "$RESOLVED_ORIGINAL_ROOT" || \
   path_overlaps "$RESOLVED_PROCESSED_ROOT" "$RESOLVED_ORIGINAL_ROOT"; then
  echo "ERROR: standalone outputs must not overlap the original processed root." >&2
  exit 2
fi

for forbidden in \
  scenefun3d/preprocessing_experiments/qwen25_relational_vggt_standalone_v1 \
  scenefun3d/processed_sam2_qwen25_relational_vggt_standalone_v1 \
  scenefun3d/preprocessing_experiments/qwen25_relational_vggt_repair_v1 \
  scenefun3d/processed_sam2_qwen25_relational_vggt_repair_candidates_v1 \
  scenefun3d/processed_sam2_qwen25_relational_vggt_repair_v1; do
  resolved_forbidden="$(resolve_path "$forbidden")"
  if path_overlaps "$RESOLVED_EXPERIMENT_ROOT" "$resolved_forbidden" || \
     path_overlaps "$RESOLVED_PROCESSED_ROOT" "$resolved_forbidden"; then
    echo "ERROR: standalone output overlaps protected prior artifact: $resolved_forbidden" >&2
    exit 2
  fi
done

if ! [[ "$N_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: N_WORKERS must be a positive integer." >&2
  exit 2
fi
for threshold_name in VAL_MIN_COVERAGE TRAIN_MIN_COVERAGE; do
  threshold_value="${!threshold_name}"
  if ! awk -v value="$threshold_value" \
    'BEGIN { ok=value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/; exit !(ok && value+0 >= 0 && value+0 <= 1) }'; then
    echo "ERROR: $threshold_name must be in [0, 1]." >&2
    exit 2
  fi
done

# All-description eligibility, strict Qwen semantics, roots, and sharding are
# runner-owned. Supplemental text may tune benign numeric thresholds only.
# shellcheck disable=SC2086
for tuning_arg in $DIRECT_TUNING_ARGS; do
  case "$tuning_arg" in
    --data-root|--data-root=*|--split|--split=*|--output-root|--output-root=*|\
    --cache-root|--cache-root=*|--description-plan|--description-plan=*|\
    --qwen-model|--qwen-model=*|--stage|--stage=*|--device|--device=*|\
    --num-shards|--num-shards=*|--shard|--shard=*|--v3-*|--v4-*|--v5-*|\
    --grounding-coordinate-mode|--grounding-coordinate-mode=*)
      echo "ERROR: DIRECT_TUNING_ARGS cannot override controlled option $tuning_arg" >&2
      exit 2
      ;;
  esac
done
# shellcheck disable=SC2086
for vggt_arg in $VGGT_EXTRA_ARGS; do
  case "$vggt_arg" in
    --data-root|--data-root=*|--split|--split=*|--anchor-root|--anchor-root=*|\
    --output-root|--output-root=*|--repair-plan|--repair-plan=*|\
    --point-root|--point-root=*|--no-point-root|--num-shards|--num-shards=*|\
    --shard|--shard=*|--allow-empty-output-shard|--allow-empty-output-shard=*|\
    --pretrack-*|--max-anchors|--max-anchors=*)
      echo "ERROR: VGGT_EXTRA_ARGS cannot override controlled option $vggt_arg" >&2
      exit 2
      ;;
  esac
done

runner_args=()
if [ "$DRY_RUN" -eq 1 ]; then runner_args+=(--dry-run); fi

run_json_report() {
  local report="$1" stderr_report
  shift
  stderr_report="${report%.json}.stderr.log"
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "$@"
    printf '2> %q | tee %q\n' "$stderr_report" "$report"
  else
    mkdir -p "$(dirname "$report")"
    "$@" 2> "$stderr_report" | tee "$report"
  fi
}

run_split() {
  local split="$1" min_coverage="$2"
  local direct_args="$DIRECT_TUNING_ARGS $REQUIRED_DIRECT_ARGS"
  local vggt_args="$VGGT_EXTRA_ARGS $REQUIRED_VGGT_ARGS"
  local direct_report="$DIAGNOSTICS_ROOT/$split/01_relational_anchor_contract.json"
  local alignment_report="$DIAGNOSTICS_ROOT/$split/02_anchor_semantic_alignment.json"
  local legacy_report="$DIAGNOSTICS_ROOT/$split/03_standalone_legacy_contract.json"

  echo "qwen25-relational standalone $split: scope=all benchmark descriptions"
  echo "qwen25-relational standalone $split: anchors=$ANCHOR_ROOT/$split"
  echo "qwen25-relational standalone $split: output=$PROCESSED_ROOT/$split"

  env \
    REPO_ROOT="$REPO_ROOT" \
    ENV_NAME="$PREPROCESS_ENV" \
    SPLIT="$split" \
    GPUS="$GPUS" \
    N_WORKERS="$N_WORKERS" \
    DATA_ROOT="$DATA_ROOT" \
    OUTPUT_ROOT="$ANCHOR_ROOT" \
    CACHE_ROOT="$CACHE_ROOT" \
    STAGE=all \
    EXTRA_ARGS="$direct_args" \
    LOG_DIR="$LOG_ROOT/direct_anchors/$split" \
    bash dataset/run_direct_anchors_parallel.sh "${runner_args[@]}"

  run_json_report "$direct_report" \
    conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step1_direct_anchors.audit \
    --data-root "$DATA_ROOT" \
    --root "$ANCHOR_ROOT" \
    --split "$split" \
    --min-status-coverage 1.0 \
    --min-description-coverage 0

  # Evaluation-only: this report is not an input to propagation, cropping,
  # publication, coverage gating, or later train preprocessing.
  run_json_report "$alignment_report" \
    conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step4_vggt_legacy.semantic_alignment_audit \
    --data-root "$DATA_ROOT" \
    --anchor-root "$ANCHOR_ROOT/$split" \
    --split "$split" \
    --max-distance-m 0.20 \
    --max-anchors 12 \
    --include-description-results

  env \
    REPO_ROOT="$REPO_ROOT" \
    ENV_NAME="$PREPROCESS_ENV" \
    SPLIT="$split" \
    GPUS="$GPUS" \
    N_WORKERS="$N_WORKERS" \
    DATA_ROOT="$DATA_ROOT" \
    ANCHOR_ROOT="$ANCHOR_ROOT/$split" \
    NO_POINT_ROOT=1 \
    SAVE_DIR="$PROCESSED_ROOT" \
    ALLOW_COPY_FALLBACK=0 \
    ALLOW_EMPTY_OUTPUT_SHARDS=1 \
    EXTRA_ARGS="$vggt_args" \
    LOG_DIR="$LOG_ROOT/vggt/$split" \
    bash dataset/run_preprocess_vggt_legacy_parallel.sh "${runner_args[@]}"

  run_json_report "$legacy_report" \
    conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step4_vggt_legacy.audit \
    --data-root "$DATA_ROOT" \
    --root "$PROCESSED_ROOT" \
    --split "$split" \
    --min-description-coverage "$min_coverage" \
    --require-hardlink-dedup
}

echo "coupled relational Qwen2.5 + VGGT standalone v2: workers=$N_WORKERS gpus=[$GPUS]"
echo "selection: all benchmark descriptions; no original leaves, repair plan, or union"
echo "validation audit: nonempty output, valid six-file leaves, loader smoke, measured coverage >= $VAL_MIN_COVERAGE"
run_split val "$VAL_MIN_COVERAGE"

if [ "$WITH_TRAIN" -eq 1 ]; then
  echo "Standalone validation passed; starting all-description train preprocessing."
  run_split train "$TRAIN_MIN_COVERAGE"
  echo "Standalone train root passed its six-file, empty-mask, loader, and hardlink audit."
  echo "Training command: GPUS=$GPUS GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 bash scripts/train.sh qwen25_relational_vggt_standalone_v2 $PROCESSED_ROOT"
else
  echo "Standalone validation passed. Re-run with --with-train to process train."
fi
