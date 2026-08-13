#!/usr/bin/env bash
# Preserve the repository's step-1--6 semantic proposals, then improve their
# temporal/3D consistency with VGGT tracks while publishing the unchanged
# processed_sam2 six-file contract.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: dataset/run_original_vggt_hybrid_experiment.sh [--dry-run] [--with-train]

Validation always runs first and must pass the original-coverage gate before
train preprocessing can start. Defaults use all eight GPUs and fresh roots.

Options:
  --dry-run     Print every preprocessing/audit command without executing it.
  --with-train  After validation passes, preprocess and audit the train split.
  -h, --help    Show this help.

Common overrides:
  GPUS, N_WORKERS, PREPROCESS_ENV, DATA_ROOT, EXPERIMENT_ROOT,
  PROCESSED_ROOT, LOG_ROOT, DIAGNOSTICS_ROOT, VGGT_EXTRA_ARGS (supplemental
  model/runtime options only),
  VAL_MIN_COVERAGE (may only raise the 323/445 floor), TRAIN_MIN_COVERAGE.

STEP6_ROOT and POINT_ROOT are deliberately pinned to the repository-original
step-6 and step-3 outputs. Use a different runner for source ablations.

After --with-train succeeds, the ordinary training entry point is unchanged:
  GPUS=0,1,2,3,4,5,6,7 GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 \
    bash scripts/train.sh original_vggt_hybrid_v1 \
    scenefun3d/processed_sam2_original_vggt_hybrid_v1
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
STEP6_ROOT="${STEP6_ROOT:-pipeline/step6_molmo_merge/molmo_merge_output}"
POINT_ROOT="${POINT_ROOT:-pipeline/step3_point_prediction/point_clipwithaffordance_output}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-scenefun3d/preprocessing_experiments/original_vggt_hybrid_v1}"
PROCESSED_ROOT="${PROCESSED_ROOT:-scenefun3d/processed_sam2_original_vggt_hybrid_v1}"
LOG_ROOT="${LOG_ROOT:-$EXPERIMENT_ROOT/logs}"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-$EXPERIMENT_ROOT/diagnostics}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
N_WORKERS="${N_WORKERS:-8}"
VAL_COVERAGE_HARD_MIN="0.7258426966292135" # exactly 323 / 445
VAL_MIN_COVERAGE="${VAL_MIN_COVERAGE:-$VAL_COVERAGE_HARD_MIN}"
TRAIN_MIN_COVERAGE="${TRAIN_MIN_COVERAGE:-0.0}"
VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:---local-files-only --sam-local-files-only}"
HYBRID_REQUIRED_ARGS="--component-selection anchor-consensus --independent-anchor-video-votes --crop-semantic-center --crop-semantic-radius 0.15 --crop-semantic-neighborhood-points 2048 --crop-semantic-neighborhood-radius 0.25 --max-published-views 6 --min-published-view-component-f1 0.10"
WORKER_EXTRA_ARGS="$VGGT_EXTRA_ARGS $HYBRID_REQUIRED_ARGS"

cd "$REPO_ROOT"
command -v realpath >/dev/null 2>&1 || {
  echo "ERROR: realpath is required for output isolation checks." >&2
  exit 1
}

resolve_path() { realpath -m "$1"; }
path_overlaps() {
  local left="$1" right="$2"
  [[ "$left" == "$right" || "$left" == "$right/"* || "$right" == "$left/"* ]]
}

RESOLVED_EXPERIMENT_ROOT="$(resolve_path "$EXPERIMENT_ROOT")"
RESOLVED_PROCESSED_ROOT="$(resolve_path "$PROCESSED_ROOT")"
RESOLVED_STEP6_ROOT="$(resolve_path "$STEP6_ROOT")"
RESOLVED_POINT_ROOT="$(resolve_path "$POINT_ROOT")"
CANONICAL_STEP6_ROOT="$(resolve_path pipeline/step6_molmo_merge/molmo_merge_output)"
CANONICAL_POINT_ROOT="$(resolve_path pipeline/step3_point_prediction/point_clipwithaffordance_output)"
RESOLVED_ORIGINAL_PROCESSED="$(resolve_path "$DATA_ROOT/processed_sam2")"
RESOLVED_DATA_ROOT="$(resolve_path "$DATA_ROOT")"
RESOLVED_EXPERIMENT_BASE="$(resolve_path "$DATA_ROOT/preprocessing_experiments")"

if [ "$RESOLVED_STEP6_ROOT" != "$CANONICAL_STEP6_ROOT" ]; then
  echo "ERROR: STEP6_ROOT is pinned to repository-original step-6 output: $CANONICAL_STEP6_ROOT" >&2
  exit 2
fi
if [ "$RESOLVED_POINT_ROOT" != "$CANONICAL_POINT_ROOT" ]; then
  echo "ERROR: POINT_ROOT is pinned to repository-original step-3 output: $CANONICAL_POINT_ROOT" >&2
  exit 2
fi

command -v awk >/dev/null 2>&1 || {
  echo "ERROR: awk is required to validate coverage thresholds." >&2
  exit 1
}
if ! awk -v requested="$VAL_MIN_COVERAGE" -v hard_min="$VAL_COVERAGE_HARD_MIN" \
  'BEGIN {
    valid = requested ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/
    exit !(valid && requested + 0 >= hard_min + 0 && requested + 0 <= 1)
  }'; then
  echo "ERROR: VAL_MIN_COVERAGE must be in [$VAL_COVERAGE_HARD_MIN, 1] (hard floor is exactly 323/445)." >&2
  exit 2
fi

# Output roots are constrained to sibling namespaces, not merely checked
# against the three immediate sources. This prevents a cleverly nested custom
# path from landing inside any older direct/VGGT experiment tree.
if [ "$(dirname "$RESOLVED_EXPERIMENT_ROOT")" != "$RESOLVED_EXPERIMENT_BASE" ]; then
  echo "ERROR: EXPERIMENT_ROOT must be a direct child of $RESOLVED_EXPERIMENT_BASE." >&2
  exit 2
fi
case "$(basename "$RESOLVED_EXPERIMENT_ROOT")" in
  original_vggt_hybrid*) ;;
  *)
    echo "ERROR: EXPERIMENT_ROOT must use the original_vggt_hybrid namespace." >&2
    exit 2
    ;;
esac
if [ "$(dirname "$RESOLVED_PROCESSED_ROOT")" != "$RESOLVED_DATA_ROOT" ]; then
  echo "ERROR: PROCESSED_ROOT must be a direct child of DATA_ROOT." >&2
  exit 2
fi
case "$(basename "$RESOLVED_PROCESSED_ROOT")" in
  processed_sam2_original_vggt_hybrid*) ;;
  *)
    echo "ERROR: PROCESSED_ROOT must use the processed_sam2_original_vggt_hybrid namespace." >&2
    exit 2
    ;;
esac

for named_output in \
  "EXPERIMENT_ROOT:$RESOLVED_EXPERIMENT_ROOT" \
  "PROCESSED_ROOT:$RESOLVED_PROCESSED_ROOT"; do
  output_name="${named_output%%:*}"
  output_path="${named_output#*:}"
  case "$output_path" in
    *original_vggt_hybrid*) ;;
    *)
      echo "ERROR: $output_name must use a clearly named original_vggt_hybrid namespace: $output_path" >&2
      exit 2
      ;;
  esac
  for source_path in \
    "$RESOLVED_STEP6_ROOT" \
    "$RESOLVED_POINT_ROOT" \
    "$RESOLVED_ORIGINAL_PROCESSED"; do
    if path_overlaps "$output_path" "$source_path"; then
      echo "ERROR: $output_name overlaps protected source data: $source_path" >&2
      exit 2
    fi
  done
done
if path_overlaps "$RESOLVED_EXPERIMENT_ROOT" "$RESOLVED_PROCESSED_ROOT"; then
  echo "ERROR: EXPERIMENT_ROOT and PROCESSED_ROOT must be separate, non-nested trees." >&2
  exit 2
fi

for named_child in "LOG_ROOT:$LOG_ROOT" "DIAGNOSTICS_ROOT:$DIAGNOSTICS_ROOT"; do
  child_name="${named_child%%:*}"
  child_path="$(resolve_path "${named_child#*:}")"
  case "$child_path" in
    "$RESOLVED_EXPERIMENT_ROOT"/*) ;;
    *) echo "ERROR: $child_name must be contained by EXPERIMENT_ROOT." >&2; exit 2 ;;
  esac
done

runner_args=()
if [ "$DRY_RUN" -eq 1 ]; then runner_args+=(--dry-run); fi

run_json_audit() {
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
  local report="$DIAGNOSTICS_ROOT/$split/legacy_contract.json"

  echo "hybrid $split: original anchors=$STEP6_ROOT/$split"
  echo "hybrid $split: original points=$POINT_ROOT"
  echo "hybrid $split: output=$PROCESSED_ROOT/$split"
  echo "hybrid $split: diagnostics=$report"

  # The child runner owns model execution and implements a true dry run. Run
  # it even in this wrapper's dry-run mode so GPU-to-shard assignment and the
  # complete worker commands remain inspectable.
  env \
    REPO_ROOT="$REPO_ROOT" \
    ENV_NAME="$PREPROCESS_ENV" \
    SPLIT="$split" \
    GPUS="$GPUS" \
    N_WORKERS="$N_WORKERS" \
    DATA_ROOT="$DATA_ROOT" \
    ANCHOR_ROOT="$STEP6_ROOT/$split" \
    POINT_ROOT="$POINT_ROOT" \
    SAVE_DIR="$PROCESSED_ROOT" \
    NO_POINT_ROOT=0 \
    ALLOW_COPY_FALLBACK=0 \
    EXTRA_ARGS="$WORKER_EXTRA_ARGS" \
    LOG_DIR="$LOG_ROOT/vggt_legacy/$split" \
    bash dataset/run_preprocess_vggt_legacy_parallel.sh "${runner_args[@]}"

  run_json_audit "$report" \
    conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step4_vggt_legacy.audit \
    --data-root "$DATA_ROOT" \
    --root "$PROCESSED_ROOT" \
    --split "$split" \
    --min-description-coverage "$min_coverage" \
    --require-hardlink-dedup
}

echo "original+VGGT hybrid: workers=$N_WORKERS gpus=[$GPUS]"
echo "validation gate: coverage >= $VAL_MIN_COVERAGE (hard floor exactly 323/445), zero invalid/empty leaves, loader smoke, required hardlinks"
run_split val "$VAL_MIN_COVERAGE"

if [ "$WITH_TRAIN" -eq 1 ]; then
  echo "Validation passed; starting train preprocessing."
  run_split train "$TRAIN_MIN_COVERAGE"
  echo "Train preprocessing passed its six-file/empty-mask/hardlink audit."
  echo "Training command: GPUS=$GPUS GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 bash scripts/train.sh original_vggt_hybrid_v1 $PROCESSED_ROOT"
else
  echo "Validation passed. Re-run with --with-train to preprocess train using the same isolated root."
fi
