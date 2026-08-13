#!/usr/bin/env bash
# Repair invalid repository-original processed_sam2 samples with VGGT while
# retaining every original leaf that already satisfies the six-file contract.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: dataset/run_original_vggt_repair_experiment.sh [--dry-run] [--with-train]

Validation is always planned, repaired, audited, unioned with the valid
repository-original leaves, and audited again before train preprocessing may
start. The default repair run uses GPUs 0-7 and fresh experiment/output roots.

Options:
  --dry-run     Print all plan, eight-GPU repair, materialization, and audit commands.
  --with-train  Run the train flow only after the final validation union passes.
  -h, --help    Show this help.

Runtime overrides:
  GPUS, N_WORKERS, PREPROCESS_ENV, EXPERIMENT_ROOT, REPAIR_ROOT, FINAL_ROOT,
  LOG_ROOT, DIAGNOSTICS_ROOT, VGGT_EXTRA_ARGS, VAL_MIN_COVERAGE,
  TRAIN_MIN_COVERAGE.

DATA_ROOT, BASELINE_ROOT, STEP6_ROOT, and POINT_ROOT are pinned to the
repository-original inputs. NO_POINT_ROOT and ALLOW_COPY_FALLBACK are ignored
and forced to their safe values for the child runner.

After --with-train succeeds, downstream training remains unchanged:
  GPUS=0,1,2,3,4,5,6,7 GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 \
    bash scripts/train.sh original_vggt_repair_v1 \
    scenefun3d/processed_sam2_original_vggt_repair_v1
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
BASELINE_ROOT="${BASELINE_ROOT:-scenefun3d/processed_sam2}"
STEP6_ROOT="${STEP6_ROOT:-pipeline/step6_molmo_merge/molmo_merge_output}"
POINT_ROOT="${POINT_ROOT:-pipeline/step3_point_prediction/point_clipwithaffordance_output}"
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-scenefun3d/preprocessing_experiments/original_vggt_repair_v1}"
REPAIR_ROOT="${REPAIR_ROOT:-scenefun3d/processed_sam2_original_vggt_repair_candidates_v1}"
FINAL_ROOT="${FINAL_ROOT:-scenefun3d/processed_sam2_original_vggt_repair_v1}"
LOG_ROOT="${LOG_ROOT:-$EXPERIMENT_ROOT/logs}"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-$EXPERIMENT_ROOT/diagnostics}"
PLAN_ROOT="${PLAN_ROOT:-$EXPERIMENT_ROOT/plans}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
N_WORKERS="${N_WORKERS:-8}"
VAL_COVERAGE_HARD_MIN="0.7258426966292135" # exactly 323 / 445
VAL_MIN_COVERAGE="${VAL_MIN_COVERAGE:-$VAL_COVERAGE_HARD_MIN}"
TRAIN_MIN_COVERAGE="${TRAIN_MIN_COVERAGE:-0.0}"
VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:---local-files-only --sam-local-files-only}"
REQUIRED_VGGT_ARGS="--component-selection anchor-consensus --independent-anchor-video-votes --crop-semantic-center --crop-semantic-radius 0.15 --crop-semantic-neighborhood-points 2048 --crop-semantic-neighborhood-radius 0.25 --max-published-views 6 --min-published-view-component-f1 0.10"
WORKER_EXTRA_ARGS="$VGGT_EXTRA_ARGS $REQUIRED_VGGT_ARGS"

cd "$REPO_ROOT"
command -v realpath >/dev/null 2>&1 || {
  echo "ERROR: realpath is required for source pinning and output isolation." >&2
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
CANONICAL_BASELINE_ROOT="$(resolve_path scenefun3d/processed_sam2)"
CANONICAL_STEP6_ROOT="$(resolve_path pipeline/step6_molmo_merge/molmo_merge_output)"
CANONICAL_POINT_ROOT="$(resolve_path pipeline/step3_point_prediction/point_clipwithaffordance_output)"
RESOLVED_DATA_ROOT="$(resolve_path "$DATA_ROOT")"
RESOLVED_BASELINE_ROOT="$(resolve_path "$BASELINE_ROOT")"
RESOLVED_STEP6_ROOT="$(resolve_path "$STEP6_ROOT")"
RESOLVED_POINT_ROOT="$(resolve_path "$POINT_ROOT")"
RESOLVED_EXPERIMENT_ROOT="$(resolve_path "$EXPERIMENT_ROOT")"
RESOLVED_REPAIR_ROOT="$(resolve_path "$REPAIR_ROOT")"
RESOLVED_FINAL_ROOT="$(resolve_path "$FINAL_ROOT")"
RESOLVED_EXPERIMENT_BASE="$(resolve_path scenefun3d/preprocessing_experiments)"

for pinned in \
  "DATA_ROOT:$RESOLVED_DATA_ROOT:$CANONICAL_DATA_ROOT" \
  "BASELINE_ROOT:$RESOLVED_BASELINE_ROOT:$CANONICAL_BASELINE_ROOT" \
  "STEP6_ROOT:$RESOLVED_STEP6_ROOT:$CANONICAL_STEP6_ROOT" \
  "POINT_ROOT:$RESOLVED_POINT_ROOT:$CANONICAL_POINT_ROOT"; do
  pinned_name="${pinned%%:*}"
  pinned_remainder="${pinned#*:}"
  pinned_actual="${pinned_remainder%%:*}"
  pinned_expected="${pinned_remainder#*:}"
  if [ "$pinned_actual" != "$pinned_expected" ]; then
    echo "ERROR: $pinned_name is pinned to repository-original input: $pinned_expected" >&2
    exit 2
  fi
done

if ! awk -v requested="$VAL_MIN_COVERAGE" -v hard_min="$VAL_COVERAGE_HARD_MIN" \
  'BEGIN {
    valid = requested ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/
    exit !(valid && requested + 0 >= hard_min + 0 && requested + 0 <= 1)
  }'; then
  echo "ERROR: VAL_MIN_COVERAGE must be in [$VAL_COVERAGE_HARD_MIN, 1] (hard floor is exactly 323/445)." >&2
  exit 2
fi
if ! awk -v requested="$TRAIN_MIN_COVERAGE" \
  'BEGIN {
    valid = requested ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/
    exit !(valid && requested + 0 >= 0 && requested + 0 <= 1)
  }'; then
  echo "ERROR: TRAIN_MIN_COVERAGE must be in [0, 1]." >&2
  exit 2
fi

if [ "$(dirname "$RESOLVED_EXPERIMENT_ROOT")" != "$RESOLVED_EXPERIMENT_BASE" ]; then
  echo "ERROR: EXPERIMENT_ROOT must be a direct child of $RESOLVED_EXPERIMENT_BASE." >&2
  exit 2
fi
case "$(basename "$RESOLVED_EXPERIMENT_ROOT")" in
  original_vggt_repair_*) ;;
  *) echo "ERROR: EXPERIMENT_ROOT must use the original_vggt_repair namespace." >&2; exit 2 ;;
esac

if [ "$(dirname "$RESOLVED_REPAIR_ROOT")" != "$CANONICAL_DATA_ROOT" ]; then
  echo "ERROR: REPAIR_ROOT must be a direct child of the pinned DATA_ROOT." >&2
  exit 2
fi
case "$(basename "$RESOLVED_REPAIR_ROOT")" in
  processed_sam2_original_vggt_repair_candidates_*) ;;
  *) echo "ERROR: REPAIR_ROOT must use the repair-candidates namespace." >&2; exit 2 ;;
esac

if [ "$(dirname "$RESOLVED_FINAL_ROOT")" != "$CANONICAL_DATA_ROOT" ]; then
  echo "ERROR: FINAL_ROOT must be a direct child of the pinned DATA_ROOT." >&2
  exit 2
fi
case "$(basename "$RESOLVED_FINAL_ROOT")" in
  processed_sam2_original_vggt_repair_v*) ;;
  *) echo "ERROR: FINAL_ROOT must use the final original_vggt_repair namespace." >&2; exit 2 ;;
esac

resolved_outputs=(
  "$RESOLVED_EXPERIMENT_ROOT"
  "$RESOLVED_REPAIR_ROOT"
  "$RESOLVED_FINAL_ROOT"
)
for ((left_index = 0; left_index < ${#resolved_outputs[@]}; left_index++)); do
  output_path="${resolved_outputs[$left_index]}"
  for protected in \
    "$CANONICAL_BASELINE_ROOT" \
    "$CANONICAL_STEP6_ROOT" \
    "$CANONICAL_POINT_ROOT"; do
    if path_overlaps "$output_path" "$protected"; then
      echo "ERROR: output root overlaps protected source data: $output_path ; $protected" >&2
      exit 2
    fi
  done
  for ((right_index = left_index + 1; right_index < ${#resolved_outputs[@]}; right_index++)); do
    if path_overlaps "$output_path" "${resolved_outputs[$right_index]}"; then
      echo "ERROR: experiment, repair, and final roots must be distinct and non-nested." >&2
      exit 2
    fi
  done
done

for named_child in \
  "LOG_ROOT:$LOG_ROOT" \
  "DIAGNOSTICS_ROOT:$DIAGNOSTICS_ROOT" \
  "PLAN_ROOT:$PLAN_ROOT"; do
  child_name="${named_child%%:*}"
  child_path="$(resolve_path "${named_child#*:}")"
  case "$child_path" in
    "$RESOLVED_EXPERIMENT_ROOT"/*) ;;
    *) echo "ERROR: $child_name must be contained by EXPERIMENT_ROOT." >&2; exit 2 ;;
  esac
done

runner_args=()
if [ "$DRY_RUN" -eq 1 ]; then runner_args+=(--dry-run); fi

run_command() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

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
  local plan="$PLAN_ROOT/$split.json"
  local repair_report="$DIAGNOSTICS_ROOT/$split/repair_contract.json"
  local materialize_report="$DIAGNOSTICS_ROOT/$split/materialize.json"
  local final_report="$DIAGNOSTICS_ROOT/$split/final_contract.json"

  echo "repair-only $split: baseline=$BASELINE_ROOT/$split"
  echo "repair-only $split: plan=$plan (scope=any-invalid)"
  echo "repair-only $split: candidates=$REPAIR_ROOT/$split"
  echo "repair-only $split: final-union=$FINAL_ROOT/$split"

  if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$PLAN_ROOT" "$LOG_ROOT" "$DIAGNOSTICS_ROOT/$split"
  fi
  run_command \
    conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step5_vggt_repair.plan \
    --data-root "$DATA_ROOT" \
    --baseline-root "$BASELINE_ROOT" \
    --split "$split" \
    --scope any-invalid \
    --output "$plan"

  # Invoke the child in dry-run mode too: it expands the exact GPU/shard map
  # and worker CLI, including the pinned repair plan, without loading models.
  env \
    REPO_ROOT="$REPO_ROOT" \
    ENV_NAME="$PREPROCESS_ENV" \
    SPLIT="$split" \
    GPUS="$GPUS" \
    N_WORKERS="$N_WORKERS" \
    DATA_ROOT="$DATA_ROOT" \
    ANCHOR_ROOT="$STEP6_ROOT/$split" \
    POINT_ROOT="$POINT_ROOT" \
    REPAIR_PLAN="$plan" \
    SAVE_DIR="$REPAIR_ROOT" \
    NO_POINT_ROOT=0 \
    ALLOW_COPY_FALLBACK=0 \
    EXTRA_ARGS="$WORKER_EXTRA_ARGS" \
    LOG_DIR="$LOG_ROOT/vggt_repair/$split" \
    bash dataset/run_preprocess_vggt_legacy_parallel.sh "${runner_args[@]}"

  run_json_report "$repair_report" \
    conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step4_vggt_legacy.audit \
    --data-root "$DATA_ROOT" \
    --root "$REPAIR_ROOT" \
    --split "$split" \
    --min-description-coverage 0 \
    --require-hardlink-dedup

  run_json_report "$materialize_report" \
    conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step5_vggt_repair.materialize \
    --baseline-root "$BASELINE_ROOT" \
    --repair-root "$REPAIR_ROOT" \
    --output-root "$FINAL_ROOT" \
    --repair-plan "$plan" \
    --split "$split"

  # This audit is the promotion gate: baseline-valid identities must all be
  # hardlinked unchanged, repairs must be valid and collision-free, no local
  # mask may be empty, and loader/description coverage must pass.
  run_json_report "$final_report" \
    conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step5_vggt_repair.audit \
    --data-root "$DATA_ROOT" \
    --baseline-root "$BASELINE_ROOT" \
    --repair-root "$REPAIR_ROOT" \
    --root "$FINAL_ROOT" \
    --repair-plan "$plan" \
    --split "$split" \
    --min-description-coverage "$min_coverage"
}

echo "original+VGGT repair-only: workers=$N_WORKERS gpus=[$GPUS]"
echo "validation promotion gate: preserve all 2,232 original-valid leaves / 323 descriptions, reject collisions, zero invalid or empty leaves"
run_split val "$VAL_MIN_COVERAGE"

if [ "$WITH_TRAIN" -eq 1 ]; then
  echo "Final validation union passed; starting train repair preprocessing."
  run_split train "$TRAIN_MIN_COVERAGE"
  echo "Final train union passed its preservation, contract, empty-mask, and loader gates."
  echo "Training command: GPUS=$GPUS GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 bash scripts/train.sh original_vggt_repair_v1 $FINAL_ROOT"
else
  echo "Final validation union passed. Re-run with --with-train to build the train union."
fi
