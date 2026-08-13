#!/usr/bin/env bash
# Strict Qwen2.5 relational anchors -> VGGT propagation -> original-first
# six-file repair union. Existing source/intermediate/experiment roots are
# immutable inputs or forbidden outputs; this runner writes only fresh roots.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: dataset/run_qwen25_relational_vggt_repair_experiment.sh [--dry-run] [--with-train]

Validation always runs first. It targets only descriptions with invalid and no
valid original leaves, requires strict Qwen2.5 reference/relation evidence,
propagates verified anchors with VGGT, and materializes an original-first union.

Options:
  --dry-run     Print every preprocessing, audit, and materialization command.
  --with-train  Build train only after validation passes its promotion gate.
  -h, --help    Show this help.

Safe runtime overrides:
  GPUS, N_WORKERS, PREPROCESS_ENV, DIRECT_TUNING_ARGS, VGGT_EXTRA_ARGS,
  VAL_MIN_COVERAGE, TRAIN_MIN_COVERAGE.

After --with-train succeeds, train with the unchanged loader:
  GPUS=0,1,2,3,4,5,6,7 GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 \
    bash scripts/train.sh qwen25_relational_vggt_repair_v1 \
    scenefun3d/processed_sam2_qwen25_relational_vggt_repair_v1
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
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-scenefun3d/preprocessing_experiments/qwen25_relational_vggt_repair_v1}"
ANCHOR_ROOT="${ANCHOR_ROOT:-$EXPERIMENT_ROOT/anchors}"
CACHE_ROOT="${CACHE_ROOT:-$EXPERIMENT_ROOT/clip_cache}"
LOG_ROOT="${LOG_ROOT:-$EXPERIMENT_ROOT/logs}"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-$EXPERIMENT_ROOT/diagnostics}"
PLAN_ROOT="${PLAN_ROOT:-$EXPERIMENT_ROOT/plans}"
REPAIR_ROOT="${REPAIR_ROOT:-scenefun3d/processed_sam2_qwen25_relational_vggt_repair_candidates_v1}"
FINAL_ROOT="${FINAL_ROOT:-scenefun3d/processed_sam2_qwen25_relational_vggt_repair_v1}"
GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
N_WORKERS="${N_WORKERS:-8}"

# Exactly 326/445: at least three newly covered descriptions, which is one
# better than the completed legacy-anchor repair v1 result (325/445).
VAL_COVERAGE_HARD_MIN="0.7325842696629213"
VAL_MIN_COVERAGE="${VAL_MIN_COVERAGE:-$VAL_COVERAGE_HARD_MIN}"
TRAIN_MIN_COVERAGE="${TRAIN_MIN_COVERAGE:-0.0}"
DIRECT_TUNING_ARGS="${DIRECT_TUNING_ARGS:-}"
VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:---local-files-only --sam-local-files-only}"

# These semantics are pinned. Tuning may add benign numeric flags but cannot
# remove strict evidence, the marked verifier, Qwen2.5, or plan isolation.
REQUIRED_DIRECT_ARGS="--local-files-only --v3-mode --v3-role-aware-targeting --v4-strict-relational-grounding --v3-conflict-filter --grounding-coordinate-mode qwen-resized-pixels --qwen-model Qwen/Qwen2.5-VL-7B-Instruct --qwen-max-pixels 2007040 --frame-stride 3 --signature-max-new-tokens 192 --qwen-max-new-tokens 256 --verification-max-new-tokens 192 --retrieval-full-weight 0.70 --retrieval-parent-weight 0.20 --retrieval-component-weight 0.10 --max-candidates 18 --max-per-video 6 --verified-anchor-target 2"
REQUIRED_VGGT_ARGS="--max-anchors 3 --context-frames 8 --context-window-seconds 3.0 --component-selection anchor-consensus --independent-anchor-video-votes --crop-semantic-center --crop-semantic-radius 0.15 --crop-semantic-neighborhood-points 2048 --crop-semantic-neighborhood-radius 0.25 --max-published-views 6 --min-published-view-component-f1 0.10"

cd "$REPO_ROOT"
command -v realpath >/dev/null 2>&1 || {
  echo "ERROR: realpath is required for source and output isolation checks." >&2
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
RESOLVED_DATA_ROOT="$(resolve_path "$DATA_ROOT")"
RESOLVED_BASELINE_ROOT="$(resolve_path "$BASELINE_ROOT")"
RESOLVED_EXPERIMENT_ROOT="$(resolve_path "$EXPERIMENT_ROOT")"
RESOLVED_ANCHOR_ROOT="$(resolve_path "$ANCHOR_ROOT")"
RESOLVED_CACHE_ROOT="$(resolve_path "$CACHE_ROOT")"
RESOLVED_REPAIR_ROOT="$(resolve_path "$REPAIR_ROOT")"
RESOLVED_FINAL_ROOT="$(resolve_path "$FINAL_ROOT")"
RESOLVED_EXPERIMENT_BASE="$(resolve_path scenefun3d/preprocessing_experiments)"

if [ "$RESOLVED_DATA_ROOT" != "$CANONICAL_DATA_ROOT" ]; then
  echo "ERROR: DATA_ROOT is pinned to $CANONICAL_DATA_ROOT" >&2
  exit 2
fi
if [ "$RESOLVED_BASELINE_ROOT" != "$CANONICAL_BASELINE_ROOT" ]; then
  echo "ERROR: BASELINE_ROOT is pinned to $CANONICAL_BASELINE_ROOT" >&2
  exit 2
fi
if [ "$(dirname "$RESOLVED_EXPERIMENT_ROOT")" != "$RESOLVED_EXPERIMENT_BASE" ]; then
  echo "ERROR: EXPERIMENT_ROOT must be a direct child of $RESOLVED_EXPERIMENT_BASE" >&2
  exit 2
fi
case "$(basename "$RESOLVED_EXPERIMENT_ROOT")" in
  qwen25_relational_vggt_repair_*) ;;
  *) echo "ERROR: EXPERIMENT_ROOT must use the qwen25_relational_vggt_repair namespace." >&2; exit 2 ;;
esac

for named_child in \
  "ANCHOR_ROOT:$RESOLVED_ANCHOR_ROOT" \
  "CACHE_ROOT:$RESOLVED_CACHE_ROOT" \
  "LOG_ROOT:$(resolve_path "$LOG_ROOT")" \
  "DIAGNOSTICS_ROOT:$(resolve_path "$DIAGNOSTICS_ROOT")" \
  "PLAN_ROOT:$(resolve_path "$PLAN_ROOT")"; do
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

for named_output in \
  "REPAIR_ROOT:$RESOLVED_REPAIR_ROOT:processed_sam2_qwen25_relational_vggt_repair_candidates_" \
  "FINAL_ROOT:$RESOLVED_FINAL_ROOT:processed_sam2_qwen25_relational_vggt_repair_"; do
  output_name="${named_output%%:*}"
  output_remainder="${named_output#*:}"
  output_path="${output_remainder%%:*}"
  required_prefix="${output_remainder#*:}"
  if [ "$(dirname "$output_path")" != "$CANONICAL_DATA_ROOT" ]; then
    echo "ERROR: $output_name must be a direct child of DATA_ROOT." >&2
    exit 2
  fi
  case "$(basename "$output_path")" in
    "$required_prefix"*) ;;
    *) echo "ERROR: $output_name must use prefix $required_prefix" >&2; exit 2 ;;
  esac
done

resolved_outputs=(
  "$RESOLVED_EXPERIMENT_ROOT"
  "$RESOLVED_REPAIR_ROOT"
  "$RESOLVED_FINAL_ROOT"
)
for ((left_index = 0; left_index < ${#resolved_outputs[@]}; left_index++)); do
  left="${resolved_outputs[$left_index]}"
  if path_overlaps "$left" "$CANONICAL_BASELINE_ROOT"; then
    echo "ERROR: output overlaps immutable original baseline: $left" >&2
    exit 2
  fi
  for ((right_index = left_index + 1; right_index < ${#resolved_outputs[@]}; right_index++)); do
    if path_overlaps "$left" "${resolved_outputs[$right_index]}"; then
      echo "ERROR: experiment, repair, and final roots must be distinct and non-nested." >&2
      exit 2
    fi
  done
done

for forbidden in \
  pipeline/step1_affordance/affordance_result \
  pipeline/step2_clipwithaffordance/clipwithaffordance_output \
  pipeline/step3_point_prediction/point_clipwithaffordance_output \
  pipeline/step6_molmo_merge/molmo_merge_output \
  scenefun3d/preprocessing_experiments/original_vggt_repair_v1 \
  scenefun3d/processed_sam2_original_vggt_repair_v1; do
  resolved_forbidden="$(resolve_path "$forbidden")"
  for output_path in "${resolved_outputs[@]}"; do
    if path_overlaps "$output_path" "$resolved_forbidden"; then
      echo "ERROR: output overlaps protected existing artifact: $resolved_forbidden" >&2
      exit 2
    fi
  done
done

if ! [[ "$N_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: N_WORKERS must be a positive integer." >&2
  exit 2
fi
if ! awk -v value="$VAL_MIN_COVERAGE" -v floor="$VAL_COVERAGE_HARD_MIN" \
  'BEGIN { ok=value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/; exit !(ok && value+0 >= floor+0 && value+0 <= 1) }'; then
  echo "ERROR: VAL_MIN_COVERAGE must be in [$VAL_COVERAGE_HARD_MIN, 1]." >&2
  exit 2
fi
if ! awk -v value="$TRAIN_MIN_COVERAGE" \
  'BEGIN { ok=value ~ /^([0-9]+([.][0-9]*)?|[.][0-9]+)$/; exit !(ok && value+0 >= 0 && value+0 <= 1) }'; then
  echo "ERROR: TRAIN_MIN_COVERAGE must be in [0, 1]." >&2
  exit 2
fi

# Do not allow tuning text to replace runner-controlled models, roots, plans,
# modes, sharding, or output semantics.
# shellcheck disable=SC2086
for tuning_arg in $DIRECT_TUNING_ARGS; do
  case "$tuning_arg" in
    --data-root|--data-root=*|--split|--split=*|--output-root|--output-root=*|\
    --cache-root|--cache-root=*|--description-plan|--description-plan=*|\
    --qwen-model|--qwen-model=*|--stage|--stage=*|--device|--device=*|\
    --num-shards|--num-shards=*|--shard|--shard=*|--v3-*|--v4-*|\
    --grounding-coordinate-mode|--grounding-coordinate-mode=*)
      echo "ERROR: DIRECT_TUNING_ARGS cannot override controlled option $tuning_arg" >&2
      exit 2
      ;;
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
  local direct_args="$DIRECT_TUNING_ARGS $REQUIRED_DIRECT_ARGS --description-plan $plan"
  local vggt_args="$VGGT_EXTRA_ARGS $REQUIRED_VGGT_ARGS"
  local direct_report="$DIAGNOSTICS_ROOT/$split/01_relational_anchor_contract.json"
  local alignment_report="$DIAGNOSTICS_ROOT/$split/02_anchor_semantic_alignment.json"
  local repair_report="$DIAGNOSTICS_ROOT/$split/03_repair_contract.json"
  local materialize_report="$DIAGNOSTICS_ROOT/$split/04_materialize.json"
  local final_report="$DIAGNOSTICS_ROOT/$split/05_final_contract.json"

  echo "qwen25-relational $split: scope=uncovered baseline=$BASELINE_ROOT/$split"
  echo "qwen25-relational $split: plan=$plan anchors=$ANCHOR_ROOT/$split"
  echo "qwen25-relational $split: repairs=$REPAIR_ROOT/$split final=$FINAL_ROOT/$split"

  if [ "$DRY_RUN" -eq 0 ]; then
    mkdir -p "$PLAN_ROOT" "$LOG_ROOT" "$DIAGNOSTICS_ROOT/$split"
  fi
  run_command \
    conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step5_vggt_repair.plan \
    --data-root "$DATA_ROOT" \
    --baseline-root "$BASELINE_ROOT" \
    --split "$split" \
    --scope uncovered \
    --output "$plan"

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
    --description-plan "$plan" \
    --min-status-coverage 1.0 \
    --min-description-coverage 0

  # Evaluation only. This report may diagnose semantic anchors but is never an
  # input to VGGT selection, cropping, publication, or materialization.
  run_json_report "$alignment_report" \
    conda run --no-capture-output -n "$PREPROCESS_ENV" \
    python -m pipeline.step4_vggt_legacy.semantic_alignment_audit \
    --data-root "$DATA_ROOT" \
    --anchor-root "$ANCHOR_ROOT/$split" \
    --split "$split" \
    --max-distance-m 0.20 \
    --max-anchors 3 \
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
    REPAIR_PLAN="$plan" \
    SAVE_DIR="$REPAIR_ROOT" \
    ALLOW_COPY_FALLBACK=0 \
    EXTRA_ARGS="$vggt_args" \
    LOG_DIR="$LOG_ROOT/vggt/$split" \
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

echo "strict Qwen2.5 + VGGT repair: workers=$N_WORKERS gpus=[$GPUS]"
echo "validation gate: preserve all valid originals, zero empty masks, coverage >= $VAL_MIN_COVERAGE (>=326/445)"
echo "useful target reported separately: 333/445 (10 additions over original)"
run_split val "$VAL_MIN_COVERAGE"

if [ "$WITH_TRAIN" -eq 1 ]; then
  echo "Validation promotion gate passed; starting train preprocessing."
  run_split train "$TRAIN_MIN_COVERAGE"
  echo "Train union passed preservation, contract, empty-mask, and loader gates."
  echo "Training command: GPUS=$GPUS GLOBAL_BATCH_SIZE=64 MAX_STEPS=50000 bash scripts/train.sh qwen25_relational_vggt_repair_v1 $FINAL_ROOT"
else
  echo "Validation passed. Re-run with --with-train to build the train union."
fi
