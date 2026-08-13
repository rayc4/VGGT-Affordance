#!/usr/bin/env bash
# Validation-only v3g experiment:
# v3e primary proposals + independent exact-v2 full-description hedges ->
# proposal oracle gate -> prediction-only VGGT/3D max-3 selection -> exact
# legacy preprocessing. All artifacts are fresh and isolated from prior runs.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: dataset/run_direct_vggt_v3g_experiment.sh [--dry-run]

Runs validation with one worker per visible GPU by default. Overrides:
  GPUS, N_WORKERS, PREPROCESS_ENV, DATA_ROOT, EXPERIMENT_ROOT,
  PROPOSAL_ROOT, CLIP_CACHE_ROOT, CONSENSUS_ROOT, PROCESSED_ROOT,
  LOG_ROOT, DIAGNOSTICS_ROOT, DIRECT_EXTRA_ARGS, CONSENSUS_EXTRA_ARGS,
  VGGT_EXTRA_ARGS, V3G_PRIMARY_ANCHOR_TARGET, V3G_HEDGE_ANCHOR_TARGET,
  PROPOSAL_POOL_MAX, SELECTED_MAX_ANCHORS.
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
EXPERIMENT_ROOT="${EXPERIMENT_ROOT:-scenefun3d/preprocessing_experiments/direct_vggt_v3g}"
PROPOSAL_ROOT="${PROPOSAL_ROOT:-$EXPERIMENT_ROOT/proposals}"
CLIP_CACHE_ROOT="${CLIP_CACHE_ROOT:-$EXPERIMENT_ROOT/clip_cache}"
CONSENSUS_ROOT="${CONSENSUS_ROOT:-$EXPERIMENT_ROOT/consensus}"
LOG_ROOT="${LOG_ROOT:-$EXPERIMENT_ROOT/logs}"
DIAGNOSTICS_ROOT="${DIAGNOSTICS_ROOT:-$EXPERIMENT_ROOT/diagnostics/val}"
PROCESSED_ROOT="${PROCESSED_ROOT:-scenefun3d/processed_sam2_vggt_direct_v3g}"
GPUS="${GPUS-}"
N_WORKERS="${N_WORKERS-}"

# Generation can publish at most six v3e-primary and four independent-v2
# proposals. The oracle and selector must see the full pool; only the selector
# may reduce it to the legacy max-three anchor contract.
V3G_PRIMARY_ANCHOR_TARGET="${V3G_PRIMARY_ANCHOR_TARGET:-6}"
V3G_HEDGE_ANCHOR_TARGET="${V3G_HEDGE_ANCHOR_TARGET:-4}"
PROPOSAL_POOL_MAX="${PROPOSAL_POOL_MAX:-$((V3G_PRIMARY_ANCHOR_TARGET + V3G_HEDGE_ANCHOR_TARGET))}"
SELECTED_MAX_ANCHORS="${SELECTED_MAX_ANCHORS:-3}"
for integer_name in \
  V3G_PRIMARY_ANCHOR_TARGET \
  V3G_HEDGE_ANCHOR_TARGET \
  PROPOSAL_POOL_MAX \
  SELECTED_MAX_ANCHORS; do
  integer_value="${!integer_name}"
  if ! [[ "$integer_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $integer_name must be a positive integer." >&2
    exit 2
  fi
done
EXPECTED_PROPOSAL_POOL_MAX=$((V3G_PRIMARY_ANCHOR_TARGET + V3G_HEDGE_ANCHOR_TARGET))
if [ "$PROPOSAL_POOL_MAX" -ne "$EXPECTED_PROPOSAL_POOL_MAX" ]; then
  echo "ERROR: PROPOSAL_POOL_MAX=$PROPOSAL_POOL_MAX must equal primary + hedge = $EXPECTED_PROPOSAL_POOL_MAX." >&2
  exit 2
fi
if [ "$V3G_HEDGE_ANCHOR_TARGET" -ne 4 ]; then
  echo "ERROR: the v3g proposal contract publishes exactly four hedge anchors." >&2
  exit 2
fi
if [ "$SELECTED_MAX_ANCHORS" -ge "$PROPOSAL_POOL_MAX" ]; then
  echo "ERROR: SELECTED_MAX_ANCHORS must be smaller than PROPOSAL_POOL_MAX." >&2
  exit 2
fi

DIRECT_EXTRA_ARGS="${DIRECT_EXTRA_ARGS:---local-files-only --v3-mode --v3-role-aware-targeting --v3-target-surface-grounding --v3-using-instrument-targeting --v3-contact-region-grounding --v3g-independent-full-description-hedge --v3-conflict-filter --grounding-coordinate-mode qwen-resized-pixels --qwen-max-pixels 2007040 --max-candidates 18 --max-per-video 18}"
CONSENSUS_EXTRA_ARGS="${CONSENSUS_EXTRA_ARGS:---local-files-only --variant-aware-proposals --cluster-radius 0.12 --min-video-support 2 --context-frames 8 --context-window-seconds 3.0 --single-proposal-min-track-frames 3}"
VGGT_EXTRA_ARGS="${VGGT_EXTRA_ARGS:---local-files-only --sam-local-files-only --variant-aware-proposals --component-selection anchor-consensus --crop-semantic-center --crop-semantic-radius 0.15 --crop-semantic-neighborhood-points 2048 --crop-semantic-neighborhood-radius 0.25 --max-published-views 6 --min-published-view-component-f1 0.10}"

EXPECTED_DESCRIPTIONS="${EXPECTED_DESCRIPTIONS:-445}"
PROPOSAL_ORACLE_HARD_MIN_ALIGNED="${PROPOSAL_ORACLE_HARD_MIN_ALIGNED:-312}"
PROPOSAL_ORACLE_TARGET_ALIGNED="${PROPOSAL_ORACLE_TARGET_ALIGNED:-334}"
DIRECT_MIN_COVERAGE="${DIRECT_MIN_COVERAGE:-0.78}"
ALIGNMENT_MIN_COVERAGE="${ALIGNMENT_MIN_COVERAGE:-0.70}"
ALIGNMENT_MAX_DISTANCE_M="${ALIGNMENT_MAX_DISTANCE_M:-0.20}"
FINAL_MIN_COVERAGE="${FINAL_MIN_COVERAGE:-0.7258}"
for integer_name in \
  EXPECTED_DESCRIPTIONS \
  PROPOSAL_ORACLE_HARD_MIN_ALIGNED \
  PROPOSAL_ORACLE_TARGET_ALIGNED; do
  integer_value="${!integer_name}"
  if ! [[ "$integer_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $integer_name must be a positive integer." >&2
    exit 2
  fi
done
if [ "$PROPOSAL_ORACLE_HARD_MIN_ALIGNED" -gt "$PROPOSAL_ORACLE_TARGET_ALIGNED" ] || \
   [ "$PROPOSAL_ORACLE_TARGET_ALIGNED" -gt "$EXPECTED_DESCRIPTIONS" ]; then
  echo "ERROR: require oracle hard minimum <= target <= expected descriptions." >&2
  exit 2
fi

cd "$REPO_ROOT"
command -v realpath >/dev/null 2>&1 || {
  echo "ERROR: realpath is required for v3g output isolation checks." >&2
  exit 1
}

FORBIDDEN_ARTIFACT_ROOTS=(
  "$(realpath -m "$DATA_ROOT/processed_sam2")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v1")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v2")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v3a")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v3b")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v3c")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v3d")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v3e")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v3f")"
  "$(realpath -m "$DATA_ROOT/processed_sam2_vggt_direct_v1")"
  "$(realpath -m "$DATA_ROOT/processed_sam2_vggt_direct_v2")"
  "$(realpath -m "$DATA_ROOT/processed_sam2_vggt_direct_v3e")"
  "$(realpath -m "$DATA_ROOT/processed_sam2_vggt_direct_v3f")"
)

assert_v3g_path() {
  local name="$1" value="$2" resolved forbidden
  resolved="$(realpath -m "$value")"
  case "$resolved" in
    *v3g*) ;;
    *)
      echo "ERROR: $name must resolve inside a clearly named v3g namespace: $resolved" >&2
      exit 2
      ;;
  esac
  for forbidden in "${FORBIDDEN_ARTIFACT_ROOTS[@]}"; do
    case "$resolved" in
      "$forbidden"|"$forbidden"/*)
        echo "ERROR: $name overlaps a forbidden artifact tree: $forbidden" >&2
        exit 2
        ;;
    esac
    case "$forbidden" in
      "$resolved"/*)
        echo "ERROR: $name is too broad and contains a forbidden artifact tree: $forbidden" >&2
        exit 2
        ;;
    esac
  done
}

assert_v3g_path EXPERIMENT_ROOT "$EXPERIMENT_ROOT"
assert_v3g_path PROPOSAL_ROOT "$PROPOSAL_ROOT"
assert_v3g_path CLIP_CACHE_ROOT "$CLIP_CACHE_ROOT"
assert_v3g_path CONSENSUS_ROOT "$CONSENSUS_ROOT"
assert_v3g_path LOG_ROOT "$LOG_ROOT"
assert_v3g_path DIAGNOSTICS_ROOT "$DIAGNOSTICS_ROOT"
assert_v3g_path PROCESSED_ROOT "$PROCESSED_ROOT"

RESOLVED_EXPERIMENT_ROOT="$(realpath -m "$EXPERIMENT_ROOT")"
for named_path in \
  "PROPOSAL_ROOT:$PROPOSAL_ROOT" \
  "CLIP_CACHE_ROOT:$CLIP_CACHE_ROOT" \
  "CONSENSUS_ROOT:$CONSENSUS_ROOT" \
  "LOG_ROOT:$LOG_ROOT" \
  "DIAGNOSTICS_ROOT:$DIAGNOSTICS_ROOT"; do
  name="${named_path%%:*}"
  value="${named_path#*:}"
  resolved="$(realpath -m "$value")"
  case "$resolved" in
    "$RESOLVED_EXPERIMENT_ROOT"/*) ;;
    *)
      echo "ERROR: $name must be contained by EXPERIMENT_ROOT." >&2
      exit 2
      ;;
  esac
done
RESOLVED_PROCESSED_ROOT="$(realpath -m "$PROCESSED_ROOT")"
if [[ "$RESOLVED_PROCESSED_ROOT" == "$RESOLVED_EXPERIMENT_ROOT" ||
      "$RESOLVED_PROCESSED_ROOT" == "$RESOLVED_EXPERIMENT_ROOT/"* ||
      "$RESOLVED_EXPERIMENT_ROOT" == "$RESOLVED_PROCESSED_ROOT/"* ]]; then
  echo "ERROR: PROCESSED_ROOT and EXPERIMENT_ROOT must be separate, non-nested trees." >&2
  exit 2
fi

run_command() {
  if [ "$DRY_RUN" -eq 1 ]; then
    printf '%q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

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

runner_args=()
if [ "$DRY_RUN" -eq 1 ]; then runner_args+=(--dry-run); fi

PROPOSAL_CONTRACT_REPORT="$DIAGNOSTICS_ROOT/01_proposal_contract.json"
PROPOSAL_ORACLE_REPORT="$DIAGNOSTICS_ROOT/02_proposal_semantic_oracle.json"
PROPOSAL_ORACLE_GATE_REPORT="$DIAGNOSTICS_ROOT/03_proposal_oracle_gate.json"
SELECTED_CONTRACT_REPORT="$DIAGNOSTICS_ROOT/04_selected_contract.json"
SELECTED_ALIGNMENT_REPORT="$DIAGNOSTICS_ROOT/05_selected_semantic_alignment.json"
LEGACY_CONTRACT_REPORT="$DIAGNOSTICS_ROOT/06_legacy_contract.json"

echo "v3g validation: proposals=$PROPOSAL_ROOT/val"
echo "v3g validation: consensus=$CONSENSUS_ROOT/val"
echo "v3g validation: legacy=$PROCESSED_ROOT/val"
echo "v3g validation: diagnostics=$DIAGNOSTICS_ROOT"
echo "workers=${N_WORKERS:-one-per-visible-gpu} gpus=${GPUS:-all-visible}"
echo "proposal pool: primary=$V3G_PRIMARY_ANCHOR_TARGET independent-v2=$V3G_HEDGE_ANCHOR_TARGET oracle/selector=$PROPOSAL_POOL_MAX publish=$SELECTED_MAX_ANCHORS"
echo "gates: proposal/direct=$DIRECT_MIN_COVERAGE (>=348) oracle target=$PROPOSAL_ORACLE_TARGET_ALIGNED/$EXPECTED_DESCRIPTIONS hard=$PROPOSAL_ORACLE_HARD_MIN_ALIGNED/$EXPECTED_DESCRIPTIONS selected/direct=$DIRECT_MIN_COVERAGE (>=348) selected/alignment=$ALIGNMENT_MIN_COVERAGE (>=312) final=$FINAL_MIN_COVERAGE (>=323)"

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
  EXTRA_ARGS="$DIRECT_EXTRA_ARGS --verified-anchor-target $V3G_PRIMARY_ANCHOR_TARGET" \
  LOG_DIR="$LOG_ROOT/proposals/val" \
  bash dataset/run_direct_anchors_parallel.sh "${runner_args[@]}"

run_json_audit "$PROPOSAL_CONTRACT_REPORT" \
  conda run --no-capture-output -n "$PREPROCESS_ENV" \
  python -m pipeline.step1_direct_anchors.audit \
  --data-root "$DATA_ROOT" \
  --root "$PROPOSAL_ROOT" \
  --split val \
  --min-description-coverage "$DIRECT_MIN_COVERAGE"

# Evaluation-only upper bound. It may stop the experiment but cannot write or
# influence the prediction-only proposal/anchor selection.
run_json_audit "$PROPOSAL_ORACLE_REPORT" \
  conda run --no-capture-output -n "$PREPROCESS_ENV" \
  python -m pipeline.step4_vggt_legacy.semantic_alignment_audit \
  --data-root "$DATA_ROOT" \
  --anchor-root "$PROPOSAL_ROOT/val" \
  --split val \
  --max-distance-m "$ALIGNMENT_MAX_DISTANCE_M" \
  --max-anchors "$PROPOSAL_POOL_MAX" \
  --variant-aware-proposals \
  --include-description-results

run_json_audit "$PROPOSAL_ORACLE_GATE_REPORT" \
  conda run --no-capture-output -n "$PREPROCESS_ENV" \
  python -m pipeline.step2_vggt_anchor_consensus.oracle_gate \
  --audit-json "$PROPOSAL_ORACLE_REPORT" \
  --expected-descriptions "$EXPECTED_DESCRIPTIONS" \
  --expected-max-anchors "$PROPOSAL_POOL_MAX" \
  --hard-minimum-aligned "$PROPOSAL_ORACLE_HARD_MIN_ALIGNED" \
  --target-aligned "$PROPOSAL_ORACLE_TARGET_ALIGNED"

run_command env \
  REPO_ROOT="$REPO_ROOT" \
  ENV_NAME="$PREPROCESS_ENV" \
  SPLIT=val \
  GPUS="$GPUS" \
  N_WORKERS="$N_WORKERS" \
  DATA_ROOT="$DATA_ROOT" \
  V3G_EXPERIMENT_ROOT="$EXPERIMENT_ROOT" \
  PROPOSAL_ROOT="$PROPOSAL_ROOT/val" \
  OUTPUT_ROOT="$CONSENSUS_ROOT/val" \
  MAX_PROPOSALS="$PROPOSAL_POOL_MAX" \
  MAX_ANCHORS="$SELECTED_MAX_ANCHORS" \
  EXTRA_ARGS="$CONSENSUS_EXTRA_ARGS" \
  LOG_DIR="$LOG_ROOT/consensus/val" \
  bash dataset/run_v3g_consensus_parallel.sh "${runner_args[@]}"

run_json_audit "$SELECTED_CONTRACT_REPORT" \
  conda run --no-capture-output -n "$PREPROCESS_ENV" \
  python -m pipeline.step1_direct_anchors.audit \
  --data-root "$DATA_ROOT" \
  --root "$CONSENSUS_ROOT" \
  --split val \
  --min-description-coverage "$DIRECT_MIN_COVERAGE"

run_json_audit "$SELECTED_ALIGNMENT_REPORT" \
  conda run --no-capture-output -n "$PREPROCESS_ENV" \
  python -m pipeline.step4_vggt_legacy.semantic_alignment_audit \
  --data-root "$DATA_ROOT" \
  --anchor-root "$CONSENSUS_ROOT/val" \
  --split val \
  --max-distance-m "$ALIGNMENT_MAX_DISTANCE_M" \
  --max-anchors "$SELECTED_MAX_ANCHORS" \
  --variant-aware-proposals \
  --min-aligned-coverage "$ALIGNMENT_MIN_COVERAGE" \
  --include-description-results

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
  LOG_DIR="$LOG_ROOT/vggt_legacy/val" \
  bash dataset/run_preprocess_vggt_legacy_parallel.sh "${runner_args[@]}"

run_json_audit "$LEGACY_CONTRACT_REPORT" \
  conda run --no-capture-output -n "$PREPROCESS_ENV" \
  python -m pipeline.step4_vggt_legacy.audit \
  --data-root "$DATA_ROOT" \
  --root "$PROCESSED_ROOT" \
  --split val \
  --min-description-coverage "$FINAL_MIN_COVERAGE" \
  --require-hardlink-dedup
