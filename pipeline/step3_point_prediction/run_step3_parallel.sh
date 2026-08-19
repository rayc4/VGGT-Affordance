#!/usr/bin/env bash
#
# Parallel Step 3 (Qwen3-VL point prediction) runner.
#
# Runs one process per GPU, each handling a disjoint 1/N slice of the visit_ids
# (data parallelism). This is the right way to use multiple GPUs here: the 7B
# model fits on a single H200, so sharding the *model* across GPUs gives no
# speedup -- only one GPU computes at a time. Sharding the *data* instead gives
# roughly linear (~Nx) speedup because every GPU is fully busy on its own copy.
#
# Re-running is safe: output is checkpointed per video ({video_id}_point.json)
# and already-completed videos are skipped, so an interrupted run just resumes.
#
# Usage:
#   pipeline/step3_point_prediction/run_step3_parallel.sh <train|val> \
#     [--input-root DIR] [--output-root DIR]
#
# Override other defaults via env vars, e.g.:
#   GPUS="0 1 2 3" pipeline/step3_point_prediction/run_step3_parallel.sh val
#   ENV_NAME=tasa N_GPUS=8 pipeline/step3_point_prediction/run_step3_parallel.sh train
#   INPUT_ROOT=/path/to/step2b OUTPUT_ROOT=/path/to/points/val \
#     pipeline/step3_point_prediction/run_step3_parallel.sh val
#
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  pipeline/step3_point_prediction/run_step3_parallel.sh <train|val> \
    [--input-root DIR] [--output-root DIR]

Directory layout:
  INPUT_ROOT   Step 2/2b root containing <split>/<visit_id>/*_result.json
  OUTPUT_ROOT  Step 3 root; outputs go under <split>/<visit_id>/*_point.json

The same paths can be set with the INPUT_ROOT and OUTPUT_ROOT environment
variables. CLIP_ROOT remains supported as a legacy alias for INPUT_ROOT.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ "$#" -lt 1 ]; then
  echo "ERROR: SPLIT is required." >&2
  usage
  exit 2
fi

SPLIT="$1"
shift
case "$SPLIT" in
  train|val) ;;
  *)
    echo "ERROR: invalid SPLIT: $SPLIT (expected train or val)." >&2
    usage
    exit 2
    ;;
esac

INPUT_ROOT_ARG=""
OUTPUT_ROOT_ARG=""
while [ "$#" -gt 0 ]; do
  case "$1" in
    --input-root)
      [ "$#" -ge 2 ] || { echo "ERROR: --input-root requires a directory." >&2; usage; exit 2; }
      INPUT_ROOT_ARG="$2"
      shift 2
      ;;
    --output-root)
      [ "$#" -ge 2 ] || { echo "ERROR: --output-root requires a directory." >&2; usage; exit 2; }
      OUTPUT_ROOT_ARG="$2"
      shift 2
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage
      exit 2
      ;;
  esac
done

# ---- config (override via env) -------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-tasa}"                     # conda env with Qwen3-VL support + qwen_vl_utils
N_GPUS="${N_GPUS:-8}"                            # number of GPUs to fan out over
GPUS="${GPUS:-$(seq 0 $((N_GPUS - 1)))}"         # explicit GPU id list overrides N_GPUS
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
INPUT_ROOT="${INPUT_ROOT_ARG:-${INPUT_ROOT:-${CLIP_ROOT:-pipeline/step2b_relational_filter/clipwithaffordance_output}}}"
input_name="$(basename "${INPUT_ROOT%/}")"
input_name="${input_name%%_output*}"
OUTPUT_ROOT="${OUTPUT_ROOT_ARG:-${OUTPUT_ROOT:-pipeline/step3_point_prediction/point_${input_name}_output}}"
AFFORDANCE_ROOT="${AFFORDANCE_ROOT:-pipeline/step1_affordance/affordance_result}"
SCRIPT="${SCRIPT:-pipeline/step3_point_prediction/qwen_point.py}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                     # any extra flags, e.g. --disable_validation
LOG_DIR="${LOG_DIR:-pipeline/step3_point_prediction/logs}"
# --------------------------------------------------------------------------

cd "$REPO_ROOT"

# Activate the conda env (matches scripts/download_parallel.sh convention).
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1                        # keep tasa's numpy/scipy off user-site

[ -f "$SCRIPT" ] || { echo "ERROR: script not found: $SCRIPT (in $REPO_ROOT)" >&2; exit 1; }
[ -d "$INPUT_ROOT/$SPLIT" ] || {
  echo "ERROR: Step 3 input split not found: $INPUT_ROOT/$SPLIT" >&2
  echo "Expected INPUT_ROOT to contain a $SPLIT/ directory." >&2
  exit 1
}

# Normalize GPU list to an array; shard count = number of GPUs.
# Unquoted word-split so both "0 1 2" and seq's newline-separated output work.
# shellcheck disable=SC2206
GPU_ARR=($GPUS)
NUM_SHARDS="${#GPU_ARR[@]}"
[ "$NUM_SHARDS" -ge 1 ] || { echo "ERROR: no GPUs specified" >&2; exit 1; }

mkdir -p "$OUTPUT_ROOT/$SPLIT" "$LOG_DIR"
echo "env=$ENV_NAME  split=$SPLIT  num_shards=$NUM_SHARDS  gpus=[${GPU_ARR[*]}]"
echo "input=$INPUT_ROOT/$SPLIT"
echo "output=$OUTPUT_ROOT/$SPLIT"
echo "python=$(which python)"
echo "logs -> $LOG_DIR/step3_shard<i>.log"

pids=()
for shard in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$shard]}"
  log="$LOG_DIR/step3_shard${shard}.log"
  echo "[*] shard $shard/$NUM_SHARDS on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
    --data_root "$DATA_ROOT" --split "$SPLIT" \
    --input_root "$INPUT_ROOT" --output_root "$OUTPUT_ROOT" \
    --affordance_root "$AFFORDANCE_ROOT" \
    --num_shards "$NUM_SHARDS" --shard "$shard" \
    $EXTRA_ARGS \
    > "$log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

if [ "$fail" -eq 0 ]; then
  echo "All shards finished."
else
  echo "Some shards failed -- check $LOG_DIR, then just re-run to resume (done videos are skipped)." >&2
  exit 1
fi
