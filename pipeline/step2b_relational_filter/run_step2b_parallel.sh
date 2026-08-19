#!/usr/bin/env bash
# Run the relational-affordance frame filter with one data shard per GPU.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  pipeline/step2b_relational_filter/run_step2b_parallel.sh <train|val|test>

Optional environment variables:
  GPUS, N_GPUS, ENV_NAME, DATA_ROOT, INPUT_ROOT, OUTPUT_ROOT, MODEL,
  EXTRA_ARGS, LOG_DIR
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi
if [ "$#" -ne 1 ]; then
  usage
  exit 2
fi

SPLIT="$1"
case "$SPLIT" in
  train|val|test) ;;
  *) echo "ERROR: invalid split: $SPLIT" >&2; usage; exit 2 ;;
esac

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-tasa}"
GPUS="${GPUS:-0,1,2,3}"
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
INPUT_ROOT="${INPUT_ROOT:-pipeline/step2_clipwithaffordance/clipwithaffordance_output}"
OUTPUT_ROOT="${OUTPUT_ROOT:-pipeline/step2b_relational_filter/clipwithaffordance_output}"
MODEL="${MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR="${LOG_DIR:-pipeline/step2b_relational_filter/logs}"

cd "$REPO_ROOT"
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1

# Accept either comma- or whitespace-separated GPU lists.
# shellcheck disable=SC2206
GPU_ARR=(${GPUS//,/ })
NUM_SHARDS="${#GPU_ARR[@]}"
[ "$NUM_SHARDS" -ge 1 ] || { echo "ERROR: no GPUs specified" >&2; exit 1; }

mkdir -p "$LOG_DIR"
echo "Step 2b: split=$SPLIT shards=$NUM_SHARDS gpus=[${GPU_ARR[*]}]"
echo "input=$INPUT_ROOT"
echo "output=$OUTPUT_ROOT"
echo "logs=$LOG_DIR/step2b_shard<i>.log"

pids=()
for shard in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$shard]}"
  log="$LOG_DIR/step2b_shard${shard}.log"
  echo "[*] shard $shard/$NUM_SHARDS on GPU $gpu -> $log"
  # EXTRA_ARGS intentionally uses shell word splitting, matching other runners.
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" python -m pipeline.step2b_relational_filter.relational_filter \
    --data_root "$DATA_ROOT" --split "$SPLIT" \
    --input_root "$INPUT_ROOT" --output_root "$OUTPUT_ROOT" \
    --model "$MODEL" --num_shards "$NUM_SHARDS" --shard "$shard" \
    $EXTRA_ARGS \
    > "$log" 2>&1 &
  pids+=("$!")
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done
if [ "$fail" -ne 0 ]; then
  echo "Some Step 2b shards failed; inspect $LOG_DIR and rerun to resume." >&2
  exit 1
fi
echo "All Step 2b shards finished."
