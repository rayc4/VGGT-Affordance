#!/usr/bin/env bash
# Run Step 6b with one persistent VGGT worker per GPU.
#
# Usage:
#   pipeline/step6b_vggt_track/run_step6b_parallel.sh <train|val> \
#     [--input-root DIR] [--output-root DIR] [--dry-run]
#
# Common overrides:
#   GPUS="0 1 2 3" ENV_NAME=affordvggt \
#     pipeline/step6b_vggt_track/run_step6b_parallel.sh val
#   EXTRA_ARGS="--local_files_only --context_frames 12" \
#     pipeline/step6b_vggt_track/run_step6b_parallel.sh train
set -euo pipefail

usage() {
  echo "Usage: $0 <train|val> [--input-root DIR] [--output-root DIR] [--dry-run]" >&2
}

if [ "$#" -lt 1 ]; then
  usage
  exit 2
fi

SPLIT="$1"
shift
DRY_RUN=0
if [ "$SPLIT" != "train" ] && [ "$SPLIT" != "val" ]; then
  echo "ERROR: split must be train or val (got '$SPLIT')." >&2
  exit 2
fi

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
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    *) echo "ERROR: unknown option: $1" >&2; usage; exit 2 ;;
  esac
done

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-affordvggt}"
N_GPUS="${N_GPUS:-4}"
GPUS="${GPUS:-$(seq 0 $((N_GPUS - 1)))}"
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
INPUT_ROOT="${INPUT_ROOT_ARG:-${INPUT_ROOT:-${STEP6_ROOT:-pipeline/step6_molmo_merge/molmo_merge_output}}}"
OUTPUT_ROOT="${OUTPUT_ROOT_ARG:-${OUTPUT_ROOT:-pipeline/step6b_vggt_track/vggt_track_output}}"
SCRIPT="${SCRIPT:-pipeline/step6b_vggt_track/vggt_track.py}"
LOG_DIR="${LOG_DIR:-pipeline/step6b_vggt_track/logs/$SPLIT}"
HF_MODEL="${HF_MODEL:-facebook/VGGT-1B}"
CKPT="${CKPT:-}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

cd "$REPO_ROOT"

# Match the other preprocessing runners and isolate the selected conda env.
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1

[ -f "$SCRIPT" ] || { echo "ERROR: script not found: $SCRIPT" >&2; exit 1; }
[ -d "$INPUT_ROOT/$SPLIT" ] || { echo "ERROR: Step 6 input not found: $INPUT_ROOT/$SPLIT" >&2; exit 1; }
[ -d "$DATA_ROOT" ] || { echo "ERROR: data root not found: $DATA_ROOT" >&2; exit 1; }

GPU_SPEC="${GPUS//,/ }"
# shellcheck disable=SC2206
GPU_ARR=($GPU_SPEC)
NUM_SHARDS="${#GPU_ARR[@]}"
[ "$NUM_SHARDS" -ge 1 ] || { echo "ERROR: no GPUs specified" >&2; exit 1; }

CKPT_ARGS=()
if [ -n "$CKPT" ]; then
  CKPT_ARGS=(--ckpt "$CKPT")
fi

mkdir -p "$OUTPUT_ROOT/$SPLIT" "$LOG_DIR"
echo "env=$ENV_NAME split=$SPLIT workers=$NUM_SHARDS gpus=[${GPU_ARR[*]}]"
echo "input=$INPUT_ROOT/$SPLIT"
echo "output=$OUTPUT_ROOT/$SPLIT"
echo "logs=$LOG_DIR/step6b_shard<i>.log"

print_command() {
  local shard="$1"
  local gpu="$2"
  local log="$3"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
  printf '%q ' python "$SCRIPT" \
    --split "$SPLIT" \
    --data_root "$DATA_ROOT" \
    --input_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --hf_model "$HF_MODEL" \
    --num_shards "$NUM_SHARDS" \
    --shard "$shard" \
    "${CKPT_ARGS[@]}"
  # Match the intentional EXTRA_ARGS word splitting used during execution.
  # shellcheck disable=SC2086
  for argument in $EXTRA_ARGS; do
    printf '%q ' "$argument"
  done
  printf '> %q 2>&1\n' "$log"
}

if [ "$DRY_RUN" -eq 1 ]; then
  for shard in "${!GPU_ARR[@]}"; do
    print_command "$shard" "${GPU_ARR[$shard]}" "$LOG_DIR/step6b_shard${shard}.log"
  done
  exit 0
fi

pids=()
for shard in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$shard]}"
  log="$LOG_DIR/step6b_shard${shard}.log"
  echo "[*] shard $shard/$NUM_SHARDS on GPU $gpu -> $log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
    --split "$SPLIT" \
    --data_root "$DATA_ROOT" \
    --input_root "$INPUT_ROOT" \
    --output_root "$OUTPUT_ROOT" \
    --hf_model "$HF_MODEL" \
    --num_shards "$NUM_SHARDS" \
    --shard "$shard" \
    "${CKPT_ARGS[@]}" \
    $EXTRA_ARGS \
    > "$log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "ERROR: one or more Step 6b workers failed; inspect $LOG_DIR." >&2
  exit 1
fi
echo "All Step 6b shards finished."
