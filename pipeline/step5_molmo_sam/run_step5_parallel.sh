#!/usr/bin/env bash
#
# Parallel Step 5 (Molmo pointing + SAM segmentation) runner.
#
# Runs one process per GPU, each handling a disjoint 1/N slice of the crop
# images (data parallelism). Molmo-7B (bf16) + SAM-huge fit comfortably on a
# single H200, so sharding the *data* gives roughly linear (~Nx) speedup.
#
# Re-running is safe: output is checkpointed per image ({name}_mask_data.npz)
# and already-completed images are skipped, so an interrupted run just resumes.
# Failures (no points / SAM failure) are checkpointed too, as empty npz
# markers; delete an image's npz to force a retry.
#
# Usage:
#   pipeline/step5_molmo_sam/run_step5_parallel.sh [train|val] \
#     [--input-root DIR] [--output-root DIR]
#
# Override defaults via env vars, e.g.:
#   GPUS="0 2 3 5" SPLIT=val pipeline/step5_molmo_sam/run_step5_parallel.sh
#   ENV_NAME=tasa N_GPUS=8 pipeline/step5_molmo_sam/run_step5_parallel.sh
#
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  pipeline/step5_molmo_sam/run_step5_parallel.sh [train|val] \
    [--input-root DIR] [--output-root DIR]

Both roots contain split subdirectories. INPUT_ROOT/<split> is read and
OUTPUT_ROOT/<split> is created. The same paths can be set with environment
variables INPUT_ROOT and OUTPUT_ROOT. AFFORDANCE_ROOT selects the Step 1 tree.
EOF
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SPLIT_ARG=""
if [ "${1:-}" = "train" ] || [ "${1:-}" = "val" ]; then
  SPLIT_ARG="$1"
  shift
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
    *) echo "ERROR: unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

# ---- config (override via env) -------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-tasa}"                     # conda env with transformers 4.51.3
N_GPUS="${N_GPUS:-4}"                            # number of GPUs to fan out over
GPUS="${GPUS:-$(seq 0 $((N_GPUS - 1)))}"         # explicit GPU id list overrides N_GPUS
SPLIT="${SPLIT_ARG:-${SPLIT:-val}}"             # train | val
case "$SPLIT" in
  train|val) ;;
  *) echo "ERROR: invalid split: $SPLIT (expected train or val)." >&2; usage; exit 2 ;;
esac
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
INPUT_ROOT="${INPUT_ROOT_ARG:-${INPUT_ROOT:-${ROOT_DIR:-pipeline/step4_crop_images/seg_image_output/point_clipwithaffordance_output}}}"
OUTPUT_ROOT="${OUTPUT_ROOT_ARG:-${OUTPUT_ROOT:-pipeline/step5_molmo_sam/molmo_output}}"
AFFORDANCE_ROOT="${AFFORDANCE_ROOT:-pipeline/step1_affordance/affordance_result}"
SCRIPT="${SCRIPT:-pipeline/step5_molmo_sam/molmo_sam.py}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                     # any extra flags
LOG_DIR="${LOG_DIR:-pipeline/step5_molmo_sam/logs/$SPLIT}"
# --------------------------------------------------------------------------

cd "$REPO_ROOT"

# Activate the conda env (matches run_step3_parallel.sh convention).
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1                        # keep tasa's numpy/scipy off user-site

[ -f "$SCRIPT" ] || { echo "ERROR: script not found: $SCRIPT (in $REPO_ROOT)" >&2; exit 1; }
[ -d "$INPUT_ROOT/$SPLIT" ] || { echo "ERROR: input dir not found: $INPUT_ROOT/$SPLIT (in $REPO_ROOT)" >&2; exit 1; }

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
echo "logs -> $LOG_DIR/step5_shard<i>.log"

pids=()
for shard in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$shard]}"
  log="$LOG_DIR/step5_shard${shard}.log"
  echo "[*] shard $shard/$NUM_SHARDS on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
    --split "$SPLIT" --data_root "$DATA_ROOT" \
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
  echo "Some shards failed -- check $LOG_DIR, then just re-run to resume (done images are skipped)." >&2
  exit 1
fi
