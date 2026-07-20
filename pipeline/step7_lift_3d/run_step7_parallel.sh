#!/usr/bin/env bash
#
# Parallel Step 7 (Molmo 2D-to-3D mask lifting) runner.
#
# Runs one process per GPU, each handling a disjoint 1/N slice of the visits
# (data parallelism). The step is projection-only (no big models), so VRAM is
# not a constraint and sharding gives roughly linear (~Nx) speedup.
#
# Re-running is safe: output is checkpointed per visit
# (lift_results_<visit_id>.json) and already-completed visits are skipped, so
# an interrupted run just resumes. Delete a visit's json to force a retry.
# After all shards finish, per-visit files are merged into lift_results_all.json.
#
# Usage:
#   pipeline/step7_lift_3d/run_step7_parallel.sh
#
# Override defaults via env vars, e.g.:
#   GPUS="0 2 3 5" SPLIT=val pipeline/step7_lift_3d/run_step7_parallel.sh
#   ENV_NAME=tasa N_GPUS=8 pipeline/step7_lift_3d/run_step7_parallel.sh
#
set -euo pipefail

# ---- config (override via env) -------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-tasa}"                     # conda env with open3d/cv2
N_GPUS="${N_GPUS:-4}"                            # number of GPUs to fan out over
GPUS="${GPUS:-$(seq 0 $((N_GPUS - 1)))}"         # explicit GPU id list overrides N_GPUS
SPLIT="${SPLIT:-train}"                          # train | val
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
MOLMO_ROOT="${MOLMO_ROOT:-pipeline/step6_molmo_merge/molmo_merge_output/$SPLIT}"
OUTPUT_DIR="${OUTPUT_DIR:-pipeline/step7_lift_3d/lift_output/$SPLIT}"
SCRIPT="${SCRIPT:-pipeline/step7_lift_3d/molmo_lift_2d_to_3d.py}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                     # any extra flags
LOG_DIR="${LOG_DIR:-pipeline/step7_lift_3d/logs/$SPLIT}"
# --------------------------------------------------------------------------

cd "$REPO_ROOT"

# Activate the conda env (matches run_step5_parallel.sh convention).
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
export PYTHONNOUSERSITE=1                        # keep tasa's numpy/scipy off user-site

[ -f "$SCRIPT" ] || { echo "ERROR: script not found: $SCRIPT (in $REPO_ROOT)" >&2; exit 1; }
[ -d "$MOLMO_ROOT" ] || { echo "ERROR: input dir not found: $MOLMO_ROOT (in $REPO_ROOT)" >&2; exit 1; }

# Normalize GPU list to an array; shard count = number of GPUs.
# Unquoted word-split so both "0 1 2" and seq's newline-separated output work.
# shellcheck disable=SC2206
GPU_ARR=($GPUS)
NUM_SHARDS="${#GPU_ARR[@]}"
[ "$NUM_SHARDS" -ge 1 ] || { echo "ERROR: no GPUs specified" >&2; exit 1; }

mkdir -p "$LOG_DIR"
echo "env=$ENV_NAME  split=$SPLIT  num_shards=$NUM_SHARDS  gpus=[${GPU_ARR[*]}]"
echo "python=$(which python)"
echo "logs -> $LOG_DIR/step7_shard<i>.log"

pids=()
for shard in "${!GPU_ARR[@]}"; do
  gpu="${GPU_ARR[$shard]}"
  log="$LOG_DIR/step7_shard${shard}.log"
  echo "[*] shard $shard/$NUM_SHARDS on GPU $gpu -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
    --data_root "$DATA_ROOT" --split "$SPLIT" \
    --molmo_root "$MOLMO_ROOT" --output_dir "$OUTPUT_DIR" \
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
  python "$SCRIPT" --data_root "$DATA_ROOT" --split "$SPLIT" \
    --output_dir "$OUTPUT_DIR" --merge_only
  echo "All shards finished."
else
  echo "Some shards failed -- check $LOG_DIR, then just re-run to resume (done visits are skipped)." >&2
  exit 1
fi
