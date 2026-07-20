#!/usr/bin/env bash
#
# Parallel SAM2 preprocessing (chunk point clouds from lifted mask_result.json).
#
# The step is I/O-bound (per-visit laser scan PLY reads), so it fans out over
# CPU worker processes, each handling a disjoint 1/N slice of the visits.
# Workers are spread round-robin over the visible GPUs. By default the script
# starts one worker per GPU; N_WORKERS can still override that parallelism.
#
# Re-running is safe: a desc is skipped when all six of its output files
# already exist, so an interrupted run just resumes. Pass
# EXTRA_ARGS="--overwrite" to force reprocessing.
#
# Usage:
#   dataset/run_preprocess_sam2_parallel.sh
#   dataset/run_preprocess_sam2_parallel.sh --dry-run
#
# Override defaults via env vars, e.g.:
#   SPLIT=val N_WORKERS=16 dataset/run_preprocess_sam2_parallel.sh
#   GPUS="0,2" EXTRA_ARGS="--radius 0.2" dataset/run_preprocess_sam2_parallel.sh
#   CUDA_VISIBLE_DEVICES="0,2" dataset/run_preprocess_sam2_parallel.sh
#
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
for arg in "$@"; do
  case "$arg" in
    --dry-run)
      DRY_RUN=1
      ;;
    -h|--help)
      echo "Usage: $0 [--dry-run]"
      echo "  --dry-run  Print resolved shard commands without launching workers."
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $arg" >&2
      echo "Usage: $0 [--dry-run]" >&2
      exit 2
      ;;
  esac
done
case "${DRY_RUN,,}" in
  1|true|yes) DRY_RUN=1 ;;
  0|false|no) DRY_RUN=0 ;;
  *)
    echo "ERROR: DRY_RUN must be 0/1, false/true, or no/yes (got '$DRY_RUN')." >&2
    exit 2
    ;;
esac

# ---- config (override via env) -------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-pt}"                     # conda env with open3d/torch
N_WORKERS="${N_WORKERS-}"                        # default: one per GPU (or one CPU worker)
GPUS="${GPUS-}"                                  # comma/space-separated IDs; overrides CUDA_VISIBLE_DEVICES
SPLIT="${SPLIT:-train}"                          # train | val
ROOT_DIR="${ROOT_DIR:-scenefun3d}"
CLIP_DIR="${CLIP_DIR:-pipeline/step7_lift_3d/lift_output_organized/$SPLIT}"
SAVE_DIR="${SAVE_DIR:-scenefun3d/processed_sam2}"
SCRIPT="${SCRIPT:-dataset/preprocess_data_sam2.py}"
EXTRA_ARGS="${EXTRA_ARGS:-}"                     # any extra flags
LOG_DIR="${LOG_DIR:-dataset/preprocess_sam2_logs}"
LOCK_FILE="${LOCK_FILE:-$SAVE_DIR/.preprocess_sam2_${SPLIT}.lock}"
# --------------------------------------------------------------------------

cd "$REPO_ROOT"

# Prevent concurrent runs from writing the same split/output directory. A dry
# run intentionally skips the lock and makes no directories.
if [ "$DRY_RUN" -eq 0 ]; then
  command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required to prevent duplicate runs." >&2
    exit 1
  }
  mkdir -p "$(dirname "$LOCK_FILE")"
  exec 9<> "$LOCK_FILE"
  if ! flock -n 9; then
    holder=""
    IFS= read -r holder < "$LOCK_FILE" || true
    echo "ERROR: preprocessing is already running for split '$SPLIT' and save dir '$SAVE_DIR'${holder:+ (PID $holder)}." >&2
    echo "Lock: $LOCK_FILE" >&2
    exit 1
  fi
  : > "$LOCK_FILE"
  printf '%s\n' "$$" >&9
fi

# Activate the conda env. Conda's activate/deactivate hook scripts (e.g. the
# gxx_linux-64 ones in pt) reference unset CONDA_BACKUP_* vars, so suspend
# nounset around activation.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
export PYTHONNOUSERSITE=1                        # keep the env's numpy/scipy off user-site

[ -f "$SCRIPT" ] || { echo "ERROR: script not found: $SCRIPT (in $REPO_ROOT)" >&2; exit 1; }
[ -d "$CLIP_DIR" ] || { echo "ERROR: input dir not found: $CLIP_DIR (in $REPO_ROOT)" >&2; exit 1; }

# Resolve the GPU list in this order:
#   1. GPUS, when non-empty
#   2. an inherited CUDA_VISIBLE_DEVICES (including an explicitly empty value)
#   3. all GPUs reported by nvidia-smi
# Both "0,2" and "0 2" are accepted. Each worker receives one entry, so its
# assigned physical GPU is exposed inside Python as logical device cuda:0.
if [ -n "$GPUS" ]; then
  GPU_SPEC="$GPUS"
elif [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
  GPU_SPEC="$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU_SPEC="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)"
else
  GPU_SPEC=""
fi

# CUDA_VISIBLE_DEVICES=-1 is the conventional way to disable CUDA.
if [ "$GPU_SPEC" = "-1" ]; then
  GPU_SPEC=""
fi
GPU_SPEC="${GPU_SPEC//,/ }"
GPU_SPEC="${GPU_SPEC//$'\n'/ }"
GPU_SPEC="${GPU_SPEC//$'\r'/ }"
read -r -a GPU_ARR <<< "$GPU_SPEC"
NUM_GPUS="${#GPU_ARR[@]}"

# Use one worker per selected GPU unless explicitly overridden. Keep a single
# worker for CPU-only operation.
if [ -z "$N_WORKERS" ]; then
  if [ "$NUM_GPUS" -gt 0 ]; then
    N_WORKERS="$NUM_GPUS"
  else
    N_WORKERS=1
  fi
fi
if ! [[ "$N_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: N_WORKERS must be a positive integer (got '$N_WORKERS')." >&2
  exit 1
fi

echo "env=$ENV_NAME  split=$SPLIT  n_workers=$N_WORKERS  gpus=[${GPU_ARR[*]:-cpu}]"
echo "python=$(which python)"
echo "logs -> $LOG_DIR/preprocess_${SPLIT}_shard<i>.log"

if [ "$DRY_RUN" -eq 1 ]; then
  echo "Dry run: no directories, logs, outputs, or workers will be created."
  for ((shard = 0; shard < N_WORKERS; shard++)); do
    if [ "$NUM_GPUS" -gt 0 ]; then
      gpu="${GPU_ARR[$((shard % NUM_GPUS))]}"
    else
      gpu=""
    fi
    log="$LOG_DIR/preprocess_${SPLIT}_shard${shard}.log"
    printf '[dry-run] shard %d/%d: CUDA_VISIBLE_DEVICES=%q ' "$shard" "$N_WORKERS" "$gpu"
    printf '%q ' python "$SCRIPT" \
      --split "$SPLIT" --root_dir "$ROOT_DIR" \
      --clip_dir "$CLIP_DIR" --save_dir "$SAVE_DIR" \
      --num_shards "$N_WORKERS" --shard "$shard"
    # Match the word splitting used by the real launch below.
    # shellcheck disable=SC2086
    for extra_arg in $EXTRA_ARGS; do
      printf '%q ' "$extra_arg"
    done
    printf '> %q 2>&1\n' "$log"
  done
  exit 0
fi

mkdir -p "$LOG_DIR"

terminate_workers() {
  local exit_code="$1"
  local -a running_pids=()
  trap - INT TERM HUP
  mapfile -t running_pids < <(jobs -pr)
  if [ "${#running_pids[@]}" -gt 0 ]; then
    echo "Stopping ${#running_pids[@]} worker(s)..." >&2
    kill "${running_pids[@]}" 2>/dev/null || true
    wait "${running_pids[@]}" 2>/dev/null || true
  fi
  exit "$exit_code"
}
trap 'terminate_workers 130' INT
trap 'terminate_workers 143' TERM
trap 'terminate_workers 129' HUP

pids=()
for ((shard = 0; shard < N_WORKERS; shard++)); do
  if [ "$NUM_GPUS" -gt 0 ]; then
    gpu="${GPU_ARR[$((shard % NUM_GPUS))]}"
  else
    gpu=""
  fi
  log="$LOG_DIR/preprocess_${SPLIT}_shard${shard}.log"
  echo "[*] shard $shard/$N_WORKERS on GPU '${gpu:-cpu}' -> $log"
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
    --split "$SPLIT" --root_dir "$ROOT_DIR" \
    --clip_dir "$CLIP_DIR" --save_dir "$SAVE_DIR" \
    --num_shards "$N_WORKERS" --shard "$shard" \
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
  echo "Some shards failed -- check $LOG_DIR, then just re-run to resume (done descs are skipped)." >&2
  exit 1
fi
