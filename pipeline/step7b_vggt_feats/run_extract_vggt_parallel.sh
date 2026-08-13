#!/usr/bin/env bash
#
# Parallel Step 7b VGGT feature extraction.
#
# Runs one persistent worker per selected GPU. Each worker loads VGGT once and
# processes a deterministic, disjoint shard of visits. Re-running is safe:
# Per-visit scan features are cached and complete per-sample
# feature/confidence/view-count bundles are skipped unless EXTRA_ARGS includes
# --overwrite/--overwrite_cache. Files are written to a separate mirrored tree,
# never into train/val inputs.
#
# Usage:
#   pipeline/step7b_vggt_feats/run_extract_vggt_parallel.sh <train|val> [--dry-run]
#
# Examples:
#   GPUS="0,1,2,3" pipeline/step7b_vggt_feats/run_extract_vggt_parallel.sh train
#   GPUS="0 2" EXTRA_ARGS="--chunk_size 16 --context_stride 2" \
#     pipeline/step7b_vggt_feats/run_extract_vggt_parallel.sh val
#
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage:
  pipeline/step7b_vggt_feats/run_extract_vggt_parallel.sh <train|val> [--dry-run]

Environment overrides:
  GPUS              Comma/space-separated GPU IDs (default: CUDA_VISIBLE_DEVICES or all GPUs)
  N_WORKERS         Number of workers (default: one per selected GPU; cannot exceed GPU count)
  ENV_NAME          Conda environment (default: tasa)
  DATA_ROOT         SceneFun3D data root (default: scenefun3d)
  PROCESSED_DIR     Processed frame tree (default: data/processed_sam2/<split>)
  FEAT_OUT_ROOT     Split-specific mirror output root for VGGT feature/reliability arrays
                    (default: <processed_sam2>/vggt_features/<split>)
  CACHE_DIR         Per-visit VGGT cache directory (default: outputs/vggt_feat_cache)
  LOG_DIR           Worker log directory
  EXTRA_ARGS        Additional extractor flags, e.g.
                    "--chunk_size 16 --context_stride 2 --confidence_head depth"
EOF
}

if [ "$#" -lt 1 ] || [ "$#" -gt 2 ]; then
  usage
  exit 2
fi
if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

SPLIT="$1"
case "$SPLIT" in
  train|val) ;;
  *)
    echo "ERROR: invalid split '$SPLIT' (expected train or val)." >&2
    exit 2
    ;;
esac

DRY_RUN=0
if [ "$#" -eq 2 ]; then
  if [ "$2" != "--dry-run" ]; then
    echo "ERROR: unknown option '$2'." >&2
    usage
    exit 2
  fi
  DRY_RUN=1
fi

# ---- config (override via env) -------------------------------------------
REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
ENV_NAME="${ENV_NAME:-tasa}"
GPUS="${GPUS-}"
N_WORKERS="${N_WORKERS-}"
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
PROCESSED_DIR="${PROCESSED_DIR:-scenefun3d/processed_sam2/$SPLIT}"
PROCESSED_PARENT="$(dirname "$PROCESSED_DIR")"
FEAT_OUT_ROOT="${FEAT_OUT_ROOT:-$PROCESSED_PARENT/vggt_features/$SPLIT}"
CACHE_DIR="${CACHE_DIR:-outputs/vggt_feat_cache}"
SCRIPT="${SCRIPT:-pipeline/step7b_vggt_feats/extract_vggt_features.py}"
LOG_DIR="${LOG_DIR:-pipeline/step7b_vggt_feats/logs/$SPLIT}"
EXTRA_ARGS="${EXTRA_ARGS-}"
OUTPUT_LOCK_ROOT="$FEAT_OUT_ROOT"
LOCK_FILE="${LOCK_FILE:-$OUTPUT_LOCK_ROOT/.extract_vggt_${SPLIT}.lock}"
# --------------------------------------------------------------------------

cd "$REPO_ROOT"

[ -f "$SCRIPT" ] || { echo "ERROR: extractor not found: $SCRIPT" >&2; exit 1; }
[ -d "$PROCESSED_DIR" ] || {
  echo "ERROR: processed input directory not found: $PROCESSED_DIR" >&2
  exit 1
}

# Resolve GPUs: GPUS > inherited CUDA_VISIBLE_DEVICES > nvidia-smi.
if [ -n "$GPUS" ]; then
  GPU_SPEC="$GPUS"
elif [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
  GPU_SPEC="$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  GPU_SPEC="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null || true)"
else
  GPU_SPEC=""
fi
if [ "$GPU_SPEC" = "-1" ]; then
  GPU_SPEC=""
fi
GPU_SPEC="${GPU_SPEC//,/ }"
GPU_SPEC="${GPU_SPEC//$'\n'/ }"
GPU_SPEC="${GPU_SPEC//$'\r'/ }"
read -r -a GPU_ARR <<< "$GPU_SPEC"
NUM_GPUS="${#GPU_ARR[@]}"
if [ "$NUM_GPUS" -eq 0 ]; then
  echo "ERROR: VGGT extraction requires at least one GPU; set GPUS or CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

if [ -z "$N_WORKERS" ]; then
  N_WORKERS="$NUM_GPUS"
fi
if ! [[ "$N_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERROR: N_WORKERS must be a positive integer (got '$N_WORKERS')." >&2
  exit 1
fi
if [ "$N_WORKERS" -gt "$NUM_GPUS" ]; then
  echo "ERROR: N_WORKERS=$N_WORKERS exceeds the $NUM_GPUS selected GPU(s)." >&2
  echo "One VGGT worker already fills a GPU; multiple workers per GPU are likely to OOM." >&2
  exit 1
fi

feat_out_args=(--feat_out_root "$FEAT_OUT_ROOT")

echo "env=$ENV_NAME  split=$SPLIT  workers=$N_WORKERS  gpus=[${GPU_ARR[*]}]"
echo "processed_dir=$PROCESSED_DIR"
echo "cache_dir=$CACHE_DIR"
echo "feature outputs=$FEAT_OUT_ROOT"
echo "logs=$LOG_DIR/vggt_${SPLIT}_shard<i>.log"

if [ "$DRY_RUN" -eq 1 ]; then
  for ((shard = 0; shard < N_WORKERS; shard++)); do
    gpu="${GPU_ARR[$shard]}"
    log="$LOG_DIR/vggt_${SPLIT}_shard${shard}.log"
    printf '[dry-run] shard %d/%d: CUDA_VISIBLE_DEVICES=%q ' "$shard" "$N_WORKERS" "$gpu"
    printf '%q ' python "$SCRIPT" \
      --data_root "$DATA_ROOT" --split "$SPLIT" \
      --processed_dir "$PROCESSED_DIR" --cache_dir "$CACHE_DIR" \
      --device cuda:0 --num_shards "$N_WORKERS" --shard "$shard" \
      "${feat_out_args[@]}"
    # Match the intentional word splitting used by the real launch.
    # shellcheck disable=SC2086
    for extra_arg in $EXTRA_ARGS; do
      printf '%q ' "$extra_arg"
    done
    printf '> %q 2>&1\n' "$log"
  done
  exit 0
fi

command -v conda >/dev/null 2>&1 || {
  echo "ERROR: conda is required to activate '$ENV_NAME'." >&2
  exit 1
}
command -v flock >/dev/null 2>&1 || {
  echo "ERROR: flock is required to prevent duplicate extraction runs." >&2
  exit 1
}

set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
export PYTHONNOUSERSITE=1

mkdir -p "$(dirname "$LOCK_FILE")"
exec 9<> "$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: VGGT extraction is already running for '$SPLIT' and '$OUTPUT_LOCK_ROOT'." >&2
  echo "Lock: $LOCK_FILE" >&2
  exit 1
fi
printf '%s\n' "$$" >&9

mkdir -p "$LOG_DIR" "$CACHE_DIR" "$FEAT_OUT_ROOT"
echo "python=$(which python)"

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
  gpu="${GPU_ARR[$shard]}"
  log="$LOG_DIR/vggt_${SPLIT}_shard${shard}.log"
  echo "[*] shard $shard/$N_WORKERS on GPU $gpu -> $log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" python "$SCRIPT" \
    --data_root "$DATA_ROOT" --split "$SPLIT" \
    --processed_dir "$PROCESSED_DIR" --cache_dir "$CACHE_DIR" \
    --device cuda:0 --num_shards "$N_WORKERS" --shard "$shard" \
    "${feat_out_args[@]}" \
    $EXTRA_ARGS \
    > "$log" 2>&1 &
  pids+=($!)
done

fail=0
for pid in "${pids[@]}"; do
  wait "$pid" || fail=1
done

if [ "$fail" -eq 0 ]; then
  echo "All VGGT extraction shards finished."
else
  echo "Some shards failed; inspect $LOG_DIR and re-run to resume completed work." >&2
  exit 1
fi
