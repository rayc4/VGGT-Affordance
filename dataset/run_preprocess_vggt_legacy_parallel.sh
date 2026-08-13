#!/usr/bin/env bash
# Parallel VGGT-track preprocessing that writes the original processed_sam2 format.
#
# Usage:
#   SPLIT=val GPUS=0,1,2,3 dataset/run_preprocess_vggt_legacy_parallel.sh
#   SPLIT=train GPUS=0,1,2,3 dataset/run_preprocess_vggt_legacy_parallel.sh
#   SPLIT=val NO_POINT_ROOT=1 ANCHOR_ROOT=path/to/direct/val \
#     dataset/run_preprocess_vggt_legacy_parallel.sh
#   dataset/run_preprocess_vggt_legacy_parallel.sh --dry-run
# Set ALLOW_COPY_FALLBACK=1 only when the output filesystem cannot hardlink and
# the larger independent-file layout is explicitly acceptable.
# Set ALLOW_EMPTY_OUTPUT_SHARDS=1 only when a later aggregate audit is the
# usability gate, as in strict standalone generation with expected abstentions.
set -euo pipefail

DRY_RUN="${DRY_RUN:-0}"
ALLOW_COPY_FALLBACK="${ALLOW_COPY_FALLBACK:-0}"
ALLOW_EMPTY_OUTPUT_SHARDS="${ALLOW_EMPTY_OUTPUT_SHARDS:-0}"
NO_POINT_ROOT="${NO_POINT_ROOT:-0}"
for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    -h|--help)
      echo "Usage: $0 [--dry-run]"
      exit 0
      ;;
    *)
      echo "ERROR: unknown option: $arg" >&2
      exit 2
      ;;
  esac
done

case "${DRY_RUN,,}" in
  1|true|yes) DRY_RUN=1 ;;
  0|false|no) DRY_RUN=0 ;;
  *) echo "ERROR: DRY_RUN must be boolean (got '$DRY_RUN')." >&2; exit 2 ;;
esac
case "${ALLOW_COPY_FALLBACK,,}" in
  1|true|yes) ALLOW_COPY_FALLBACK=1 ;;
  0|false|no) ALLOW_COPY_FALLBACK=0 ;;
  *) echo "ERROR: ALLOW_COPY_FALLBACK must be boolean (got '$ALLOW_COPY_FALLBACK')." >&2; exit 2 ;;
esac
case "${ALLOW_EMPTY_OUTPUT_SHARDS,,}" in
  1|true|yes) ALLOW_EMPTY_OUTPUT_SHARDS=1 ;;
  0|false|no) ALLOW_EMPTY_OUTPUT_SHARDS=0 ;;
  *) echo "ERROR: ALLOW_EMPTY_OUTPUT_SHARDS must be boolean (got '$ALLOW_EMPTY_OUTPUT_SHARDS')." >&2; exit 2 ;;
esac
case "${NO_POINT_ROOT,,}" in
  1|true|yes) NO_POINT_ROOT=1 ;;
  0|false|no) NO_POINT_ROOT=0 ;;
  *) echo "ERROR: NO_POINT_ROOT must be boolean (got '$NO_POINT_ROOT')." >&2; exit 2 ;;
esac

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ENV_NAME="${ENV_NAME:-tasa}"
N_WORKERS="${N_WORKERS-}"
GPUS="${GPUS-}"
SPLIT="${SPLIT:-val}"
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
ANCHOR_ROOT="${ANCHOR_ROOT:-pipeline/step6_molmo_merge/molmo_merge_output/$SPLIT}"
POINT_ROOT="${POINT_ROOT:-pipeline/step3_point_prediction/point_clipwithaffordance_output}"
REPAIR_PLAN="${REPAIR_PLAN:-}"
SAVE_DIR="${SAVE_DIR:-scenefun3d/processed_sam2_vggt_tracks_v1}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
LOG_DIR="${LOG_DIR:-dataset/preprocess_vggt_legacy_logs}"
LOCK_FILE="${LOCK_FILE:-$SAVE_DIR/.preprocess_vggt_legacy_${SPLIT}.lock}"
MODULE="pipeline.step4_vggt_legacy.preprocess"
STORAGE_ARGS=()
EXECUTION_ARGS=()

# Match the intentional EXTRA_ARGS word splitting used for worker commands.
OFFLINE_MODE=0
# shellcheck disable=SC2086
for extra_arg in $EXTRA_ARGS; do
  case "$extra_arg" in
    --local-files-only)
      OFFLINE_MODE=1
      ;;
    --dry-run|-h|--help|--no-hardlink-dedup|--require-hardlink-dedup|\
    --allow-empty-output-shard|--allow-empty-output-shard=*|\
    --point-root|--point-root=*|--no-point-root|--no-point-root=*|\
    --repair-plan|--repair-plan=*)
      echo "ERROR: '$extra_arg' is runner-controlled and cannot appear in EXTRA_ARGS." >&2
      echo "Use --dry-run/DRY_RUN, ALLOW_COPY_FALLBACK, or NO_POINT_ROOT on the runner itself." >&2
      exit 2
      ;;
  esac
done

if [ "$NO_POINT_ROOT" -eq 1 ]; then
  POINT_ARGS=(--no-point-root)
else
  POINT_ARGS=(--point-root "$POINT_ROOT")
fi

REPAIR_ARGS=()
if [ -n "$REPAIR_PLAN" ]; then
  REPAIR_ARGS=(--repair-plan "$REPAIR_PLAN")
fi

if [ "$ALLOW_COPY_FALLBACK" -eq 0 ]; then
  STORAGE_ARGS+=(--require-hardlink-dedup)
fi
if [ "$ALLOW_EMPTY_OUTPUT_SHARDS" -eq 1 ]; then
  EXECUTION_ARGS+=(--allow-empty-output-shard)
fi

if [ "$SPLIT" != "train" ] && [ "$SPLIT" != "val" ]; then
  echo "ERROR: SPLIT must be train or val (got '$SPLIT')." >&2
  exit 2
fi

cd "$REPO_ROOT"

command -v realpath >/dev/null 2>&1 || {
  echo "ERROR: realpath is required for output safety checks." >&2
  exit 1
}
RESOLVED_SAVE_DIR="$(realpath -m "$SAVE_DIR")"
RESOLVED_ORIGINAL_DIR="$(realpath -m "$DATA_ROOT/processed_sam2")"
case "$RESOLVED_SAVE_DIR" in
  "$RESOLVED_ORIGINAL_DIR"|"$RESOLVED_ORIGINAL_DIR"/*)
    echo "ERROR: SAVE_DIR must not be the original processed_sam2 tree or a descendant." >&2
    exit 2
    ;;
esac

set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u
export PYTHONNOUSERSITE=1
if [ "$OFFLINE_MODE" -eq 1 ]; then
  export HF_HUB_OFFLINE=1
  export TRANSFORMERS_OFFLINE=1
fi

[ -d "$DATA_ROOT" ] || { echo "ERROR: data root not found: $DATA_ROOT" >&2; exit 1; }
if [ "$DRY_RUN" -eq 0 ] && [ ! -d "$ANCHOR_ROOT" ]; then
  echo "ERROR: anchor root not found: $ANCHOR_ROOT" >&2
  exit 1
fi
if [ "$DRY_RUN" -eq 0 ] && [ -n "$REPAIR_PLAN" ] && [ ! -f "$REPAIR_PLAN" ]; then
  echo "ERROR: repair plan not found: $REPAIR_PLAN" >&2
  exit 1
fi

if [ "$DRY_RUN" -eq 0 ] && [ "$ALLOW_COPY_FALLBACK" -eq 0 ]; then
  if ! python - "$SAVE_DIR" <<'PY'
import os
import sys
import tempfile
from pathlib import Path

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
try:
    with tempfile.TemporaryDirectory(prefix=".vggt-hardlink-check-", dir=root) as raw:
        probe = Path(raw)
        source = probe / "source"
        source.touch()
        os.link(source, probe / "link")
except OSError as exc:
    print(f"hardlink probe failed in {root}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
  then
    echo "ERROR: SAVE_DIR does not support hardlink deduplication." >&2
    echo "Set ALLOW_COPY_FALLBACK=1 only if independent repeated files are acceptable." >&2
    exit 1
  fi
fi

if [ -n "$GPUS" ]; then
  GPU_SPEC="$GPUS"
elif [ "${CUDA_VISIBLE_DEVICES+x}" = x ]; then
  GPU_SPEC="$CUDA_VISIBLE_DEVICES"
elif command -v nvidia-smi >/dev/null 2>&1; then
  if ! GPU_SPEC="$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null)"; then
    GPU_SPEC=""
  fi
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
declare -A SEEN_GPUS=()
for gpu in "${GPU_ARR[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ || "$gpu" =~ ^GPU-[[:alnum:]-]+$ || "$gpu" =~ ^MIG-[[:alnum:]_./-]+$ ]]; then
    echo "ERROR: invalid GPU selection '$gpu'; use CUDA indices or GPU/MIG UUIDs." >&2
    exit 1
  fi
  if [ "${SEEN_GPUS[$gpu]+present}" = present ]; then
    echo "ERROR: duplicate GPU selection '$gpu' would oversubscribe one device." >&2
    exit 1
  fi
  SEEN_GPUS["$gpu"]=1
done

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
if [ "$NUM_GPUS" -gt 0 ] && [ "$N_WORKERS" -gt "$NUM_GPUS" ]; then
  echo "ERROR: N_WORKERS=$N_WORKERS exceeds selected GPUs=$NUM_GPUS; each worker loads VGGT-1B and SAM-H." >&2
  exit 1
fi
if [ "$DRY_RUN" -eq 0 ] && [ "$NUM_GPUS" -eq 0 ]; then
  echo "ERROR: quality preprocessing requires at least one selected CUDA GPU; set GPUS or CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

echo "env=$ENV_NAME split=$SPLIT workers=$N_WORKERS gpus=[${GPU_ARR[*]:-cpu}]"
echo "output=$SAVE_DIR"
echo "shared_storage=$([ "$ALLOW_COPY_FALLBACK" -eq 0 ] && echo required-hardlinks || echo hardlinks-with-copy-fallback)"
echo "logs=$LOG_DIR/preprocess_${SPLIT}_shard<i>.log"

print_command() {
  local shard="$1"
  local gpu="$2"
  local log="$3"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
  printf '%q ' python -m "$MODULE"
  # Match the intentional word splitting used for EXTRA_ARGS below.
  # shellcheck disable=SC2086
  for extra_arg in $EXTRA_ARGS; do
    printf '%q ' "$extra_arg"
  done
  printf '%q ' --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --anchor-root "$ANCHOR_ROOT" \
    "${POINT_ARGS[@]}" \
    "${REPAIR_ARGS[@]}" \
    --output-root "$SAVE_DIR" \
    --num-shards "$N_WORKERS" \
    --shard "$shard" \
    "${STORAGE_ARGS[@]}" \
    "${EXECUTION_ARGS[@]}"
  printf '> %q 2>&1\n' "$log"
}

if [ "$DRY_RUN" -eq 1 ]; then
  for ((shard = 0; shard < N_WORKERS; shard++)); do
    if [ "$NUM_GPUS" -gt 0 ]; then
      gpu="${GPU_ARR[$((shard % NUM_GPUS))]}"
    else
      gpu=""
    fi
    print_command "$shard" "$gpu" "$LOG_DIR/preprocess_${SPLIT}_shard${shard}.log"
  done
  exit 0
fi

command -v flock >/dev/null 2>&1 || {
  echo "ERROR: flock is required." >&2
  exit 1
}
mkdir -p "$(dirname "$LOCK_FILE")"
exec 9<> "$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: preprocessing already holds $LOCK_FILE" >&2
  exit 1
fi
: > "$LOCK_FILE"
printf '%s\n' "$$" >&9

mkdir -p "$LOG_DIR"

terminate_workers() {
  local exit_code="$1"
  local -a running_pids=()
  trap - INT TERM HUP
  mapfile -t running_pids < <(jobs -pr)
  if [ "${#running_pids[@]}" -gt 0 ]; then
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
  echo "[*] shard $shard/$N_WORKERS on ${gpu:-cpu} -> $log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" python -m "$MODULE" \
    $EXTRA_ARGS \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --anchor-root "$ANCHOR_ROOT" \
    "${POINT_ARGS[@]}" \
    "${REPAIR_ARGS[@]}" \
    --output-root "$SAVE_DIR" \
    --num-shards "$N_WORKERS" \
    --shard "$shard" \
    "${STORAGE_ARGS[@]}" \
    "${EXECUTION_ARGS[@]}" \
    > "$log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  wait "$pid" || failed=1
done
if [ "$failed" -ne 0 ]; then
  echo "ERROR: one or more shards failed; inspect $LOG_DIR and rerun to resume." >&2
  exit 1
fi
echo "All $SPLIT shards finished."
