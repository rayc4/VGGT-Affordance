#!/usr/bin/env bash
# Run the prediction-only v3e VGGT/3D anchor-consensus selector, one shard per
# selected GPU. PROPOSAL_ROOT and OUTPUT_ROOT are split-specific directories.
set -euo pipefail

usage() {
  cat >&2 <<'EOF'
Usage: dataset/run_v3e_consensus_parallel.sh [--dry-run]

Environment overrides:
  SPLIT          train or val (default: val)
  GPUS           comma/space-separated GPU IDs (default: all visible GPUs)
  N_WORKERS      workers to launch (default: one per selected GPU)
  ENV_NAME       preprocessing conda environment (default: tasa)
  DATA_ROOT      SceneFun3D root (default: scenefun3d)
  PROPOSAL_ROOT  split-specific v3e proposal directory
  OUTPUT_ROOT    split-specific selected-anchor directory
  EXTRA_ARGS     model/threshold arguments for the selector
  LOG_DIR        per-shard log directory
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
ENV_NAME="${ENV_NAME:-tasa}"
SPLIT="${SPLIT:-val}"
GPUS="${GPUS-}"
N_WORKERS="${N_WORKERS-}"
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
PROPOSAL_ROOT="${PROPOSAL_ROOT:-scenefun3d/preprocessing_experiments/direct_vggt_v3e/proposals/$SPLIT}"
OUTPUT_ROOT="${OUTPUT_ROOT:-scenefun3d/preprocessing_experiments/direct_vggt_v3e/consensus/$SPLIT}"
EXTRA_ARGS="${EXTRA_ARGS:---local-files-only --max-proposals 6 --max-anchors 3 --cluster-radius 0.12 --min-video-support 2 --context-frames 8 --context-window-seconds 3.0}"
LOG_DIR="${LOG_DIR:-scenefun3d/preprocessing_experiments/direct_vggt_v3e/logs/consensus/$SPLIT}"
LOCK_FILE="${LOCK_FILE:-$OUTPUT_ROOT/.v3e_consensus.lock}"
MODULE="pipeline.step2_vggt_anchor_consensus.select"
MODULE_FILE="$REPO_ROOT/pipeline/step2_vggt_anchor_consensus/select.py"

case "$SPLIT" in
  train|val) ;;
  *) echo "ERROR: SPLIT must be train or val (got '$SPLIT')." >&2; exit 2 ;;
esac

OFFLINE_MODE=0
# EXTRA_ARGS intentionally follows the repository's whitespace-splitting
# convention. Paths belong in the named environment variables above.
# shellcheck disable=SC2086
for extra_arg in $EXTRA_ARGS; do
  case "$extra_arg" in
    --local-files-only) OFFLINE_MODE=1 ;;
    --data-root|--data-root=*|--split|--split=*|\
    --proposal-root|--proposal-root=*|--output-root|--output-root=*|\
    --device|--device=*|--num-shards|--num-shards=*|--shard|--shard=*|\
    --dry-run|-h|--help)
      echo "ERROR: '$extra_arg' is runner-controlled and cannot appear in EXTRA_ARGS." >&2
      exit 2
      ;;
  esac
done

cd "$REPO_ROOT"
command -v realpath >/dev/null 2>&1 || {
  echo "ERROR: realpath is required for output safety checks." >&2
  exit 1
}
RESOLVED_PROPOSALS="$(realpath -m "$PROPOSAL_ROOT")"
RESOLVED_OUTPUT="$(realpath -m "$OUTPUT_ROOT")"
case "$RESOLVED_PROPOSALS:$RESOLVED_OUTPUT" in
  "$RESOLVED_PROPOSALS:$RESOLVED_PROPOSALS"|\
  "$RESOLVED_PROPOSALS:$RESOLVED_PROPOSALS"/*|\
  "$RESOLVED_OUTPUT"/*:"$RESOLVED_OUTPUT")
    echo "ERROR: proposal and consensus roots must be separate, non-nested directories." >&2
    exit 2
    ;;
esac

PROTECTED_ROOTS=(
  "$DATA_ROOT/processed_sam2"
  "pipeline/step1_affordance/affordance_result"
  "pipeline/step2_clipwithaffordance/clipwithaffordance_output"
  "pipeline/step3_point_prediction/point_clipwithaffordance_output"
  "pipeline/step4_crop_images/seg_image_output"
  "pipeline/step5_molmo_sam/molmo_output"
  "pipeline/step6_molmo_merge/molmo_merge_output"
)
for protected in "${PROTECTED_ROOTS[@]}"; do
  resolved_protected="$(realpath -m "$protected")"
  case "$RESOLVED_OUTPUT" in
    "$resolved_protected"|"$resolved_protected"/*)
      echo "ERROR: OUTPUT_ROOT must not overwrite an original pipeline tree: $protected" >&2
      exit 2
      ;;
  esac
  case "$resolved_protected" in
    "$RESOLVED_OUTPUT"/*)
      echo "ERROR: OUTPUT_ROOT is too broad and contains an original pipeline tree: $protected" >&2
      exit 2
      ;;
  esac
done

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
if [ "$GPU_SPEC" = "-1" ]; then GPU_SPEC=""; fi
GPU_SPEC="${GPU_SPEC//,/ }"
GPU_SPEC="${GPU_SPEC//$'\n'/ }"
GPU_SPEC="${GPU_SPEC//$'\r'/ }"
read -r -a GPU_ARR <<< "$GPU_SPEC"
NUM_GPUS="${#GPU_ARR[@]}"

declare -A SEEN_GPUS=()
for gpu in "${GPU_ARR[@]}"; do
  if ! [[ "$gpu" =~ ^[0-9]+$ || "$gpu" =~ ^GPU-[[:alnum:]-]+$ || "$gpu" =~ ^MIG-[[:alnum:]_./-]+$ ]]; then
    echo "ERROR: invalid GPU selection '$gpu'." >&2
    exit 2
  fi
  if [ "${SEEN_GPUS[$gpu]+present}" = present ]; then
    echo "ERROR: duplicate GPU selection '$gpu'." >&2
    exit 2
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
  exit 2
fi
if [ "$NUM_GPUS" -gt 0 ] && [ "$N_WORKERS" -gt "$NUM_GPUS" ]; then
  echo "ERROR: N_WORKERS=$N_WORKERS exceeds selected GPUs=$NUM_GPUS; use one model worker per GPU." >&2
  exit 2
fi
if [ "$DRY_RUN" -eq 0 ] && [ "$NUM_GPUS" -eq 0 ]; then
  echo "ERROR: v3e consensus requires CUDA; set GPUS or CUDA_VISIBLE_DEVICES." >&2
  exit 1
fi

echo "env=$ENV_NAME split=$SPLIT workers=$N_WORKERS gpus=[${GPU_ARR[*]:-none}]"
echo "proposals=$PROPOSAL_ROOT"
echo "consensus=$OUTPUT_ROOT"
echo "logs=$LOG_DIR/v3e_consensus_${SPLIT}_shard<i>.log"

print_command() {
  local shard="$1" gpu="$2" log="$3"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
  printf '%q ' python -m "$MODULE"
  # shellcheck disable=SC2086
  for extra_arg in $EXTRA_ARGS; do printf '%q ' "$extra_arg"; done
  printf '%q ' \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --proposal-root "$PROPOSAL_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda:0 \
    --num-shards "$N_WORKERS" \
    --shard "$shard"
  printf '> %q 2>&1\n' "$log"
}

if [ "$DRY_RUN" -eq 1 ]; then
  for ((shard=0; shard<N_WORKERS; shard++)); do
    gpu=""
    if [ "$NUM_GPUS" -gt 0 ]; then
      gpu="${GPU_ARR[$shard]}"
    fi
    print_command "$shard" "$gpu" "$LOG_DIR/v3e_consensus_${SPLIT}_shard${shard}.log"
  done
  exit 0
fi

[ -f "$MODULE_FILE" ] || { echo "ERROR: module not found: $MODULE_FILE" >&2; exit 1; }
[ -d "$DATA_ROOT" ] || { echo "ERROR: data root not found: $DATA_ROOT" >&2; exit 1; }
[ -d "$PROPOSAL_ROOT" ] || { echo "ERROR: proposal root not found: $PROPOSAL_ROOT" >&2; exit 1; }
command -v conda >/dev/null 2>&1 || { echo "ERROR: conda is required." >&2; exit 1; }
command -v flock >/dev/null 2>&1 || { echo "ERROR: flock is required." >&2; exit 1; }

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

mkdir -p "$(dirname "$LOCK_FILE")"
exec 9<> "$LOCK_FILE"
if ! flock -n 9; then
  echo "ERROR: v3e consensus already holds $LOCK_FILE" >&2
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
for ((shard=0; shard<N_WORKERS; shard++)); do
  gpu="${GPU_ARR[$shard]}"
  log="$LOG_DIR/v3e_consensus_${SPLIT}_shard${shard}.log"
  echo "[*] shard $shard/$N_WORKERS on GPU $gpu -> $log"
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpu" python -m "$MODULE" \
    $EXTRA_ARGS \
    --data-root "$DATA_ROOT" \
    --split "$SPLIT" \
    --proposal-root "$PROPOSAL_ROOT" \
    --output-root "$OUTPUT_ROOT" \
    --device cuda:0 \
    --num-shards "$N_WORKERS" \
    --shard "$shard" \
    > "$log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do wait "$pid" || failed=1; done
if [ "$failed" -ne 0 ]; then
  echo "ERROR: one or more v3e consensus shards failed; inspect $LOG_DIR and rerun to resume." >&2
  exit 1
fi
echo "All $SPLIT v3e consensus shards finished."
