#!/usr/bin/env bash
# Calibrate probability thresholds for the base and frozen VGGT checkpoints.

set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  pipeline/step8_3d_training_vggt/run_threshold_sweep_matrix.sh [--dry-run]

Environment:
  GPUS             Comma/space-separated physical GPU IDs (default: 0,1,2,3)
  ENV_NAME         Conda environment (default: pt)
  OUTPUT_ROOT      Result root (default: outputs/2026-07-30_threshold_sweep)
  THRESHOLDS       Comma list or START:END:STEP (default: 0.20:0.80:0.025)
  BATCH_SIZE       Per-GPU evaluation batch (default: 32)
  NUM_WORKERS      DataLoader workers per process (default: 2)
  PROCESSED_SAM2_DIR Processed data tree (default: scenefun3d/processed_sam2)
  VGGT_FEAT_ROOT   VGGT feature tree (default: <processed>/vggt_features)
EOF
}

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
    shift
elif [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi
if [[ $# -ne 0 ]]; then
    usage >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVALUATOR="$REPO_ROOT/scripts/eval_threshold_sweep.py"
SUMMARIZER="$REPO_ROOT/scripts/summarize_threshold_sweeps.py"
ENV_NAME="${ENV_NAME:-pt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$REPO_ROOT/outputs/2026-07-30_threshold_sweep}"
THRESHOLDS="${THRESHOLDS:-0.20:0.80:0.025}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-2}"
PROCESSED_SAM2_DIR="${PROCESSED_SAM2_DIR:-$REPO_ROOT/scenefun3d/processed_sam2}"
VGGT_FEAT_ROOT="${VGGT_FEAT_ROOT:-$PROCESSED_SAM2_DIR/vggt_features}"
GPU_SPEC="${GPUS:-0,1,2,3}"
GPU_SPEC="${GPU_SPEC//,/ }"
read -r -a GPU_IDS <<< "$GPU_SPEC"

if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
    echo 'ERROR: no GPUs selected.' >&2
    exit 1
fi
for value_name in BATCH_SIZE; do
    value="${!value_name}"
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || {
        echo "ERROR: $value_name must be a positive integer." >&2
        exit 1
    }
done
[[ "$NUM_WORKERS" =~ ^[0-9]+$ ]] || {
    echo 'ERROR: NUM_WORKERS must be a non-negative integer.' >&2
    exit 1
}

JOBS=(
  "base_best_map|base|outputs/2026-07-29_20-29-16_base/ckpt/mask_refinement_model003480.pt|"
  "base_best_miou|base|outputs/2026-07-29_20-29-16_base/ckpt/mask_refinement_model004350.pt|"
  "uniform_best_map|vggt|outputs/2026-07-29_22-32-34_uniform-vggt-conf-frozen/ckpt/mask_refinement_model001595.pt|vggt_feat_uniform.npy"
  "uniform_best_miou|vggt|outputs/2026-07-29_22-32-34_uniform-vggt-conf-frozen/ckpt/mask_refinement_model005220.pt|vggt_feat_uniform.npy"
  "weighted_best_map|vggt|outputs/2026-07-30_15-31-58_weighted-vggt-conf-frozen/ckpt/mask_refinement_model001595.pt|vggt_feat.npy"
  "weighted_best_miou|vggt|outputs/2026-07-30_15-31-58_weighted-vggt-conf-frozen/ckpt/mask_refinement_model005800.pt|vggt_feat.npy"
)

cd "$REPO_ROOT"
for job in "${JOBS[@]}"; do
    IFS='|' read -r label kind checkpoint feature_name <<< "$job"
    [[ -f "$checkpoint" ]] || {
        echo "ERROR: checkpoint not found: $checkpoint" >&2
        exit 1
    }
done

build_command() {
    local job="$1"
    local -n destination="$2"
    local label kind checkpoint feature_name
    IFS='|' read -r label kind checkpoint feature_name <<< "$job"
    destination=(
        conda run --no-capture-output -n "$ENV_NAME" python "$EVALUATOR"
        --kind "$kind"
        --label "$label"
        --checkpoint "$checkpoint"
        --output-dir "$OUTPUT_ROOT/$label"
        --thresholds "$THRESHOLDS"
        --device cuda:0
        --batch-size "$BATCH_SIZE"
        --num-workers "$NUM_WORKERS"
        --expected-frames 5619
        --processed-sam2-dir "$PROCESSED_SAM2_DIR"
        --vggt-feature-root "$VGGT_FEAT_ROOT"
    )
    if [[ -n "$feature_name" ]]; then
        destination+=(--feature-name "$feature_name")
    fi
}

if [[ "$DRY_RUN" -eq 1 ]]; then
    for index in "${!JOBS[@]}"; do
        gpu="${GPU_IDS[$((index % ${#GPU_IDS[@]}))]}"
        command=()
        build_command "${JOBS[$index]}" command
        printf 'CUDA_VISIBLE_DEVICES=%q ' "$gpu"
        printf '%q ' "${command[@]}"
        printf '\n'
    done
    exit 0
fi

mkdir -p "$OUTPUT_ROOT/logs"
worker() {
    local worker_index="$1"
    local gpu="$2"
    local index label kind checkpoint feature_name log_path
    local -a command
    for ((index=worker_index; index<${#JOBS[@]}; index+=${#GPU_IDS[@]})); do
        IFS='|' read -r label kind checkpoint feature_name <<< "${JOBS[$index]}"
        log_path="$OUTPUT_ROOT/logs/$label.log"
        command=()
        build_command "${JOBS[$index]}" command
        echo "[GPU $gpu] start $label"
        CUDA_VISIBLE_DEVICES="$gpu" PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
            "${command[@]}" >"$log_path" 2>&1
        echo "[GPU $gpu] complete $label (log: $log_path)"
    done
}

pids=()
for worker_index in "${!GPU_IDS[@]}"; do
    worker "$worker_index" "${GPU_IDS[$worker_index]}" &
    pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done
if [[ "$status" -ne 0 ]]; then
    echo "ERROR: one or more threshold sweeps failed; inspect $OUTPUT_ROOT/logs." >&2
    exit "$status"
fi

conda run --no-capture-output -n "$ENV_NAME" python "$SUMMARIZER" "$OUTPUT_ROOT" \
    | tee "$OUTPUT_ROOT/summary.txt"
