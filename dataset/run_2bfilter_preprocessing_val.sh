#!/usr/bin/env bash
# Run the complete relational-filter preprocessing path for the val split.
#
# Defaults write to fresh *_2bfilter_val trees so the existing original and
# *_2bfilter validation outputs remain untouched. Every root is overridable.
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  dataset/run_2bfilter_preprocessing_val.sh [--dry-run] [--from STAGE]

Stages (in order):
  1  2  2b  3  4  5  6  6b  7  organize  process  7b

Examples:
  dataset/run_2bfilter_preprocessing_val.sh --dry-run
  dataset/run_2bfilter_preprocessing_val.sh
  dataset/run_2bfilter_preprocessing_val.sh --from 6b

Environment overrides:
  GPUS                Physical GPUs, comma- or space-separated
                      (default: "4 5 6 7")
  TASA_ENV            Steps 1-7 and 7b environment (default: tasa)
  VGGT_ENV            Step 6b environment (default: affordvggt)
  PROCESS_ENV         processed_sam2 environment (default: pt)
  DATA_ROOT           SceneFun3D root (default: scenefun3d)
  OUTPUT_TAG          Suffix for every default output tree
                      (default: 2bfilter_val)
  STEP1_ROOT          Step 1 output root
  STEP2_ROOT          Step 2 output root
  STEP2B_ROOT         Step 2b output root
  STEP3_ROOT          Step 3 output root
  STEP4_ROOT          Step 4 output root
  STEP5_ROOT          Step 5 output root
  STEP6_ROOT          Step 6 output root
  STEP6B_ROOT         Step 6b output root
  STEP7_ROOT          Step 7 output root
  ORGANIZED_ROOT      Organized Step 7 output root
  PROCESSED_ROOT      processed_sam2 output root
  FEATURE_ROOT        Step 7b feature output root
  FEATURE_CACHE_ROOT  Step 7b cache root
  STEP1_EXTRA_ARGS    Extra Step 1 arguments
  STEP2_EXTRA_ARGS    Extra Step 2 arguments
  STEP2B_EXTRA_ARGS   Extra Step 2b arguments
  STEP3_EXTRA_ARGS    Extra Step 3 arguments
  STEP5_EXTRA_ARGS    Extra Step 5 arguments
  STEP6B_EXTRA_ARGS   Extra Step 6b arguments
  STEP7_EXTRA_ARGS    Extra Step 7 arguments
  PROCESS_EXTRA_ARGS  Extra processed_sam2 arguments
  STEP7B_EXTRA_ARGS   Extra Step 7b arguments
EOF
}

DRY_RUN=0
FROM_STAGE="1"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --from)
      [[ $# -ge 2 ]] || {
        echo "ERROR: --from requires a stage." >&2
        usage >&2
        exit 2
      }
      FROM_STAGE="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

STAGES=(1 2 2b 3 4 5 6 6b 7 organize process 7b)
stage_index() {
  local wanted="$1"
  local index
  for index in "${!STAGES[@]}"; do
    if [[ "${STAGES[$index]}" == "$wanted" ]]; then
      printf '%s\n' "$index"
      return 0
    fi
  done
  return 1
}
FROM_INDEX="$(stage_index "$FROM_STAGE")" || {
  echo "ERROR: invalid --from stage: $FROM_STAGE" >&2
  usage >&2
  exit 2
}

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
GPUS="${GPUS:-4 5 6 7}"
TASA_ENV="${TASA_ENV:-tasa}"
VGGT_ENV="${VGGT_ENV:-affordvggt}"
PROCESS_ENV="${PROCESS_ENV:-pt}"
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
OUTPUT_TAG="${OUTPUT_TAG:-2bfilter_val}"
SPLIT=val

STEP1_ROOT="${STEP1_ROOT:-pipeline/step1_affordance/affordance_result_${OUTPUT_TAG}}"
STEP2_ROOT="${STEP2_ROOT:-pipeline/step2_clipwithaffordance/clipwithaffordance_output_${OUTPUT_TAG}}"
STEP2B_ROOT="${STEP2B_ROOT:-pipeline/step2b_relational_filter/clipwithaffordance_output_${OUTPUT_TAG}}"
STEP3_ROOT="${STEP3_ROOT:-pipeline/step3_point_prediction/point_clipwithaffordance_output_${OUTPUT_TAG}}"
STEP4_ROOT="${STEP4_ROOT:-pipeline/step4_crop_images/seg_image_output/point_clipwithaffordance_output_${OUTPUT_TAG}}"
STEP5_ROOT="${STEP5_ROOT:-pipeline/step5_molmo_sam/molmo_output_${OUTPUT_TAG}}"
STEP6_ROOT="${STEP6_ROOT:-pipeline/step6_molmo_merge/molmo_merge_output_${OUTPUT_TAG}}"
STEP6B_ROOT="${STEP6B_ROOT:-pipeline/step6b_vggt_track/vggt_track_output_${OUTPUT_TAG}}"
STEP7_ROOT="${STEP7_ROOT:-pipeline/step7_lift_3d/lift_output_${OUTPUT_TAG}}"
ORGANIZED_ROOT="${ORGANIZED_ROOT:-pipeline/step7_lift_3d/lift_output_organized_${OUTPUT_TAG}}"
PROCESSED_ROOT="${PROCESSED_ROOT:-$DATA_ROOT/processed_sam2_${OUTPUT_TAG}}"
FEATURE_ROOT="${FEATURE_ROOT:-$PROCESSED_ROOT/vggt_features}"
FEATURE_CACHE_ROOT="${FEATURE_CACHE_ROOT:-outputs/vggt_feat_cache_${OUTPUT_TAG}}"

LOG_ROOT="${LOG_ROOT:-pipeline/${OUTPUT_TAG}_logs}"
STEP1_EXTRA_ARGS="${STEP1_EXTRA_ARGS:-}"
STEP2_EXTRA_ARGS="${STEP2_EXTRA_ARGS:-}"
STEP2B_EXTRA_ARGS="${STEP2B_EXTRA_ARGS:-}"
STEP3_EXTRA_ARGS="${STEP3_EXTRA_ARGS:-}"
STEP5_EXTRA_ARGS="${STEP5_EXTRA_ARGS:-}"
STEP6B_EXTRA_ARGS="${STEP6B_EXTRA_ARGS:-}"
STEP7_EXTRA_ARGS="${STEP7_EXTRA_ARGS:-}"
PROCESS_EXTRA_ARGS="${PROCESS_EXTRA_ARGS:-}"
STEP7B_EXTRA_ARGS="${STEP7B_EXTRA_ARGS:-}"

# Direct Step 1/2 commands need arrays; the parallel runners accept their
# EXTRA_ARGS strings themselves.
# shellcheck disable=SC2206
STEP1_EXTRA_ARR=($STEP1_EXTRA_ARGS)
# shellcheck disable=SC2206
STEP2_EXTRA_ARR=($STEP2_EXTRA_ARGS)

# Normalize comma- or whitespace-separated GPU specifications once. Child
# runners accept the resulting whitespace-separated form.
GPU_SPEC="${GPUS//,/ }"
# shellcheck disable=SC2206
GPU_ARR=($GPU_SPEC)
[[ "${#GPU_ARR[@]}" -ge 1 ]] || {
  echo "ERROR: expected at least one GPU, got an empty GPUS value." >&2
  exit 2
}
PRIMARY_GPU="${GPU_ARR[0]}"
GPU_LIST="${GPU_ARR[*]}"

cd "$REPO_ROOT"

reject_overlapping_output() {
  local label="$1" candidate="$2" protected="$3"
  local resolved_candidate resolved_protected
  resolved_candidate="$(realpath -m "$candidate")"
  resolved_protected="$(realpath -m "$protected")"
  case "$resolved_candidate/" in
    "$resolved_protected/"*) ;;
    *)
      case "$resolved_protected/" in
        "$resolved_candidate/"*) ;;
        *) return 0 ;;
      esac
      ;;
  esac
  echo "ERROR: $label overlaps a protected existing tree: $protected" >&2
  echo "Resolved output: $resolved_candidate" >&2
  exit 2
}

# Protect both the legacy outputs and the completed 2b-filter val data used by
# current evaluations. The default *_2bfilter_val paths are siblings of these.
reject_overlapping_output STEP1_ROOT "$STEP1_ROOT" \
  pipeline/step1_affordance/affordance_result
reject_overlapping_output STEP2_ROOT "$STEP2_ROOT" \
  pipeline/step2_clipwithaffordance/clipwithaffordance_output
reject_overlapping_output STEP2B_ROOT "$STEP2B_ROOT" \
  pipeline/step2b_relational_filter/clipwithaffordance_output
for protected in \
  pipeline/step3_point_prediction/point_clipwithaffordance_output \
  pipeline/step3_point_prediction/point_clipwithaffordance_output_2bfilter; do
  reject_overlapping_output STEP3_ROOT "$STEP3_ROOT" "$protected"
done
for protected in \
  pipeline/step4_crop_images/seg_image_output/point_clipwithaffordance_output \
  pipeline/step4_crop_images/seg_image_output/point_clipwithaffordance_output_2bfilter; do
  reject_overlapping_output STEP4_ROOT "$STEP4_ROOT" "$protected"
done
for spec in \
  "STEP5_ROOT|$STEP5_ROOT|pipeline/step5_molmo_sam/molmo_output" \
  "STEP5_ROOT|$STEP5_ROOT|pipeline/step5_molmo_sam/molmo_output_2bfilter" \
  "STEP6_ROOT|$STEP6_ROOT|pipeline/step6_molmo_merge/molmo_merge_output" \
  "STEP6_ROOT|$STEP6_ROOT|pipeline/step6_molmo_merge/molmo_merge_output_2bfilter" \
  "STEP6B_ROOT|$STEP6B_ROOT|pipeline/step6b_vggt_track/vggt_track_output" \
  "STEP6B_ROOT|$STEP6B_ROOT|pipeline/step6b_vggt_track/vggt_track_output_2bfilter" \
  "STEP7_ROOT|$STEP7_ROOT|pipeline/step7_lift_3d/lift_output" \
  "STEP7_ROOT|$STEP7_ROOT|pipeline/step7_lift_3d/lift_output_2bfilter" \
  "ORGANIZED_ROOT|$ORGANIZED_ROOT|pipeline/step7_lift_3d/lift_output_organized" \
  "ORGANIZED_ROOT|$ORGANIZED_ROOT|pipeline/step7_lift_3d/lift_output_organized_2bfilter" \
  "PROCESSED_ROOT|$PROCESSED_ROOT|$DATA_ROOT/processed_sam2" \
  "PROCESSED_ROOT|$PROCESSED_ROOT|$DATA_ROOT/processed_sam2_2bfilter" \
  "FEATURE_ROOT|$FEATURE_ROOT|$DATA_ROOT/processed_sam2/vggt_features" \
  "FEATURE_ROOT|$FEATURE_ROOT|$DATA_ROOT/processed_sam2_2bfilter/vggt_features"; do
  IFS='|' read -r label candidate protected <<< "$spec"
  reject_overlapping_output "$label" "$candidate" "$protected"
done

for required in \
  pipeline/step1_affordance/qwen.py \
  pipeline/step2_clipwithaffordance/clip_affordance.py \
  pipeline/step2b_relational_filter/run_step2b_parallel.sh \
  pipeline/step3_point_prediction/run_step3_parallel.sh \
  pipeline/step4_crop_images/qwen_seg_image.py \
  pipeline/step5_molmo_sam/run_step5_parallel.sh \
  pipeline/step6_molmo_merge/molmo_merge.py \
  pipeline/step6b_vggt_track/run_step6b_parallel.sh \
  pipeline/step7_lift_3d/run_step7_parallel.sh \
  pipeline/step7_lift_3d/convert_lift_results_to_clip_dir.py \
  dataset/run_preprocess_sam2_parallel.sh \
  pipeline/step7b_vggt_feats/run_extract_vggt_parallel.sh; do
  [[ -f "$required" ]] || {
    echo "ERROR: required file not found: $required" >&2
    exit 1
  }
done
[[ -f "$DATA_ROOT/benchmark_file_lists/${SPLIT}_set.csv" ]] || {
  echo "ERROR: benchmark split file not found: $DATA_ROOT/benchmark_file_lists/${SPLIT}_set.csv" >&2
  exit 1
}

print_command() {
  printf '  '
  printf '%q ' "$@"
  printf '\n'
}

run_stage() {
  local stage="$1"
  shift
  local index
  index="$(stage_index "$stage")"
  if (( index < FROM_INDEX )); then
    echo "[skip] $stage (--from $FROM_STAGE)"
    return 0
  fi
  echo
  echo "========== Stage $stage =========="
  print_command "$@"
  if [[ "$DRY_RUN" -eq 0 ]]; then
    "$@"
  fi
}

require_nonempty() {
  local stage="$1" root="$2" pattern="$3"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    return 0
  fi
  local count
  count="$(find "$root" -type f -name "$pattern" 2>/dev/null | wc -l)"
  if [[ "$count" -eq 0 ]]; then
    echo "ERROR: Stage $stage produced no $pattern files under $root" >&2
    exit 1
  fi
  echo "Stage $stage check: $count $pattern file(s) under $root"
}

if [[ "$DRY_RUN" -eq 0 ]]; then
  command -v flock >/dev/null 2>&1 || {
    echo "ERROR: flock is required." >&2
    exit 1
  }
  mkdir -p "$PROCESSED_ROOT"
  LOCK_FILE="${LOCK_FILE:-$PROCESSED_ROOT/.2bfilter_val_pipeline.lock}"
  exec 9<> "$LOCK_FILE"
  if ! flock -n 9; then
    echo "ERROR: the 2b-filter val pipeline is already running: $LOCK_FILE" >&2
    exit 1
  fi
  printf '%s\n' "$$" >&9
fi

echo "2b-filter val preprocessing (Steps 1 through 7b)"
echo "repo=$REPO_ROOT"
echo "gpus=[${GPU_LIST}]"
echo "from=$FROM_STAGE dry_run=$DRY_RUN"
echo "output_tag=$OUTPUT_TAG"
echo "processed=$PROCESSED_ROOT/$SPLIT"
echo "features=$FEATURE_ROOT/$SPLIT"

run_stage 1 env CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" \
  conda run --no-capture-output -n "$TASA_ENV" \
  python pipeline/step1_affordance/qwen.py \
  --data_root "$DATA_ROOT" --split "$SPLIT" --output_root "$STEP1_ROOT" \
  "${STEP1_EXTRA_ARR[@]}"
require_nonempty 1 "$STEP1_ROOT/$SPLIT" '*_affordance.json'

run_stage 2 env CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" \
  conda run --no-capture-output -n "$TASA_ENV" \
  python pipeline/step2_clipwithaffordance/clip_affordance.py \
  --data_root "$DATA_ROOT" --split "$SPLIT" \
  --affordance_root "$STEP1_ROOT" --output_root "$STEP2_ROOT" \
  "${STEP2_EXTRA_ARR[@]}"
require_nonempty 2 "$STEP2_ROOT/$SPLIT" '*_result.json'

run_stage 2b env \
  REPO_ROOT="$REPO_ROOT" ENV_NAME="$TASA_ENV" GPUS="$GPU_LIST" \
  DATA_ROOT="$DATA_ROOT" INPUT_ROOT="$STEP2_ROOT" OUTPUT_ROOT="$STEP2B_ROOT" \
  LOG_DIR="$LOG_ROOT/step2b/$SPLIT" EXTRA_ARGS="$STEP2B_EXTRA_ARGS" \
  bash pipeline/step2b_relational_filter/run_step2b_parallel.sh "$SPLIT"
require_nonempty 2b "$STEP2B_ROOT/$SPLIT" '*_result.json'

run_stage 3 env \
  REPO_ROOT="$REPO_ROOT" ENV_NAME="$TASA_ENV" GPUS="$GPU_LIST" \
  DATA_ROOT="$DATA_ROOT" AFFORDANCE_ROOT="$STEP1_ROOT" \
  LOG_DIR="$LOG_ROOT/step3/$SPLIT" EXTRA_ARGS="$STEP3_EXTRA_ARGS" \
  bash pipeline/step3_point_prediction/run_step3_parallel.sh "$SPLIT" \
  --input-root "$STEP2B_ROOT" --output-root "$STEP3_ROOT"
require_nonempty 3 "$STEP3_ROOT/$SPLIT" '*_point.json'

run_stage 4 env CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" \
  conda run --no-capture-output -n "$TASA_ENV" \
  python pipeline/step4_crop_images/qwen_seg_image.py \
  --split "$SPLIT" --raw_data_root "$DATA_ROOT" \
  --input_root "$STEP3_ROOT" --output_root "$STEP4_ROOT"
require_nonempty 4 "$STEP4_ROOT/$SPLIT" '*_crop.jpg'

run_stage 5 env \
  REPO_ROOT="$REPO_ROOT" ENV_NAME="$TASA_ENV" GPUS="$GPU_LIST" \
  DATA_ROOT="$DATA_ROOT" AFFORDANCE_ROOT="$STEP1_ROOT" \
  LOG_DIR="$LOG_ROOT/step5/$SPLIT" EXTRA_ARGS="$STEP5_EXTRA_ARGS" \
  bash pipeline/step5_molmo_sam/run_step5_parallel.sh "$SPLIT" \
  --input-root "$STEP4_ROOT" --output-root "$STEP5_ROOT"
require_nonempty 5 "$STEP5_ROOT/$SPLIT" '*_mask_data.npz'

run_stage 6 env CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" \
  conda run --no-capture-output -n "$TASA_ENV" \
  python pipeline/step6_molmo_merge/molmo_merge.py \
  --split "$SPLIT" --cropinfo_root "$STEP4_ROOT" \
  --molmo_root "$STEP5_ROOT" --merge_root "$STEP6_ROOT"
require_nonempty 6 "$STEP6_ROOT/$SPLIT" '*_mask_data.npz'

run_stage 6b env \
  REPO_ROOT="$REPO_ROOT" ENV_NAME="$VGGT_ENV" GPUS="$GPU_LIST" \
  DATA_ROOT="$DATA_ROOT" LOG_DIR="$LOG_ROOT/step6b/$SPLIT" \
  EXTRA_ARGS="$STEP6B_EXTRA_ARGS" \
  bash pipeline/step6b_vggt_track/run_step6b_parallel.sh "$SPLIT" \
  --input-root "$STEP6_ROOT" --output-root "$STEP6B_ROOT"
require_nonempty 6b "$STEP6B_ROOT/$SPLIT" '*_mask_data.npz'

run_stage 7 env \
  REPO_ROOT="$REPO_ROOT" ENV_NAME="$TASA_ENV" GPUS="$GPU_LIST" \
  SPLIT="$SPLIT" DATA_ROOT="$DATA_ROOT" \
  MOLMO_ROOT="$STEP6B_ROOT/$SPLIT" OUTPUT_DIR="$STEP7_ROOT/$SPLIT" \
  LOG_DIR="$LOG_ROOT/step7/$SPLIT" EXTRA_ARGS="$STEP7_EXTRA_ARGS" \
  bash pipeline/step7_lift_3d/run_step7_parallel.sh
require_nonempty 7 "$STEP7_ROOT/$SPLIT" 'lift_results_[0-9]*.json'

run_stage organize env CUDA_VISIBLE_DEVICES="$PRIMARY_GPU" \
  conda run --no-capture-output -n "$TASA_ENV" \
  python pipeline/step7_lift_3d/convert_lift_results_to_clip_dir.py \
  --split "$SPLIT" --data_root "$DATA_ROOT" \
  --lift_dir "$STEP7_ROOT/$SPLIT" --output_dir "$ORGANIZED_ROOT/$SPLIT"
require_nonempty organize "$ORGANIZED_ROOT/$SPLIT" mask_result.json

run_stage process env \
  REPO_ROOT="$REPO_ROOT" ENV_NAME="$PROCESS_ENV" GPUS="$GPU_LIST" \
  N_WORKERS="${#GPU_ARR[@]}" SPLIT="$SPLIT" ROOT_DIR="$DATA_ROOT" \
  CLIP_DIR="$ORGANIZED_ROOT/$SPLIT" SAVE_DIR="$PROCESSED_ROOT" \
  LOG_DIR="$LOG_ROOT/process/$SPLIT" EXTRA_ARGS="$PROCESS_EXTRA_ARGS" \
  bash dataset/run_preprocess_sam2_parallel.sh
require_nonempty process "$PROCESSED_ROOT/$SPLIT" filtered_point_cloud.ply

run_stage 7b env \
  REPO_ROOT="$REPO_ROOT" ENV_NAME="$TASA_ENV" GPUS="$GPU_LIST" \
  N_WORKERS="${#GPU_ARR[@]}" DATA_ROOT="$DATA_ROOT" \
  PROCESSED_DIR="$PROCESSED_ROOT/$SPLIT" FEAT_OUT_ROOT="$FEATURE_ROOT/$SPLIT" \
  CACHE_DIR="$FEATURE_CACHE_ROOT/$SPLIT" LOG_DIR="$LOG_ROOT/step7b/$SPLIT" \
  EXTRA_ARGS="$STEP7B_EXTRA_ARGS" \
  bash pipeline/step7b_vggt_feats/run_extract_vggt_parallel.sh "$SPLIT"
require_nonempty 7b "$FEATURE_ROOT/$SPLIT" vggt_dpt_source_feat.npy

if [[ "$DRY_RUN" -eq 0 ]]; then
  processed_count="$(find "$PROCESSED_ROOT/$SPLIT" -type f -name filtered_point_cloud.ply | wc -l)"
  feature_count="$(find "$FEATURE_ROOT/$SPLIT" -type f -name vggt_dpt_source_feat.npy | wc -l)"
  confidence_count="$(find "$FEATURE_ROOT/$SPLIT" -type f -name vggt_source_conf.npy | wc -l)"
  view_count="$(find "$FEATURE_ROOT/$SPLIT" -type f -name vggt_source_view_count.npy | wc -l)"
  metadata_count="$(find "$FEATURE_ROOT/$SPLIT" -type f -name vggt_source_meta.json | wc -l)"
  if [[ "$processed_count" -ne "$feature_count" \
        || "$processed_count" -ne "$confidence_count" \
        || "$processed_count" -ne "$view_count" \
        || "$processed_count" -ne "$metadata_count" ]]; then
    echo "ERROR: incomplete processed/VGGT bundles:" >&2
    echo "  processed=$processed_count feature=$feature_count confidence=$confidence_count view_count=$view_count metadata=$metadata_count" >&2
    exit 1
  fi
  echo
  echo "Complete: $processed_count processed val samples and complete VGGT bundles."
else
  echo
  echo "Dry run complete; no files or directories were created."
fi
