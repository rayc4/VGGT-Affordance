#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

bash -n \
  dataset/run_original_vggt_hybrid_experiment.sh \
  dataset/run_preprocess_vggt_legacy_parallel.sh

dry_run="$(bash dataset/run_original_vggt_hybrid_experiment.sh --dry-run --with-train)"

required_fragments=(
  "original_vggt_hybrid_v1"
  "processed_sam2_original_vggt_hybrid_v1"
  "molmo_merge_output/val"
  "molmo_merge_output/train"
  "point_clipwithaffordance_output"
  "--component-selection anchor-consensus"
  "--independent-anchor-video-votes"
  "--crop-semantic-center"
  "--crop-semantic-neighborhood-points 2048"
  "--max-published-views 6"
  "--min-published-view-component-f1 0.10"
  "--min-description-coverage 0.7258426966292135"
  "--require-hardlink-dedup"
  "legacy_contract.json"
  "Validation passed; starting train preprocessing."
  "scripts/train.sh original_vggt_hybrid_v1"
)
for fragment in "${required_fragments[@]}"; do
  case "$dry_run" in
    *"$fragment"*) ;;
    *) echo "ERROR: hybrid dry run omitted: $fragment" >&2; exit 1 ;;
  esac
done

for gpu in 0 1 2 3 4 5 6 7; do
  case "$dry_run" in
    *"CUDA_VISIBLE_DEVICES=$gpu"*) ;;
    *) echo "ERROR: hybrid dry run omitted GPU $gpu" >&2; exit 1 ;;
  esac
done

case "$dry_run" in
  *"--anchor-root pipeline/step6_molmo_merge/molmo_merge_output/val"*) ;;
  *) echo "ERROR: validation does not consume original step-6 artifacts" >&2; exit 1 ;;
esac
case "$dry_run" in
  *"--point-root pipeline/step3_point_prediction/point_clipwithaffordance_output"*) ;;
  *) echo "ERROR: hybrid run does not consume original step-3 points" >&2; exit 1 ;;
esac
case "$dry_run" in
  *"NO_POINT_ROOT=1"*|*"step1_direct_anchors"*|*"direct_vggt_v3g"*)
    echo "ERROR: hybrid runner references replacement preprocessing artifacts" >&2
    exit 1
    ;;
esac

if PROCESSED_ROOT=scenefun3d/processed_sam2 \
  bash dataset/run_original_vggt_hybrid_experiment.sh --dry-run >/dev/null 2>&1; then
  echo "ERROR: runner accepted the repository-original processed output root" >&2
  exit 1
fi

if EXPERIMENT_ROOT=pipeline/step6_molmo_merge/molmo_merge_output/original_vggt_hybrid \
  bash dataset/run_original_vggt_hybrid_experiment.sh --dry-run >/dev/null 2>&1; then
  echo "ERROR: runner accepted an experiment root nested under step-6 inputs" >&2
  exit 1
fi

if LOG_ROOT=scenefun3d/preprocessing_experiments/unrelated/logs \
  bash dataset/run_original_vggt_hybrid_experiment.sh --dry-run >/dev/null 2>&1; then
  echo "ERROR: runner accepted logs outside the isolated experiment root" >&2
  exit 1
fi

if VAL_MIN_COVERAGE=0 \
  bash dataset/run_original_vggt_hybrid_experiment.sh --dry-run >/dev/null 2>&1; then
  echo "ERROR: runner accepted validation coverage below 323/445" >&2
  exit 1
fi

if VAL_MIN_COVERAGE=0.7258 \
  bash dataset/run_original_vggt_hybrid_experiment.sh --dry-run >/dev/null 2>&1; then
  echo "ERROR: runner accepted rounded validation coverage below exact 323/445" >&2
  exit 1
fi

if STEP6_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v3g/proposals \
  bash dataset/run_original_vggt_hybrid_experiment.sh --dry-run >/dev/null 2>&1; then
  echo "ERROR: runner accepted non-original step-6 anchors" >&2
  exit 1
fi

if POINT_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v3g/points \
  bash dataset/run_original_vggt_hybrid_experiment.sh --dry-run >/dev/null 2>&1; then
  echo "ERROR: runner accepted non-original step-3 points" >&2
  exit 1
fi

hostile_env="$(
  NO_POINT_ROOT=1 \
  ALLOW_COPY_FALLBACK=1 \
  VGGT_EXTRA_ARGS=--no-autocast \
    bash dataset/run_original_vggt_hybrid_experiment.sh --dry-run
)"
case "$hostile_env" in
  *"--point-root pipeline/step3_point_prediction/point_clipwithaffordance_output"*) ;;
  *) echo "ERROR: ambient NO_POINT_ROOT bypassed original step-3 points" >&2; exit 1 ;;
esac
case "$hostile_env" in
  *"--no-point-root"*)
    echo "ERROR: hostile environment disabled original step-3 points" >&2
    exit 1
    ;;
esac
case "$hostile_env" in
  *"shared_storage=required-hardlinks"*) ;;
  *) echo "ERROR: ambient copy fallback weakened storage requirement" >&2; exit 1 ;;
esac
case "$hostile_env" in
  *"--independent-anchor-video-votes"*) ;;
  *) echo "ERROR: supplemental VGGT args removed required hybrid voting" >&2; exit 1 ;;
esac

val_only="$(bash dataset/run_original_vggt_hybrid_experiment.sh --dry-run)"
case "$val_only" in
  *"molmo_merge_output/train"*)
    echo "ERROR: validation-only mode unexpectedly starts train preprocessing" >&2
    exit 1
    ;;
esac

echo "original+VGGT hybrid shell, isolation, and dry-run tests passed."
