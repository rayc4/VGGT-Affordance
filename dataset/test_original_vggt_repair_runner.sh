#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

bash -n \
  dataset/run_original_vggt_repair_experiment.sh \
  dataset/run_preprocess_vggt_legacy_parallel.sh

dry_run="$(bash dataset/run_original_vggt_repair_experiment.sh --dry-run --with-train)"

required_fragments=(
  "pipeline.step5_vggt_repair.plan"
  "--scope any-invalid"
  "--baseline-root scenefun3d/processed_sam2"
  "plans/val.json"
  "plans/train.json"
  "processed_sam2_original_vggt_repair_candidates_v1"
  "processed_sam2_original_vggt_repair_v1"
  "--repair-plan scenefun3d/preprocessing_experiments/original_vggt_repair_v1/plans/val.json"
  "--repair-plan scenefun3d/preprocessing_experiments/original_vggt_repair_v1/plans/train.json"
  "pipeline.step4_vggt_legacy.audit"
  "pipeline.step5_vggt_repair.materialize"
  "pipeline.step5_vggt_repair.audit"
  "--component-selection anchor-consensus"
  "--independent-anchor-video-votes"
  "--crop-semantic-center"
  "--max-published-views 6"
  "--require-hardlink-dedup"
  "--min-description-coverage 0.7258426966292135"
  "Final validation union passed; starting train repair preprocessing."
  "scripts/train.sh original_vggt_repair_v1"
)
for fragment in "${required_fragments[@]}"; do
  case "$dry_run" in
    *"$fragment"*) ;;
    *) echo "ERROR: repair-only dry run omitted: $fragment" >&2; exit 1 ;;
  esac
done

for gpu in 0 1 2 3 4 5 6 7; do
  case "$dry_run" in
    *"CUDA_VISIBLE_DEVICES=$gpu"*) ;;
    *) echo "ERROR: repair-only dry run omitted GPU $gpu" >&2; exit 1 ;;
  esac
done

case "$dry_run" in
  *"--anchor-root pipeline/step6_molmo_merge/molmo_merge_output/val"*) ;;
  *) echo "ERROR: validation repair does not consume pinned original anchors" >&2; exit 1 ;;
esac
case "$dry_run" in
  *"--point-root pipeline/step3_point_prediction/point_clipwithaffordance_output"*) ;;
  *) echo "ERROR: repair does not consume pinned original points" >&2; exit 1 ;;
esac
case "$dry_run" in
  *"--anchor-root pipeline/step6_molmo_merge/molmo_merge_output/train"*) ;;
  *) echo "ERROR: train repair does not consume pinned original step-6 anchors" >&2; exit 1 ;;
esac
case "$dry_run" in
  *"--no-point-root"*|*"copy-fallback"*|*"step1_direct_anchors"*|*"direct_vggt_v3"*)
    echo "ERROR: repair-only runner weakened or replaced an original input/storage invariant" >&2
    exit 1
    ;;
esac

line_of() {
  local needle="$1"
  awk -v needle="$needle" 'index($0, needle) { print NR; exit }' <<<"$dry_run"
}
val_repair_audit_line="$(line_of 'pipeline.step4_vggt_legacy.audit --data-root scenefun3d --root scenefun3d/processed_sam2_original_vggt_repair_candidates_v1 --split val')"
val_materialize_line="$(line_of 'pipeline.step5_vggt_repair.materialize --baseline-root scenefun3d/processed_sam2')"
val_final_audit_line="$(line_of 'pipeline.step5_vggt_repair.audit --data-root scenefun3d')"
train_plan_line="$(line_of 'pipeline.step5_vggt_repair.plan --data-root scenefun3d --baseline-root scenefun3d/processed_sam2 --split train')"
if [ -z "$val_repair_audit_line" ] || [ -z "$val_materialize_line" ] || \
   [ -z "$val_final_audit_line" ] || [ -z "$train_plan_line" ] || \
   [ "$val_repair_audit_line" -ge "$val_materialize_line" ] || \
   [ "$val_materialize_line" -ge "$val_final_audit_line" ] || \
   [ "$val_final_audit_line" -ge "$train_plan_line" ]; then
  echo "ERROR: validation repair/materialize/promotion gates do not precede train" >&2
  exit 1
fi

case "$dry_run" in
  *"pipeline.step5_vggt_repair.audit --data-root scenefun3d --baseline-root scenefun3d/processed_sam2 --repair-root scenefun3d/processed_sam2_original_vggt_repair_candidates_v1 --root scenefun3d/processed_sam2_original_vggt_repair_v1 --repair-plan scenefun3d/preprocessing_experiments/original_vggt_repair_v1/plans/val.json --split val"*) ;;
  *) echo "ERROR: final validation audit omitted an explicit guarded source/plan" >&2; exit 1 ;;
esac

expect_rejected() {
  local label="$1"
  shift
  if env "$@" bash dataset/run_original_vggt_repair_experiment.sh --dry-run \
    >/dev/null 2>&1; then
    echo "ERROR: repair-only runner accepted unsafe $label" >&2
    exit 1
  fi
}

expect_rejected "DATA_ROOT override" DATA_ROOT=scenefun3d_copy
expect_rejected "BASELINE_ROOT override" BASELINE_ROOT=scenefun3d/processed_sam2_anchor
expect_rejected "STEP6_ROOT override" STEP6_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v3g/proposals
expect_rejected "POINT_ROOT override" POINT_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v3g/points
expect_rejected "original baseline as REPAIR_ROOT" REPAIR_ROOT=scenefun3d/processed_sam2
expect_rejected "same candidate/final roots" \
  REPAIR_ROOT=scenefun3d/processed_sam2_original_vggt_repair_candidates_v1 \
  FINAL_ROOT=scenefun3d/processed_sam2_original_vggt_repair_candidates_v1
expect_rejected "logs outside experiment" \
  LOG_ROOT=scenefun3d/preprocessing_experiments/unrelated/logs
expect_rejected "plan outside experiment" PLAN_ROOT=/tmp/vggt-repair-hostile-plan
expect_rejected "rounded-below-baseline coverage" VAL_MIN_COVERAGE=0.7258
expect_rejected "disabled point root through supplemental args" \
  VGGT_EXTRA_ARGS=--no-point-root
expect_rejected "foreign repair plan through supplemental args" \
  VGGT_EXTRA_ARGS=--repair-plan=/tmp/foreign.json
expect_rejected "copy fallback through supplemental args" \
  VGGT_EXTRA_ARGS=--no-hardlink-dedup

hostile_env="$(
  NO_POINT_ROOT=1 \
  ALLOW_COPY_FALLBACK=1 \
  REPAIR_PLAN=/tmp/foreign.json \
  EXTRA_ARGS=--no-point-root \
  VGGT_EXTRA_ARGS=--no-autocast \
    bash dataset/run_original_vggt_repair_experiment.sh --dry-run
)"
case "$hostile_env" in
  *"--point-root pipeline/step3_point_prediction/point_clipwithaffordance_output"*) ;;
  *) echo "ERROR: ambient environment bypassed the pinned point root" >&2; exit 1 ;;
esac
case "$hostile_env" in
  *"shared_storage=required-hardlinks"*) ;;
  *) echo "ERROR: ambient environment enabled copy fallback" >&2; exit 1 ;;
esac
case "$hostile_env" in
  *"--repair-plan scenefun3d/preprocessing_experiments/original_vggt_repair_v1/plans/val.json"*) ;;
  *) echo "ERROR: ambient environment replaced the generated repair plan" >&2; exit 1 ;;
esac
case "$hostile_env" in
  *"--no-point-root"*|*"/tmp/foreign.json"*)
    echo "ERROR: hostile ambient runner arguments leaked into a worker command" >&2
    exit 1
    ;;
esac

val_only="$(bash dataset/run_original_vggt_repair_experiment.sh --dry-run)"
case "$val_only" in
  *"--split train"*|*"plans/train.json"*)
    echo "ERROR: validation-only mode unexpectedly starts train preprocessing" >&2
    exit 1
    ;;
esac

echo "original+VGGT repair-only shell, isolation, gate-order, and dry-run tests passed."
