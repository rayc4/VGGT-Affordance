#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

bash -n \
  dataset/run_qwen25_relational_vggt_repair_experiment.sh \
  dataset/run_preprocess_vggt_legacy_parallel.sh

dry_run="$(
  bash dataset/run_qwen25_relational_vggt_repair_experiment.sh \
    --dry-run --with-train
)"

required_fragments=(
  "--scope uncovered"
  "--description-plan scenefun3d/preprocessing_experiments/qwen25_relational_vggt_repair_v1/plans/val.json"
  "--v4-strict-relational-grounding"
  "--v3-role-aware-targeting"
  "--v3-conflict-filter"
  "--qwen-model Qwen/Qwen2.5-VL-7B-Instruct"
  "--verified-anchor-target 2"
  "--min-status-coverage 1.0"
  "pipeline.step4_vggt_legacy.semantic_alignment_audit"
  "--anchor-root scenefun3d/preprocessing_experiments/qwen25_relational_vggt_repair_v1/anchors/val"
  "--no-point-root"
  "--repair-plan scenefun3d/preprocessing_experiments/qwen25_relational_vggt_repair_v1/plans/val.json"
  "--component-selection anchor-consensus"
  "--context-frames 8"
  "--max-published-views 6"
  "--require-hardlink-dedup"
  "processed_sam2_qwen25_relational_vggt_repair_candidates_v1"
  "processed_sam2_qwen25_relational_vggt_repair_v1"
  "--min-description-coverage 0.7325842696629213"
  "scripts/train.sh qwen25_relational_vggt_repair_v1"
)
for fragment in "${required_fragments[@]}"; do
  case "$dry_run" in
    *"$fragment"*) ;;
    *) echo "ERROR: relational repair dry run omitted: $fragment" >&2; exit 1 ;;
  esac
done

for gpu in 0 1 2 3 4 5 6 7; do
  case "$dry_run" in
    *"CUDA_VISIBLE_DEVICES=$gpu"*) ;;
    *) echo "ERROR: relational repair dry run omitted GPU $gpu" >&2; exit 1 ;;
  esac
done

case "$dry_run" in
  *"pipeline/step6_molmo_merge/molmo_merge_output"*|*"point_clipwithaffordance_output"*)
    echo "ERROR: relational experiment consumed original semantic intermediates" >&2
    exit 1
    ;;
esac
case "$dry_run" in
  *"--v3-verifier-relocation"*|*"--v3-target-surface-grounding"*|*"--v3g-independent"*)
    echo "ERROR: relational experiment bypassed or relocated strict verification" >&2
    exit 1
    ;;
esac

line_of() {
  local needle="$1"
  awk -v needle="$needle" 'index($0, needle) { print NR; exit }' <<<"$dry_run"
}
val_direct="$(line_of 'pipeline.step1_direct_anchors.audit')"
val_vggt="$(line_of 'pipeline.step4_vggt_legacy.preprocess')"
val_materialize="$(line_of 'pipeline.step5_vggt_repair.materialize')"
val_final="$(line_of 'pipeline.step5_vggt_repair.audit')"
train_plan="$(line_of 'pipeline.step5_vggt_repair.plan --data-root scenefun3d --baseline-root scenefun3d/processed_sam2 --split train')"
if [ -z "$val_direct" ] || [ -z "$val_vggt" ] || [ -z "$val_materialize" ] || \
   [ -z "$val_final" ] || [ -z "$train_plan" ] || \
   [ "$val_direct" -ge "$val_vggt" ] || [ "$val_vggt" -ge "$val_materialize" ] || \
   [ "$val_materialize" -ge "$val_final" ] || [ "$val_final" -ge "$train_plan" ]; then
  echo "ERROR: strict anchor, VGGT, union, and validation promotion order is wrong" >&2
  exit 1
fi

expect_rejected() {
  local label="$1"
  shift
  if env "$@" bash dataset/run_qwen25_relational_vggt_repair_experiment.sh \
    --dry-run >/dev/null 2>&1; then
    echo "ERROR: relational runner accepted unsafe $label" >&2
    exit 1
  fi
}

expect_rejected "DATA_ROOT" DATA_ROOT=scenefun3d_copy
expect_rejected "BASELINE_ROOT" BASELINE_ROOT=scenefun3d/processed_sam2_copy
expect_rejected "baseline output" REPAIR_ROOT=scenefun3d/processed_sam2
expect_rejected "shared candidate/final root" \
  REPAIR_ROOT=scenefun3d/processed_sam2_qwen25_relational_vggt_repair_candidates_same \
  FINAL_ROOT=scenefun3d/processed_sam2_qwen25_relational_vggt_repair_candidates_same
expect_rejected "foreign experiment namespace" \
  EXPERIMENT_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v3g
expect_rejected "plan outside experiment" PLAN_ROOT=/tmp/qwen25-relational-plan
expect_rejected "below-promotion coverage" VAL_MIN_COVERAGE=0.7303370786516854
expect_rejected "model override" DIRECT_TUNING_ARGS='--qwen-model other/model'
expect_rejected "mode override" DIRECT_TUNING_ARGS='--v3-target-surface-grounding'
expect_rejected "foreign repair plan" VGGT_EXTRA_ARGS='--repair-plan /tmp/foreign.json'

val_only="$(bash dataset/run_qwen25_relational_vggt_repair_experiment.sh --dry-run)"
case "$val_only" in
  *"--split train"*|*"plans/train.json"*)
    echo "ERROR: validation-only mode unexpectedly starts train preprocessing" >&2
    exit 1
    ;;
esac

echo "Qwen2.5 relational VGGT repair shell, isolation, and gate-order tests passed."
