#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

bash -n \
  dataset/run_qwen25_relational_vggt_standalone_experiment.sh \
  dataset/run_preprocess_vggt_legacy_parallel.sh

dry_run="$(
  bash dataset/run_qwen25_relational_vggt_standalone_experiment.sh \
    --dry-run --with-train
)"

required_fragments=(
  "scope=all benchmark descriptions"
  "qwen25_relational_vggt_standalone_v2"
  "processed_sam2_qwen25_relational_vggt_standalone_v2"
  "--v5-coupled-relational-grounding"
  "--v3-role-aware-targeting"
  "--v3-conflict-filter"
  "--qwen-model Qwen/Qwen2.5-VL-7B-Instruct"
  "--verified-anchor-target 4"
  "--min-status-coverage 1.0"
  "pipeline.step4_vggt_legacy.semantic_alignment_audit"
  "--anchor-root scenefun3d/preprocessing_experiments/qwen25_relational_vggt_standalone_v2/anchors/val"
  "--no-point-root"
  "--component-selection anchor-consensus"
  "--independent-anchor-video-votes"
  "--pretrack-semantic-lift-gate"
  "--pretrack-candidate-pool 12"
  "--pretrack-consensus-radius 0.20"
  "--pretrack-min-independent-videos 2"
  "--pretrack-single-anchor-min-confidence 0.85"
  "--context-frames 8"
  "--max-published-views 6"
  "--allow-empty-output-shard"
  "--require-hardlink-dedup"
  "03_standalone_legacy_contract.json"
  "--min-description-coverage 0.0"
  "scripts/train.sh qwen25_relational_vggt_standalone_v2"
)
for fragment in "${required_fragments[@]}"; do
  case "$dry_run" in
    *"$fragment"*) ;;
    *) echo "ERROR: relational standalone dry run omitted: $fragment" >&2; exit 1 ;;
  esac
done

for gpu in 0 1 2 3 4 5 6 7; do
  case "$dry_run" in
    *"CUDA_VISIBLE_DEVICES=$gpu"*) ;;
    *) echo "ERROR: relational standalone dry run omitted GPU $gpu" >&2; exit 1 ;;
  esac
done

case "$dry_run" in
  *"--description-plan"*|*"--repair-plan"*|*"pipeline.step5_vggt_repair"*|*"--baseline-root"*)
    echo "ERROR: standalone experiment depends on repair planning or unioning" >&2
    exit 1
    ;;
esac
case "$dry_run" in
  *"--v3-verifier-relocation"*|*"--v3-target-surface-grounding"*|*"--v3g-independent"*)
    echo "ERROR: standalone experiment bypassed or relocated strict verification" >&2
    exit 1
    ;;
esac

line_of() {
  local needle="$1"
  awk -v needle="$needle" 'index($0, needle) { print NR; exit }' <<<"$dry_run"
}
val_direct="$(line_of 'pipeline.step1_direct_anchors.audit')"
val_vggt="$(line_of 'pipeline.step4_vggt_legacy.preprocess')"
val_final="$(line_of '03_standalone_legacy_contract.json')"
train_direct="$(line_of 'qwen25-relational standalone train: scope=all benchmark descriptions')"
if [ -z "$val_direct" ] || [ -z "$val_vggt" ] || [ -z "$val_final" ] || \
   [ -z "$train_direct" ] || [ "$val_direct" -ge "$val_vggt" ] || \
   [ "$val_vggt" -ge "$val_final" ] || [ "$val_final" -ge "$train_direct" ]; then
  echo "ERROR: standalone anchor, VGGT, audit, and train promotion order is wrong" >&2
  exit 1
fi

expect_rejected() {
  local label="$1"
  shift
  if env "$@" bash dataset/run_qwen25_relational_vggt_standalone_experiment.sh \
    --dry-run >/dev/null 2>&1; then
    echo "ERROR: standalone runner accepted unsafe $label" >&2
    exit 1
  fi
}

expect_rejected "DATA_ROOT" DATA_ROOT=scenefun3d_copy
expect_rejected "original processed root" PROCESSED_ROOT=scenefun3d/processed_sam2
expect_rejected "repair namespace" \
  EXPERIMENT_ROOT=scenefun3d/preprocessing_experiments/qwen25_relational_vggt_repair_v1
expect_rejected "standalone v1 namespace" \
  EXPERIMENT_ROOT=scenefun3d/preprocessing_experiments/qwen25_relational_vggt_standalone_v1
expect_rejected "processed root outside standalone namespace" \
  PROCESSED_ROOT=scenefun3d/processed_sam2_qwen25_relational_vggt_repair_v1
expect_rejected "anchors outside experiment" ANCHOR_ROOT=/tmp/standalone-anchors
expect_rejected "description allowlist" \
  DIRECT_TUNING_ARGS='--description-plan /tmp/foreign.json'
expect_rejected "model override" DIRECT_TUNING_ARGS='--qwen-model other/model'
expect_rejected "mode override" DIRECT_TUNING_ARGS='--v3-target-surface-grounding'
expect_rejected "repair plan" VGGT_EXTRA_ARGS='--repair-plan /tmp/foreign.json'
expect_rejected "invalid coverage" VAL_MIN_COVERAGE=1.1

val_only="$(bash dataset/run_qwen25_relational_vggt_standalone_experiment.sh --dry-run)"
case "$val_only" in
  *"--split train"*|*"standalone train:"*)
    echo "ERROR: validation-only mode unexpectedly starts train preprocessing" >&2
    exit 1
    ;;
esac

empty_allowed="$(
  GPUS=0 N_WORKERS=1 SPLIT=val NO_POINT_ROOT=1 \
  ANCHOR_ROOT=scenefun3d/preprocessing_experiments/qwen25_relational_vggt_standalone_v2/anchors/val \
  SAVE_DIR=scenefun3d/processed_sam2_qwen25_relational_vggt_standalone_v2 \
  ALLOW_EMPTY_OUTPUT_SHARDS=1 \
    bash dataset/run_preprocess_vggt_legacy_parallel.sh --dry-run
)"
case "$empty_allowed" in
  *"--allow-empty-output-shard"*) ;;
  *) echo "ERROR: standalone empty-shard policy was not forwarded" >&2; exit 1 ;;
esac

default_policy="$(
  GPUS=0 N_WORKERS=1 SPLIT=val NO_POINT_ROOT=1 \
  ANCHOR_ROOT=scenefun3d/preprocessing_experiments/qwen25_relational_vggt_standalone_v2/anchors/val \
  SAVE_DIR=scenefun3d/processed_sam2_qwen25_relational_vggt_standalone_v2 \
    bash dataset/run_preprocess_vggt_legacy_parallel.sh --dry-run
)"
case "$default_policy" in
  *"--allow-empty-output-shard"*)
    echo "ERROR: historical VGGT runs silently allow empty output shards" >&2
    exit 1
    ;;
esac

echo "Qwen2.5 relational VGGT standalone shell, isolation, and ordering tests passed."
