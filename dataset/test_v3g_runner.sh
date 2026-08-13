#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"

bash -n \
  dataset/run_direct_vggt_v3g_experiment.sh \
  dataset/run_v3g_consensus_parallel.sh \
  dataset/run_v3e_consensus_parallel.sh

GPUS_8="0,1,2,3,4,5,6,7"
resolved="$(GPUS="$GPUS_8" N_WORKERS=8 bash dataset/run_direct_vggt_v3g_experiment.sh --dry-run)"

required_fragments=(
  "direct_vggt_v3g/proposals"
  "direct_vggt_v3g/clip_cache"
  "direct_vggt_v3g/consensus"
  "direct_vggt_v3g/diagnostics/val/01_proposal_contract.json"
  "direct_vggt_v3g/diagnostics/val/02_proposal_semantic_oracle.json"
  "direct_vggt_v3g/diagnostics/val/03_proposal_oracle_gate.json"
  "direct_vggt_v3g/diagnostics/val/04_selected_contract.json"
  "direct_vggt_v3g/diagnostics/val/05_selected_semantic_alignment.json"
  "direct_vggt_v3g/diagnostics/val/06_legacy_contract.json"
  "processed_sam2_vggt_direct_v3g"
  "--v3g-independent-full-description-hedge"
  "--qwen-max-pixels"
  "2007040"
  "--variant-aware-proposals"
  "--verified-anchor-target"
  "proposal pool: primary=6 independent-v2=4 oracle/selector=10 publish=3"
  "--hard-minimum-aligned 312"
  "--target-aligned 334"
  "--expected-max-anchors 10"
  "--max-anchors 10"
  "MAX_PROPOSALS=10"
  "MAX_ANCHORS=3"
  "--max-anchors 3"
  "--min-aligned-coverage 0.70"
  "--min-description-coverage 0.7258"
  "--require-hardlink-dedup"
  "NO_POINT_ROOT=1"
  "N_WORKERS=8"
)
for fragment in "${required_fragments[@]}"; do
  case "$resolved" in
    *"$fragment"*) ;;
    *) echo "ERROR: v3g dry run omitted: $fragment" >&2; exit 1 ;;
  esac
done
case "$resolved" in
  *direct_vggt_v3f/proposals*|*direct_vggt_v3e/proposals*|*processed_sam2/val*)
    echo "ERROR: v3g dry run references a forbidden artifact tree" >&2
    exit 1
    ;;
esac

consensus="$(GPUS="$GPUS_8" N_WORKERS=8 bash dataset/run_v3g_consensus_parallel.sh --dry-run)"
case "$consensus" in
  *"--max-proposals 10"*"--max-anchors 3"*) ;;
  *) echo "ERROR: consensus does not consume the complete max-10 proposal pool" >&2; exit 1 ;;
esac
for gpu in 0 1 2 3 4 5 6 7; do
  case "$consensus" in
    *"CUDA_VISIBLE_DEVICES=$gpu"*) ;;
    *) echo "ERROR: consensus dry run omitted GPU $gpu" >&2; exit 1 ;;
  esac
done

direct="$(
  GPUS="$GPUS_8" N_WORKERS=8 \
  OUTPUT_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v3g/dry_proposals \
  CACHE_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v3g/dry_cache \
  bash dataset/run_direct_anchors_parallel.sh --dry-run
)"
for gpu in 0 1 2 3 4 5 6 7; do
  case "$direct" in
    *"CUDA_VISIBLE_DEVICES=$gpu"*) ;;
    *) echo "ERROR: proposal dry run omitted GPU $gpu" >&2; exit 1 ;;
  esac
done

if PROPOSAL_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v3f/proposals \
  GPUS="$GPUS_8" N_WORKERS=8 \
  bash dataset/run_direct_vggt_v3g_experiment.sh --dry-run >/dev/null 2>&1; then
  echo "ERROR: v3g runner accepted a v3f proposal root" >&2
  exit 1
fi

if PROPOSAL_POOL_MAX=8 \
  GPUS="$GPUS_8" N_WORKERS=8 \
  bash dataset/run_direct_vggt_v3g_experiment.sh --dry-run >/dev/null 2>&1; then
  echo "ERROR: v3g runner accepted an inconsistent oracle/selector pool" >&2
  exit 1
fi

if EXPERIMENT_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v3g \
  PROPOSAL_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v3g_other/proposals \
  GPUS="$GPUS_8" N_WORKERS=8 \
  bash dataset/run_direct_vggt_v3g_experiment.sh --dry-run >/dev/null 2>&1; then
  echo "ERROR: v3g runner accepted proposal artifacts outside EXPERIMENT_ROOT" >&2
  exit 1
fi

if PROPOSAL_ROOT=scenefun3d/preprocessing_experiments/direct_vggt_v3f/proposals/val \
  GPUS="$GPUS_8" N_WORKERS=8 \
  bash dataset/run_v3g_consensus_parallel.sh --dry-run >/dev/null 2>&1; then
  echo "ERROR: v3g consensus runner accepted a v3f proposal root" >&2
  exit 1
fi

echo "v3g shell and dry-run tests passed."
