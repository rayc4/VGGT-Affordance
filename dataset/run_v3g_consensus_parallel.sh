#!/usr/bin/env bash
# Isolated v3g entry point for the shared prediction-only consensus worker.
# Code is shared with v3e/v3f, but artifacts, locks, and logs are v3g-only.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SPLIT="${SPLIT:-val}"
DATA_ROOT="${DATA_ROOT:-scenefun3d}"
V3G_EXPERIMENT_ROOT="${V3G_EXPERIMENT_ROOT:-scenefun3d/preprocessing_experiments/direct_vggt_v3g}"

export REPO_ROOT SPLIT DATA_ROOT
export PROPOSAL_ROOT="${PROPOSAL_ROOT:-$V3G_EXPERIMENT_ROOT/proposals/$SPLIT}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$V3G_EXPERIMENT_ROOT/consensus/$SPLIT}"
export LOG_DIR="${LOG_DIR:-$V3G_EXPERIMENT_ROOT/logs/consensus/$SPLIT}"
export LOCK_FILE="${LOCK_FILE:-$OUTPUT_ROOT/.v3g_consensus.lock}"
MAX_PROPOSALS="${MAX_PROPOSALS:-10}"
MAX_ANCHORS="${MAX_ANCHORS:-3}"
EXTRA_ARGS="${EXTRA_ARGS:---local-files-only --variant-aware-proposals --cluster-radius 0.12 --min-video-support 2 --context-frames 8 --context-window-seconds 3.0 --single-proposal-min-track-frames 3}"

for integer_name in MAX_PROPOSALS MAX_ANCHORS; do
  integer_value="${!integer_name}"
  if ! [[ "$integer_value" =~ ^[1-9][0-9]*$ ]]; then
    echo "ERROR: $integer_name must be a positive integer." >&2
    exit 2
  fi
done
if [ "$MAX_ANCHORS" -ge "$MAX_PROPOSALS" ]; then
  echo "ERROR: MAX_ANCHORS must be smaller than MAX_PROPOSALS." >&2
  exit 2
fi

# The experiment runner owns both ceilings so its persisted oracle and the
# selector cannot silently evaluate different proposal pools.
# shellcheck disable=SC2086
for extra_arg in $EXTRA_ARGS; do
  case "$extra_arg" in
    --max-proposals|--max-proposals=*|--max-anchors|--max-anchors=*)
      echo "ERROR: '$extra_arg' is controlled by MAX_PROPOSALS/MAX_ANCHORS." >&2
      exit 2
      ;;
  esac
done
export EXTRA_ARGS="$EXTRA_ARGS --max-proposals $MAX_PROPOSALS --max-anchors $MAX_ANCHORS"

cd "$REPO_ROOT"
command -v realpath >/dev/null 2>&1 || {
  echo "ERROR: realpath is required for v3g output isolation checks." >&2
  exit 1
}
FORBIDDEN_ARTIFACT_ROOTS=(
  "$(realpath -m "$DATA_ROOT/processed_sam2")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v1")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v2")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v3a")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v3b")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v3c")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v3d")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v3e")"
  "$(realpath -m "$DATA_ROOT/preprocessing_experiments/direct_vggt_v3f")"
  "$(realpath -m "$DATA_ROOT/processed_sam2_vggt_direct_v1")"
  "$(realpath -m "$DATA_ROOT/processed_sam2_vggt_direct_v2")"
  "$(realpath -m "$DATA_ROOT/processed_sam2_vggt_direct_v3e")"
  "$(realpath -m "$DATA_ROOT/processed_sam2_vggt_direct_v3f")"
)
for named_path in \
  "V3G_EXPERIMENT_ROOT:$V3G_EXPERIMENT_ROOT" \
  "PROPOSAL_ROOT:$PROPOSAL_ROOT" \
  "OUTPUT_ROOT:$OUTPUT_ROOT" \
  "LOG_DIR:$LOG_DIR" \
  "LOCK_FILE:$LOCK_FILE"; do
  name="${named_path%%:*}"
  value="${named_path#*:}"
  resolved="$(realpath -m "$value")"
  case "$resolved" in
    *v3g*) ;;
    *) echo "ERROR: $name must resolve inside a v3g namespace: $resolved" >&2; exit 2 ;;
  esac
  for forbidden in "${FORBIDDEN_ARTIFACT_ROOTS[@]}"; do
    case "$resolved" in
      "$forbidden"|"$forbidden"/*)
        echo "ERROR: $name overlaps a forbidden artifact tree: $forbidden" >&2
        exit 2
        ;;
    esac
    case "$forbidden" in
      "$resolved"/*)
        echo "ERROR: $name is too broad and contains a forbidden artifact tree: $forbidden" >&2
        exit 2
        ;;
    esac
  done
done

resolved_experiment="$(realpath -m "$V3G_EXPERIMENT_ROOT")"
for named_path in \
  "PROPOSAL_ROOT:$PROPOSAL_ROOT" \
  "OUTPUT_ROOT:$OUTPUT_ROOT" \
  "LOG_DIR:$LOG_DIR" \
  "LOCK_FILE:$LOCK_FILE"; do
  name="${named_path%%:*}"
  value="${named_path#*:}"
  resolved="$(realpath -m "$value")"
  case "$resolved" in
    "$resolved_experiment"/*) ;;
    *)
      echo "ERROR: $name must be contained by V3G_EXPERIMENT_ROOT." >&2
      exit 2
      ;;
  esac
done

exec bash "$REPO_ROOT/dataset/run_v3e_consensus_parallel.sh" "$@"
