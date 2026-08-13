#!/usr/bin/env bash
# Train the recommended dense DPT point-head feature adapter on a fixed base.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_CONFIG="${MODEL_CONFIG:-cdm_vggt_adapter_dpt_frozen}"
export EXP_NAME="${EXP_NAME:-dpt-pointlatent-conf-frozen}"
export VGGT_FEATURE_NAME="${VGGT_FEATURE_NAME:-vggt_dpt_feat.npy}"

exec "$SCRIPT_DIR/run_vggt_adapter_from_base.sh" "$@"
