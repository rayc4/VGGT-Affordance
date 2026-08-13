#!/usr/bin/env bash
# Source-frame DPT pixels + source confidence, with a frozen residual adapter.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export MODEL_CONFIG="${MODEL_CONFIG:-cdm_vggt_adapter_dpt_frozen}"
export EXP_NAME="${EXP_NAME:-source-vggt-dpt-conf-frozen}"
export VGGT_FEATURE_NAME="${VGGT_FEATURE_NAME:-vggt_dpt_source_feat.npy}"
export VGGT_CONF_NAME="${VGGT_CONF_NAME:-vggt_source_conf.npy}"
export VGGT_VIEW_COUNT_NAME="${VGGT_VIEW_COUNT_NAME:-vggt_source_view_count.npy}"

exec "$SCRIPT_DIR/run_vggt_adapter_from_base.sh" "$@"
