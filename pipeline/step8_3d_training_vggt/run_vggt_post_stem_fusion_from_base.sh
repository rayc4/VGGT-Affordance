#!/usr/bin/env bash
# Controlled post-stem VGGT fusion initialized from the mask-only base.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export MODEL_CONFIG=cdm_vggt_post_stem_fusion
export EXP_NAME="${EXP_NAME:-weighted-vggt-post-stem-fusion}"

exec "$REPO_ROOT/pipeline/step8_3d_training_vggt/run_vggt_early_fusion_from_base.sh" "$@"
