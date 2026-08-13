#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <experiment-dir> [seed]" >&2
    exit 1
fi
EXP_DIR="$1"
SEED="${2:-2023}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v conda >/dev/null 2>&1; then
    echo "Conda is required to activate the 'pt' environment." >&2
    exit 1
fi
CONDA_BASE="$(conda info --base)"
CONDA_SH="${CONDA_BASE}/etc/profile.d/conda.sh"
if [[ ! -f "$CONDA_SH" ]]; then
    echo "Conda activation script not found: ${CONDA_SH}" >&2
    exit 1
fi
# shellcheck source=/dev/null
set +u
if ! source "$CONDA_SH"; then
    set -u
    echo "Unable to initialize Conda from ${CONDA_SH}." >&2
    exit 1
fi
if conda activate pt; then
    CONDA_ACTIVATED=1
else
    CONDA_ACTIVATED=0
fi
set -u
if [[ "$CONDA_ACTIVATED" -ne 1 ]]; then
    echo "Unable to activate the Conda environment 'pt'." >&2
    exit 1
fi
echo "Conda environment: ${CONDA_DEFAULT_ENV}"

# Select the GPU with GPUS=<index> (or CUDA_VISIBLE_DEVICES)
GPUS="${GPUS:-0,1,2,3}"

cd "$REPO_ROOT"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES="$GPUS" python pipeline/step8_3d_training/eval_no_diff.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_dir="${EXP_DIR}" \
            seed="${SEED}" \
            output_dir=outputs \
            diffusion.steps=500 \
            task=contact_gen \
            model=cdm \
            task.dataset.num_points=8192 \
            task.evaluator.eval_nbatch=32
