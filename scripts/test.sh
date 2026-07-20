EXP_DIR=$1
SEED=$2

if [ -z "$SEED" ]
then
    SEED=2023
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# Select the GPU with GPUS=<index> (or CUDA_VISIBLE_DEVICES)
GPUS="${GPUS:-0,1}"

cd "$REPO_ROOT"

PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES="$GPUS" python pipeline/step8_3d_training/eval_no_diff.py hydra/job_logging=none hydra/hydra_logging=none \
            exp_dir=${EXP_DIR} \
            seed=${SEED} \
            output_dir=outputs \
            diffusion.steps=500 \
            task=contact_gen \
            model=cdm \
            task.dataset.num_points=8192 \
            task.evaluator.eval_nbatch=32 \
