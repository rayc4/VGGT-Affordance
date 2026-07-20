"""Step 8 (VGGT variant): train the mask refinement model with lifted VGGT features.

Parallel path to pipeline/step8_3d_training/train_no_diff.py; nothing in the
original path is touched. Requires vggt_feat.npy per sample (see step7b).

Single GPU:
    CUDA_VISIBLE_DEVICES=0 python pipeline/step8_3d_training_vggt/train_vggt.py \
        exp_name=vggt_refine task.train.batch_size=64

Multi GPU (one process per GPU via torchrun; task.train.batch_size is per-GPU):
    CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nproc_per_node=4 \
        pipeline/step8_3d_training_vggt/train_vggt.py \
        exp_name=vggt_refine task.train.batch_size=16
"""
import os
import sys

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from dataset.AffordanceDatasetVGGT import AffordanceDatasetVGGT
from dataset.misc import collate_fn_general
from models.base import create_model
import models.cdm_vggt  # noqa: F401 -- registers CDMVGGT
from utils.io import mkdir_if_not_exists, Board
from utils.training import SimpleMaskRefinementTrainLoop


def _setup_distributed():
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    is_distributed = world_size > 1

    if is_distributed:
        if not torch.cuda.is_available():
            raise RuntimeError('Distributed training requires CUDA GPUs.')
        torch.cuda.set_device(local_rank)
        if not dist.is_initialized():
            dist.init_process_group(backend='nccl')

    return is_distributed, rank, local_rank


def _cleanup_distributed():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def train(cfg: DictConfig) -> None:
    is_distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if is_distributed:
        device = f'cuda:{local_rank}'
        log_rank = rank
    elif cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
        log_rank = cfg.gpu
    else:
        device = 'cpu'
        log_rank = 0

    train_dataset = AffordanceDatasetVGGT(
        root_dir=cfg.task.dataset.root_dir,
        split=cfg.task.dataset.split,
        use_processed_final_train=True,
        vggt_feat_name=cfg.task.dataset.vggt_feat_name,
        vggt_feat_root=cfg.task.dataset.vggt_feat_root,
    )
    logger.info(f'Load train dataset size: {len(train_dataset)}')

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.task.train.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.train.num_workers,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )

    model = create_model(cfg, device=device)
    model.to(device)
    if is_distributed:
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
        )

    SimpleMaskRefinementTrainLoop(
        cfg=cfg.task.train,
        model=model,
        dataloader=train_dataloader,
        device=device,
        save_dir=cfg.ckpt_dir,
        gpu=log_rank,
        is_distributed=is_distributed,
    ).run_loop()


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    is_distributed, rank, _ = _setup_distributed()
    board_created = False

    try:
        if rank != 0:
            logger.remove()

        if rank == 0:
            mkdir_if_not_exists(cfg.log_dir)
            mkdir_if_not_exists(cfg.ckpt_dir)
            mkdir_if_not_exists(cfg.eval_dir)

        if is_distributed:
            dist.barrier()

        if rank == 0:
            logger.add(cfg.log_dir + '/runtime.log')
            Board().create_board(cfg.platform, project=cfg.project, log_dir=cfg.log_dir)
            board_created = True

            logger.info('[Configuration]\n' + OmegaConf.to_yaml(cfg) + '\n')
            logger.info('[Train] ==> Begin training..')

        train(cfg)

        if rank == 0:
            logger.info('[Train] ==> End training..')
            if board_created:
                Board().close()
    finally:
        _cleanup_distributed()


if __name__ == '__main__':
    import random
    import numpy as np

    SEED = 2023
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    main()
