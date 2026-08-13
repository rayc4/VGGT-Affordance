import os

import hydra
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import numpy as np
import random
from dataset.AffordanceDataset import AffordanceDataset
from dataset.misc import collate_fn_general
from models.base import create_model_and_diffusion
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from utils.io import Board
from utils.training import SimpleMaskRefinementTrainLoop

SPLIT = 'train'


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


def train(cfg: DictConfig, is_distributed: bool, rank: int, local_rank: int) -> None:
    is_main = rank == 0
    processed_sam2_dir = os.environ.get('TASA_PROCESSED_SAM2_DIR')
    if not processed_sam2_dir:
        raise RuntimeError(
            'TASA_PROCESSED_SAM2_DIR is required. Run scripts/train.sh with '
            '<experiment-name> <processed-sam2-dir>.'
        )

    if is_distributed:
        device = f'cuda:{local_rank}'
    elif cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    
    train_dataset = AffordanceDataset(
        root_dir='',
        processed_sam2_dir=processed_sam2_dir,
        split=SPLIT,
        # use_processed_final_train=True
        use_sam2=True,
        require_nonempty_gt=cfg.task.dataset.require_nonempty_gt,
    )
    logger.info(f'Load train dataset size: {len(train_dataset)} frames')
    if train_dataset.num_skipped_empty_gt:
        logger.info(f'Skipped {train_dataset.num_skipped_empty_gt} frames with an empty '
                    f'gt_mask_local (require_nonempty_gt=True)')

    val_dataset = AffordanceDataset(
        root_dir='',
        processed_sam2_dir=processed_sam2_dir,
        split='val',
        use_sam2=True,
        require_nonempty_gt=False,
    )
    logger.info(f'Load validation dataset size: {len(val_dataset)} frames')

    train_sampler = DistributedSampler(train_dataset, shuffle=True) if is_distributed else None
    val_sampler = DistributedSampler(val_dataset, shuffle=False) if is_distributed else None
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.task.train.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.train.num_workers,
        shuffle=train_sampler is None,
        sampler=train_sampler,
    )
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=cfg.task.train.val_batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.train.val_num_workers,
        shuffle=False,
        sampler=val_sampler,
    )

    model, _ = create_model_and_diffusion(cfg, device=device)
    model.to(device)
    if is_distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank], output_device=local_rank)

    SimpleMaskRefinementTrainLoop(
        cfg=cfg.task.train,
        model=model,
        dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        device=device,
        save_dir=cfg.ckpt_dir,
        eval_dir=cfg.eval_dir,
        is_main=is_main,
        is_distributed=is_distributed,
    ).run_loop()

@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    is_distributed, rank, local_rank = _setup_distributed()

    try:
        if rank != 0:
            logger.remove()

        if rank == 0:
            os.makedirs(cfg.log_dir, exist_ok=True)
            os.makedirs(cfg.ckpt_dir, exist_ok=True)
            os.makedirs(cfg.eval_dir, exist_ok=True)

        if is_distributed:
            dist.barrier()

        if rank == 0:
            logger.add(cfg.log_dir + '/runtime.log')
            Board().create_board(cfg.platform, project=cfg.project, log_dir=cfg.log_dir)
            logger.info('[Configuration]\n' + OmegaConf.to_yaml(cfg) + '\n')
            logger.info('[Train] ==> Begin training..')

        train(cfg, is_distributed, rank, local_rank)

        if rank == 0:
            logger.info('[Train] ==> End training..')
            Board().close()
    finally:
        _cleanup_distributed()


if __name__ == '__main__':
    SEED = 2023
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    
    main()
