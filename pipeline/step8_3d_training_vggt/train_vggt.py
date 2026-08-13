"""Step 8 (VGGT variant): train the mask refinement model with lifted VGGT features.

Parallel path to pipeline/step8_3d_training/train_no_diff.py; nothing in the
original path is touched. Requires the per-sample VGGT feature, confidence, and
view-count arrays produced by step7b.

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


def _require_dataset_size(dataset, expected, label):
    """Fail instead of silently changing a controlled ablation's sample set."""
    if expected is None:
        return
    expected = int(expected)
    if len(dataset) != expected:
        raise ValueError(
            f'{label} dataset has {len(dataset)} frames; expected exactly '
            f'{expected}. Check processed samples and VGGT cache completeness.'
        )


def _load_mask_only_init(model, path):
    """Strictly initialize a VGGT model from a compatible checkpoint."""
    saved_state = torch.load(path, map_location='cpu')
    if not isinstance(saved_state, dict):
        raise TypeError(f'{path}: expected a state-dict checkpoint')

    normalized_state = {}
    for source_key, value in saved_state.items():
        key = source_key[7:] if source_key.startswith('module.') else source_key
        if key in normalized_state:
            raise ValueError(f'{path}: duplicate normalized parameter key {key!r}')
        normalized_state[key] = value

    model_state = model.state_dict()

    # A complete VGGT checkpoint can initialize a new optimization stage with
    # fresh optimizer/early-stopping state.
    if normalized_state.keys() == model_state.keys():
        mismatched = sorted(
            key for key in model_state
            if model_state[key].shape != normalized_state[key].shape
        )
        if mismatched:
            raise ValueError(
                f'{path}: full-model shape mismatches: ' + str([
                    (key, tuple(normalized_state[key].shape), tuple(model_state[key].shape))
                    for key in mismatched[:10]
                ])
            )
        model.load_state_dict(normalized_state, strict=True)
        return 'full VGGT model'

    if bool(getattr(model, 'post_stem_fusion', False)):
        new_prefixes = (
            'vggt_proj.',
            'vggt_gate.',
            'vggt_reliability_gate.',
            'scene_model.enc1.0.vggt_fusion.',
        )
        new_keys = {
            key for key in model_state if key.startswith(new_prefixes)
        }
        shared_keys = set(model_state) - new_keys
        missing = sorted(shared_keys - normalized_state.keys())
        unexpected = sorted(normalized_state.keys() - shared_keys)
        mismatched = sorted(
            key for key in shared_keys & normalized_state.keys()
            if model_state[key].shape != normalized_state[key].shape
        )
        if missing or unexpected or mismatched:
            raise ValueError(
                f'{path}: incompatible mask-only post-stem initialization; '
                f'missing={missing[:10]}; unexpected={unexpected[:10]}; '
                f'shape_mismatches={mismatched[:10]}'
            )

        incompatible = model.load_state_dict(
            {key: normalized_state[key] for key in shared_keys}, strict=False
        )
        if set(incompatible.missing_keys) != new_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f'{path}: unexpected post-stem load result: '
                f'missing={incompatible.missing_keys}, '
                f'unexpected={incompatible.unexpected_keys}'
            )
        with torch.no_grad():
            fusion = model.scene_model.enc1[0].vggt_fusion
            fusion.weight.zero_()
            fusion.bias.zero_()
        return 'mask-only model with zero-initialized post-stem VGGT fusion'

    if bool(getattr(model, 'split_input_stem', False)):
        stem_source_key = 'scene_model.enc1.0.linear.weight'
        stem_base_key = 'scene_model.enc1.0.linear.base.weight'
        stem_vggt_key = 'scene_model.enc1.0.linear.vggt.weight'
        vggt_prefixes = (
            'vggt_proj.',
            'vggt_gate.',
            'vggt_reliability_gate.',
        )
        new_keys = {
            key for key in model_state
            if key.startswith(vggt_prefixes) or key == stem_vggt_key
        }
        direct_shared_keys = set(model_state) - new_keys - {stem_base_key}
        expected_source_keys = direct_shared_keys | {stem_source_key}
        missing = sorted(expected_source_keys - normalized_state.keys())
        unexpected = sorted(normalized_state.keys() - expected_source_keys)
        mismatched = sorted(
            key for key in direct_shared_keys
            if key in normalized_state
            and model_state[key].shape != normalized_state[key].shape
        )
        if stem_source_key in normalized_state and (
            model_state[stem_base_key].shape
            != normalized_state[stem_source_key].shape
        ):
            mismatched.append(stem_source_key)
        if missing or unexpected or mismatched:
            raise ValueError(
                f'{path}: incompatible mask-only early-fusion initialization; '
                f'missing={missing[:10]}; unexpected={unexpected[:10]}; '
                f'shape_mismatches={mismatched[:10]}'
            )

        load_state = {
            key: normalized_state[key]
            for key in direct_shared_keys
        }
        load_state[stem_base_key] = normalized_state[stem_source_key]
        incompatible = model.load_state_dict(load_state, strict=False)
        if set(incompatible.missing_keys) != new_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                f'{path}: unexpected early-fusion load result: '
                f'missing={incompatible.missing_keys}, '
                f'unexpected={incompatible.unexpected_keys}'
            )
        with torch.no_grad():
            model.scene_model.enc1[0].linear.vggt.weight.zero_()
        return 'mask-only model with zero-initialized VGGT input columns'

    shared_keys = {
        key for key in model_state if not key.startswith('vggt_adapter.')
    }
    missing = sorted(shared_keys - normalized_state.keys())
    unexpected = sorted(normalized_state.keys() - shared_keys)
    mismatched = sorted(
        key for key in shared_keys & normalized_state.keys()
        if model_state[key].shape != normalized_state[key].shape
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f'missing shared keys: {missing[:10]}')
        if unexpected:
            details.append(f'unexpected keys: {unexpected[:10]}')
        if mismatched:
            details.append(
                'shape mismatches: ' + str([
                    (key, tuple(normalized_state[key].shape), tuple(model_state[key].shape))
                    for key in mismatched[:10]
                ])
            )
        raise ValueError(f'{path}: incompatible mask-only initialization; ' + '; '.join(details))

    incompatible = model.load_state_dict(
        {key: normalized_state[key] for key in shared_keys}, strict=False
    )
    expected_missing = sorted(
        key for key in model_state if key.startswith('vggt_adapter.')
    )
    if sorted(incompatible.missing_keys) != expected_missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f'{path}: unexpected load result: missing={incompatible.missing_keys}, '
            f'unexpected={incompatible.unexpected_keys}'
        )
    return 'mask-only model with zero-initialized residual adapter'


def train(cfg: DictConfig) -> None:
    is_distributed = int(os.environ.get('WORLD_SIZE', '1')) > 1
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))

    if is_distributed:
        device = f'cuda:{local_rank}'
    elif cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'

    train_dataset = AffordanceDatasetVGGT(
        root_dir=cfg.task.dataset.root_dir,
        split=cfg.task.dataset.split,
        use_sam2=True,
        processed_sam2_dir=cfg.task.dataset.processed_sam2_dir,
        vggt_feat_name=cfg.task.dataset.vggt_feat_name,
        vggt_conf_name=cfg.task.dataset.vggt_conf_name,
        vggt_view_count_name=cfg.task.dataset.vggt_view_count_name,
        vggt_feat_root=cfg.task.dataset.vggt_feat_root,
        require_nonempty_gt=cfg.task.dataset.require_nonempty_gt,
        load_vggt=bool(cfg.model.get('use_vggt', True)),
        load_reliability=bool(cfg.model.get('use_reliability', True)),
    )
    logger.info(f'Load train dataset size: {len(train_dataset)} frames')
    if train_dataset.num_skipped_empty_gt:
        logger.info(f'Skipped {train_dataset.num_skipped_empty_gt} frames with an empty '
                    f'gt_mask_local (require_nonempty_gt=True)')
    _require_dataset_size(
        train_dataset, cfg.task.dataset.get('expected_train_frames'), 'train'
    )

    val_dataset = AffordanceDatasetVGGT(
        root_dir=cfg.task.dataset.root_dir,
        split='val',
        use_sam2=True,
        processed_sam2_dir=cfg.task.dataset.processed_sam2_dir,
        vggt_feat_name=cfg.task.dataset.vggt_feat_name,
        vggt_conf_name=cfg.task.dataset.vggt_conf_name,
        vggt_view_count_name=cfg.task.dataset.vggt_view_count_name,
        vggt_feat_root=cfg.task.dataset.vggt_feat_root,
        require_nonempty_gt=False,
        load_vggt=bool(cfg.model.get('use_vggt', True)),
        load_reliability=bool(cfg.model.get('use_reliability', True)),
    )
    logger.info(f'Load validation dataset size: {len(val_dataset)} frames')
    _require_dataset_size(
        val_dataset, cfg.task.dataset.get('expected_val_frames'), 'validation'
    )

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

    model = create_model(cfg, device=device)
    init_ckpt = str(cfg.task.train.get('init_ckpt', '') or '')
    resume_ckpt = str(cfg.task.train.get('resume_ckpt', '') or '')
    if init_ckpt and resume_ckpt:
        raise ValueError('task.train.init_ckpt and resume_ckpt are mutually exclusive')
    if bool(cfg.model.get('freeze_base', False)) and not (init_ckpt or resume_ckpt):
        raise ValueError(
            'model.freeze_base=True requires task.train.init_ckpt or resume_ckpt; '
            'refusing to freeze a randomly initialized base'
        )
    if bool(cfg.model.get('require_init_ckpt', False)) and not (init_ckpt or resume_ckpt):
        raise ValueError(
            'this model configuration requires task.train.init_ckpt or '
            'resume_ckpt; refusing to train without its base initialization'
        )
    if init_ckpt:
        init_ckpt = os.path.abspath(os.path.expanduser(init_ckpt))
        initialization = _load_mask_only_init(model, init_ckpt)
        if rank == 0:
            logger.info(f'Initialized {initialization} from {init_ckpt}')
    if bool(cfg.model.get('freeze_base', False)) and rank == 0:
        logger.info(
            'Adapter-only optimization: PointTransformer and contact head are '
            'frozen in eval mode; only vggt_adapter parameters will train.'
        )
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
        val_dataloader=val_dataloader,
        device=device,
        save_dir=cfg.ckpt_dir,
        eval_dir=cfg.eval_dir,
        is_main=rank == 0,
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
