import glob
import json
import os
import random
import re

import hydra
import numpy as np
import torch
from loguru import logger
from omegaconf import DictConfig, OmegaConf
from natsort import natsorted
from dataset.AffordanceDataset import AffordanceDataset
from dataset.misc import collate_fn_general
from models.base import create_model_and_diffusion
from utils.io import mkdir_if_not_exists, time_str
from utils.training import load_ckpt
from torch.utils.data import DataLoader
from utils.evaluator import Segment3DEvaluator
from utils.metrics import MAP_METRIC_VERSION

def test(cfg: DictConfig) -> None:
    test_dir = os.path.join(cfg.eval_dir, 'test-' + time_str(Y=False))
    mkdir_if_not_exists(test_dir)
    viz_dir = os.path.join(test_dir, 'viz')
    mkdir_if_not_exists(viz_dir)
    logger.add(os.path.join(test_dir, 'test.log'))
    logger.info('[Configuration]\n' + OmegaConf.to_yaml(cfg) + '\n')
    logger.info('[Test] ==> Begin testing..')

    if cfg.gpu is not None:
        device = f'cuda:{cfg.gpu}'
    else:
        device = 'cpu'
    
    test_dataset = AffordanceDataset(
        root_dir='scenefun3d',
        split='val',
        use_processed_data=False,
        use_division=False,
        use_processed_data_3=False,
        use_sam2=True,
        use_sam2_1=False,
    )
    logger.info(f'Load test dataset size: {len(test_dataset)} frames')

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=cfg.task.test.batch_size,
        collate_fn=collate_fn_general,
        num_workers=cfg.task.test.num_workers,
        shuffle=True,
    )

    model, diffusion = create_model_and_diffusion(cfg, device=device)
    model.to(device)

    if cfg.checkpoint:
        checkpoint = os.path.abspath(os.path.expanduser(str(cfg.checkpoint)))
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError(f'Checkpoint not found: {checkpoint}')
    else:
        best_checkpoint = os.path.join(
            cfg.exp_dir, 'ckpt', 'mask_refinement_model_best.pt'
        )
        if os.path.isfile(best_checkpoint):
            checkpoint = best_checkpoint
        else:
            ckpts = natsorted(glob.glob(
                os.path.join(cfg.exp_dir, 'ckpt', 'mask_refinement_model[0-9]*.pt')
            ))
            assert len(ckpts) > 0, 'No checkpoint found.'
            checkpoint = ckpts[-1]
    load_ckpt(model, checkpoint, map_location=device)
    logger.info(f'Load checkpoint from {checkpoint}')

    exp_tag = f"{cfg.eval_dir}"
    evaluator = Segment3DEvaluator(
        exp_tag=exp_tag,
        viz_dir=viz_dir,
        threshold=cfg.task.train.validation_threshold,
    )

    model.eval()

    for i, data in enumerate(test_dataloader):
        logger.info(f"batch index: {i}, case desc_id: {data['c_desc_id']}")
        x = data['pred_mask_local'].to(device).unsqueeze(-1)

        x_kwargs = {}
        for key in data:
            if key.startswith('c_'):
                if torch.is_tensor(data[key]):
                    x_kwargs[key] = data[key].to(device)
                else:
                    x_kwargs[key] = data[key]
        with torch.no_grad():
            pred_mask = torch.sigmoid(model(x, **x_kwargs).squeeze(-1))

        for batch_index in range(pred_mask.shape[0]):
            evaluator.register(
                [data['c_visit_id'][batch_index]],
                [data['c_desc_id'][batch_index]],
                data['gt_mask_local'][batch_index].to(device).squeeze(),
                pred_mask[batch_index].squeeze(),
                "",
                device,
            )

        if i + 1 >= cfg.task.evaluator.eval_nbatch:
            break

    print(evaluator.get_latex_str())

    with open(os.path.join(test_dir, 'results.json'), 'w') as f:
        evaluator.save(f)

    step_match = re.search(r'model(\d+)\.pt$', os.path.basename(checkpoint))
    checkpoint_step = int(step_match.group(1)) if step_match else None
    if checkpoint_step is None and os.path.basename(checkpoint) == 'mask_refinement_model_best.pt':
        best_info_path = os.path.join(os.path.dirname(checkpoint), 'best_checkpoint.json')
        if os.path.isfile(best_info_path):
            with open(best_info_path, 'r') as f:
                checkpoint_step = json.load(f).get('step')
    with open(os.path.join(test_dir, 'metadata.json'), 'w') as f:
        json.dump({
            'checkpoint': os.path.abspath(checkpoint),
            'checkpoint_step': checkpoint_step,
            'metric_versions': {'mAP': MAP_METRIC_VERSION},
        }, f, indent=2)

    logger.info(f'Save results to {os.path.join(test_dir, "results.json")}')


@hydra.main(version_base=None, config_path="./configs", config_name="default")
def main(cfg: DictConfig) -> None:
    SEED = cfg.seed
    torch.backends.cudnn.benchmark = False     
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    random.seed(SEED)
    np.random.seed(SEED)

    mkdir_if_not_exists(cfg.log_dir)
    mkdir_if_not_exists(cfg.ckpt_dir)
    mkdir_if_not_exists(cfg.eval_dir)

    test(cfg)


if __name__ == '__main__':
    main()
