# This code is based on https://github.com/GuyTevet/motion-diffusion-model
import os
import functools
import json
import math
import re
import torch
import torch.distributed as dist
import torch.nn as nn
from loguru import logger

from utils.io import Board, time_str
from utils.metrics import MAP_METRIC_VERSION, compute_average_precision
from models.diffusion.resample import uniform_sampling

class TrainLoop:
    def __init__(self, *, cfg, model, diffusion, dataloader, **kwargs) -> None:
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader

        self.lr = cfg.lr
        self.max_steps = cfg.max_steps
        self.max_epochs = cfg.max_steps // len(self.dataloader) + 1
        self.log_every_step = cfg.log_every_step
        self.save_every_step = cfg.save_every_step

        self.resume_checkpoint = cfg.resume_ckpt
        self.weight_decay = cfg.weight_decay
        self.lr_anneal_steps = cfg.lr_anneal_steps
        
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else '/tmp'
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0
        self.is_distributed = kwargs['is_distributed'] if 'is_distributed' in kwargs else False

        self.step = 1
        self.resume_step = self._load_and_sync_parameters()

        ## set optimizer
        params = []
        nparams = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                params.append(p)
                nparams.append(p.nelement())
                if self.gpu == 0:
                    logger.info(f'Add {n} {p.shape} for optimization.')
        if self.gpu == 0:
            logger.info(f'{len(params)} parameters for optimization.')
            logger.info(f'Total model size is {(sum(nparams) / 1e6):.2f} M.')
        
        self.optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )
        if self.resume_step:
            self.step = self.resume_step + 1
            self._load_optimizer_state()
        
    def _load_and_sync_parameters(self):
        """ Load model from checkpoint if provided for resuming. """
        def parse_resume_step_from_filename(path):
            filename = os.path.basename(path)
            return int(filename.replace('.pt', '').replace('model', ''))
        
        resume_step = 0
        if self.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            load_ckpt(self.model, self.resume_checkpoint)
            if self.gpu == 0:
                logger.info(f"Loading model from checkpoint: {self.resume_checkpoint}...")
            
        return resume_step
        
    def _load_optimizer_state(self):
        """ Load optimizer state from checkpoint if provided for resuming. """
        opt_checkpoint = os.path.join(
            os.path.dirname(self.resume_checkpoint),
            "opt.pt"
        )
        
        if os.path.exists(opt_checkpoint):
            self.optimizer.load_state_dict(
                torch.load(opt_checkpoint)
            )
            if self.gpu == 0:
                logger.info(f"Loading optimizer state from checkpoint: {opt_checkpoint}...")

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _save(self):
        """ Save model and optimizer state. """
        saved_state_dict = {}
        model_state_dict = self.model.state_dict()
        for key in model_state_dict:
            if 'scene_model' in key or 'clip_model' in key or 'text_model' in key or 'bert_model' in key:
                continue

            saved_state_dict[key] = model_state_dict[key]
        
        with open(os.path.join(self.save_dir, f"model{self.step:06d}.pt"), "wb") as f:
            torch.save(saved_state_dict, f)

        with open(os.path.join(self.save_dir, f"opt.pt"), "wb") as f: # only save the last optimizer state for saving space
            torch.save(self.optimizer.state_dict(), f)
        
        if self.gpu == 0:
            logger.info(f'Model saved! [Step: {self.step:06d}]')
    
    def _freeze_scene_model_batchnorm(self):
        """ Freeze batchnorm in scene model if the model has scene model. """
        if hasattr(self.model, 'scene_model') and self.model.freeze_scene_model :
            for m in self.model.scene_model.modules():
                if isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d):
                    m.eval()

    def run_loop(self):
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            self._freeze_scene_model_batchnorm()
            if self.is_distributed:
                self.dataloader.sampler.set_epoch(epoch)
            for it, data in enumerate(self.dataloader): 
                x = data['x'].to(self.device)

                x_kwargs = {}
                if 'x_mask' in data:
                    x_kwargs['x_mask'] = data['x_mask'].to(self.device)
                
                for key in data:
                    if key.startswith('c_') :
                        if torch.is_tensor(data[key]):
                            x_kwargs[key] = data[key].to(self.device)
                        else:
                            x_kwargs[key] = data[key]

                ## one step optimization
                self.optimizer.zero_grad()

                t = uniform_sampling(x.shape[0], self.device, self.diffusion.num_timesteps)
                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model,
                    x,
                    t,
                    model_kwargs=x_kwargs,
                    epoch=epoch
                )
                terms = compute_losses()
                loss = terms['loss'].mean()
                loss.backward()

                self.optimizer.step()
                self._anneal_lr()
                
                ## log and save
                ## log with loguru, plot with Board
                if self.gpu == 0 and self.step % self.log_every_step == 0:
                    ## log with loguru
                    losses = {key: terms[key].mean().item() for key in terms}

                    logger.info(
                        f"[TRAIN] ==> Epoch: {epoch:3d} | Iter: {it+1:5d} | Step: {self.step:7d} | Loss: {losses['loss']:8.5f}"
                    )

                    ## plot with Board
                    write_dict = {'step': self.step, 'train/epoch': epoch}
                    for key in losses:
                        write_dict[f'train/{key}'] = losses[key]
                    Board().write(write_dict)

                if self.gpu == 0 and self.step % self.save_every_step == 0:
                    ## save model
                    self._save()
                
                ## update step and check max steps
                self.step += 1
                if self.step > self.max_steps:
                    return

class CVAETrainLoop:
    def __init__(self, *, cfg, model, dataloader, **kwargs) -> None:
        """ Customized training loop for HUMANISE CVAE
        """
        self.model = model
        self.dataloader = dataloader

        self.lr = cfg.lr
        self.max_steps = cfg.max_steps
        self.max_epochs = cfg.max_steps // len(self.dataloader) + 1
        self.log_every_step = cfg.log_every_step
        self.save_every_step = cfg.save_every_step

        self.resume_checkpoint = cfg.resume_ckpt
        self.weight_decay = cfg.weight_decay
        self.lr_anneal_steps = cfg.lr_anneal_steps
        
        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else '/tmp'
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0

        self.step = 1
        self.resume_step = self._load_and_sync_parameters()

        ## set optimizer
        tune_params, train_params = [], []
        nparams = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                if 'scene_model' in n:
                    tune_params.append(p)
                else:
                    train_params.append(p)
                nparams.append(p.nelement())
                if self.gpu == 0:
                    logger.info(f'Add {n} {p.shape} for optimization.')

        if self.gpu == 0:
            logger.info(f'{len(tune_params) + len(train_params)} parameters for optimization.')
            logger.info(f'Total model size is {(sum(nparams) / 1e6):.2f} M.')
        
        self.optimizer = torch.optim.Adam(
            [
                {'params': tune_params, 'lr': self.lr * 0.1},
                {'params': train_params}
            ],
            lr=self.lr
        )
        if self.resume_step:
            self.step = self.resume_step + 1
            self._load_optimizer_state()
        
    def _load_and_sync_parameters(self):
        """ Load model from checkpoint if provided for resuming. """
        def parse_resume_step_from_filename(path):
            filename = os.path.basename(path)
            return int(filename.replace('.pt', '').replace('model', ''))
        
        resume_step = 0
        if self.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            load_ckpt(self.model, self.resume_checkpoint)
            if self.gpu == 0:
                logger.info(f"Loading model from checkpoint: {self.resume_checkpoint}...")
            
        return resume_step
        
    def _load_optimizer_state(self):
        """ Load optimizer state from checkpoint if provided for resuming. """
        opt_checkpoint = os.path.join(
            os.path.dirname(self.resume_checkpoint),
            "opt.pt"
        )
        
        if os.path.exists(opt_checkpoint):
            self.optimizer.load_state_dict(
                torch.load(opt_checkpoint)
            )
            if self.gpu == 0:
                logger.info(f"Loading optimizer state from checkpoint: {opt_checkpoint}...")

    def _anneal_lr(self):
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _save(self):
        """ Save model and optimizer state. """
        saved_state_dict = {}
        model_state_dict = self.model.state_dict()
        for key in model_state_dict:
            if 'clip_model' in key or 'text_model' in key or 'bert_model' in key:
                continue

            saved_state_dict[key] = model_state_dict[key]
        
        with open(os.path.join(self.save_dir, f"model{self.step:06d}.pt"), "wb") as f:
            torch.save(saved_state_dict, f)

        with open(os.path.join(self.save_dir, f"opt.pt"), "wb") as f: # only save the last optimizer state for saving space
            torch.save(self.optimizer.state_dict(), f)
        
        if self.gpu == 0:
            logger.info(f'Model saved! [Step: {self.step:06d}]')

    def run_loop(self):
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()
            for it, data in enumerate(self.dataloader): 
                x = data['x'].to(self.device)
                print(x.shape)

                x_kwargs = {}
                if 'x_mask' in data:
                    x_kwargs['x_mask'] = data['x_mask'].to(self.device)
                
                for key in data:
                    if key.startswith('c_') :
                        if torch.is_tensor(data[key]):
                            x_kwargs[key] = data[key].to(self.device)
                        else:
                            x_kwargs[key] = data[key]

                ## one step optimization
                self.optimizer.zero_grad()

                terms = self.model.compute_losses(x, x_kwargs)
                loss = terms['loss'].mean()
                loss.backward()

                self.optimizer.step()
                self._anneal_lr()
                
                ## log and save
                ## log with loguru, plot with Board
                if self.gpu == 0 and self.step % self.log_every_step == 0:
                    ## log with loguru
                    losses = {key: terms[key].mean().item() for key in terms}

                    logger.info(
                        f"[TRAIN] ==> Epoch: {epoch:3d} | Iter: {it+1:5d} | Step: {self.step:7d} | Loss: {losses['loss']:8.5f}"
                    )

                    ## plot with Board
                    write_dict = {'step': self.step, 'train/epoch': epoch}
                    for key in losses:
                        write_dict[f'train/{key}'] = losses[key]
                    Board().write(write_dict)

                if self.gpu == 0 and self.step % self.save_every_step == 0:
                    ## save model
                    self._save()
                
                ## update step and check max steps
                self.step += 1
                if self.step > self.max_steps:
                    return

def load_ckpt(model: torch.nn.Module, path: str, map_location='cpu') -> None:
    """ Load checkpoint for model

    Args:
        model: current model
        path: save path
        map_location: device to load checkpoint (default 'cpu' to avoid GPU OOM)
    """
    assert os.path.exists(path), 'Can\'t find provided ckpt.'

    saved_state_dict = torch.load(path, map_location=map_location)
    model_state_dict = model.state_dict()

    unchanged_weights = []
    used_weights = []
    for key in model_state_dict:
        ## current state and saved state both on single GPU or both on multi GPUs 
        if key in saved_state_dict:
            model_state_dict[key] = saved_state_dict[key]
            logger.info(f'Load parameter {key} for current model.')
            used_weights.append(key)
        
        ## current state on single GPU and saved state on multi GPUs
        if 'module.'+key in saved_state_dict:
            model_state_dict[key] = saved_state_dict['module.'+key]
            logger.info(f'Load parameter module.{key} for current model [Trained on multi GPUs].')
            used_weights.append('module.'+key)
        
        if key not in saved_state_dict and 'module.'+key not in saved_state_dict:
            unchanged_weights.append(key)

    unused_weights = []
    for key in saved_state_dict:
        if key not in used_weights:
            unused_weights.append(key)

    for key in unchanged_weights:
        logger.info(f'Unchanged_weight: {key}')
    
    for key in unused_weights:
        logger.info(f'Unused_weight: {key}')
    
    model.load_state_dict(model_state_dict)

class PointCloudMaskRefinementTrainLoop:
    """Training loop for point cloud mask refinement.

    - Conditions on point cloud geometry to optimize mask quality.
    - Supports progressive mask optimization.
    - Keeps point cloud structure fixed, only adjusts mask labels.
    """

    def __init__(self, *, cfg, model, diffusion, dataloader, **kwargs) -> None:
        self.model = model
        self.diffusion = diffusion
        self.dataloader = dataloader

        self.lr = cfg.lr
        self.max_steps = cfg.max_steps
        self.max_epochs = cfg.max_steps // len(self.dataloader) + 1
        self.log_every_step = cfg.log_every_step
        self.save_every_step = cfg.save_every_step

        self.resume_checkpoint = cfg.resume_ckpt
        self.weight_decay = cfg.weight_decay
        self.lr_anneal_steps = cfg.lr_anneal_steps

        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else '/tmp'
        self.gpu = kwargs['gpu'] if 'gpu' in kwargs else 0
        self.is_distributed = kwargs['is_distributed'] if 'is_distributed' in kwargs else False

        self.mask_loss_weight = getattr(cfg, 'mask_loss_weight', 1.0)
        self.geometry_loss_weight = getattr(cfg, 'geometry_loss_weight', 0.1)
        self.boundary_loss_weight = getattr(cfg, 'boundary_loss_weight', 0.5)

        self.step = 1
        self.resume_step = self._load_and_sync_parameters()

        params = []
        nparams = []
        for n, p in model.named_parameters():
            if p.requires_grad:
                params.append(p)
                nparams.append(p.nelement())
                if self.gpu == 0:
                    logger.info(f'Add {n} {p.shape} for mask refinement optimization.')

        if self.gpu == 0:
            logger.info(f'{len(params)} parameters for mask refinement optimization.')
            logger.info(f'Total model size is {(sum(nparams) / 1e6):.2f} M.')

        self.optimizer = torch.optim.AdamW(
            params, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self.step = self.resume_step + 1
            self._load_optimizer_state()

    def _load_and_sync_parameters(self):
        """Load model checkpoint."""
        def parse_resume_step_from_filename(path):
            filename = os.path.basename(path)
            return int(filename.replace('.pt', '').replace('model', ''))
        
        resume_step = 0
        if self.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            load_ckpt(self.model, self.resume_checkpoint)
            if self.gpu == 0:
                logger.info(f"Loading mask refinement model from checkpoint: {self.resume_checkpoint}...")
            
        return resume_step
    
    def _load_optimizer_state(self):
        """Load optimizer state."""
        opt_checkpoint = os.path.join(
            os.path.dirname(self.resume_checkpoint),
            "opt.pt"
        )
        
        if os.path.exists(opt_checkpoint):
            self.optimizer.load_state_dict(
                torch.load(opt_checkpoint)
            )
            if self.gpu == 0:
                logger.info(f"Loading optimizer state from checkpoint: {opt_checkpoint}...")

    def _anneal_lr(self):
        """Learning rate annealing."""
        if not self.lr_anneal_steps:
            return
        frac_done = (self.step + self.resume_step) / self.lr_anneal_steps
        lr = self.lr * (1 - frac_done)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def _save(self):
        """Save model and optimizer state."""
        saved_state_dict = {}
        model_state_dict = self.model.state_dict()

        for key in model_state_dict:
            saved_state_dict[key] = model_state_dict[key]
        
        with open(os.path.join(self.save_dir, f"mask_refinement_model{self.step:06d}.pt"), "wb") as f:
            torch.save(saved_state_dict, f)

        with open(os.path.join(self.save_dir, f"mask_refinement_opt.pt"), "wb") as f:
            torch.save(self.optimizer.state_dict(), f)
        
        if self.gpu == 0:
            logger.info(f'Mask refinement model saved! [Step: {self.step:06d}]')

    def _compute_mask_refinement_loss(self, pred_mask, gt_mask, point_cloud, initial_mask):
        """Compute composite mask refinement loss.

        Args:
            pred_mask: predicted mask [B, N]
            gt_mask: ground truth mask [B, N]
            point_cloud: point cloud coords [B, N, 3]
            initial_mask: initial mask [B, N]
        """
        mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_mask, gt_mask.float()
        )

        geometry_loss = self._compute_geometry_consistency_loss(
            pred_mask, point_cloud
        )

        boundary_loss = self._compute_boundary_loss(
            pred_mask, initial_mask, point_cloud
        )

        smoothness_loss = self._compute_smoothness_loss(pred_mask, point_cloud)

        total_loss = (
            self.mask_loss_weight * mask_loss +
            self.geometry_loss_weight * geometry_loss +
            self.boundary_loss_weight * boundary_loss +
            0.1 * smoothness_loss
        )
        
        return {
            'loss': total_loss,
            'mask_loss': mask_loss,
            'geometry_loss': geometry_loss,
            'boundary_loss': boundary_loss,
            'smoothness_loss': smoothness_loss
        }
    
    def _compute_geometry_consistency_loss(self, pred_mask, point_cloud):
        """Compute geometry consistency loss (neighbors should have similar mask values)."""
        batch_size, num_points, _ = point_cloud.shape

        dist_matrix = torch.cdist(point_cloud, point_cloud)

        K = 8
        _, knn_indices = torch.topk(dist_matrix, k=K+1, dim=-1, largest=False)
        knn_indices = knn_indices[:, :, 1:]

        pred_probs = torch.sigmoid(pred_mask)
        consistency_loss = 0
        
        for i in range(batch_size):
            for j in range(num_points):
                neighbor_indices = knn_indices[i, j]
                neighbor_masks = pred_probs[i, neighbor_indices]
                center_mask = pred_probs[i, j]

                diff = torch.abs(neighbor_masks - center_mask)
                consistency_loss += torch.mean(diff)
        
        return consistency_loss / (batch_size * num_points)
    
    def _compute_boundary_loss(self, pred_mask, initial_mask, point_cloud):
        """Compute boundary refinement loss (smooth mask transition near boundaries)."""
        pred_probs = torch.sigmoid(pred_mask)

        boundary_loss = torch.nn.functional.mse_loss(
            pred_probs, initial_mask.float()
        )
        
        return boundary_loss
    
    def _compute_smoothness_loss(self, pred_mask, point_cloud):
        """Compute smoothness loss."""
        pred_probs = torch.sigmoid(pred_mask)

        smoothness_loss = torch.mean(torch.abs(
            pred_probs[:, 1:] - pred_probs[:, :-1]
        ))
        
        return smoothness_loss

    def run_loop(self):
        """Mask refinement training loop."""
        for epoch in range(1, self.max_epochs + 1):
            self.model.train()

            if self.is_distributed:
                self.dataloader.sampler.set_epoch(epoch)

            for it, data in enumerate(self.dataloader):
                point_cloud = data['point_cloud'].to(self.device)
                initial_mask = data['initial_mask'].to(self.device)
                gt_mask = data['gt_mask'].to(self.device)

                x_kwargs = {
                    'point_cloud': point_cloud,
                    'initial_mask': initial_mask
                }

                self.optimizer.zero_grad()

                t = uniform_sampling(point_cloud.shape[0], self.device, self.diffusion.num_timesteps)

                compute_losses = functools.partial(
                    self.diffusion.training_losses,
                    self.model,
                    gt_mask,
                    t,
                    model_kwargs=x_kwargs,
                    epoch=epoch
                )
                
                terms = compute_losses()

                pred_mask = self.model(gt_mask, t, **x_kwargs)
                refinement_losses = self._compute_mask_refinement_loss(
                    pred_mask, gt_mask, point_cloud, initial_mask
                )

                total_loss = terms['loss'].mean() + 0.1 * refinement_losses['loss']

                total_loss.backward()
                self.optimizer.step()
                self._anneal_lr()

                if self.gpu == 0 and self.step % self.log_every_step == 0:
                    losses = {key: terms[key].mean().item() for key in terms}
                    refinement_losses_dict = {
                        key: refinement_losses[key].item() for key in refinement_losses
                    }
                    
                    logger.info(
                        f"[MASK REFINEMENT] ==> Epoch: {epoch:3d} | Iter: {it+1:5d} | "
                        f"Step: {self.step:7d} | Loss: {losses['loss']:8.5f} | "
                        f"Refinement Loss: {refinement_losses_dict['loss']:8.5f}"
                    )

                    write_dict = {
                        'step': self.step, 
                        'train/epoch': epoch,
                        'train/total_loss': total_loss.item()
                    }
                    
                    for key in losses:
                        write_dict[f'train/{key}'] = losses[key]
                    for key in refinement_losses_dict:
                        write_dict[f'train/refinement_{key}'] = refinement_losses_dict[key]
                    
                    Board().write(write_dict)

                if self.gpu == 0 and self.step % self.save_every_step == 0:
                    self._save()

                self.step += 1
                if self.step > self.max_steps:
                    return

class SimpleMaskRefinementTrainLoop:
    """Simple point cloud mask refinement training loop (no diffusion).

    - Direct model forward, no diffusion.
    - Conditions on point cloud geometry to optimize mask quality.
    - Supports multiple loss combinations.
    - Keeps point cloud structure fixed, only adjusts mask labels.
    """

    def __init__(self, *, cfg, model, dataloader, val_dataloader=None, **kwargs) -> None:
        self.model = model
        self.dataloader = dataloader
        self.val_dataloader = val_dataloader

        self.lr = cfg.lr
        self.lr_multipliers = {
            str(pattern): float(multiplier)
            for pattern, multiplier in cfg.get('lr_multipliers', {}).items()
        }
        if any(multiplier <= 0 for multiplier in self.lr_multipliers.values()):
            raise ValueError('all lr_multipliers must be positive')
        self.base_freeze_steps = int(cfg.get('base_freeze_steps', 0))
        if self.base_freeze_steps < 0:
            raise ValueError('base_freeze_steps cannot be negative')
        self.max_steps = cfg.max_steps
        self.max_epochs = cfg.max_steps // len(self.dataloader) + 1
        self.log_every_step = cfg.log_every_step

        self.resume_checkpoint = cfg.resume_ckpt
        self.weight_decay = cfg.weight_decay
        self.lr_anneal_steps = cfg.lr_anneal_steps
        self.validate_every_epoch = int(cfg.get('validate_every_epoch', 1))
        self.validate_before_training = bool(
            cfg.get('validate_before_training', False)
        )
        self.early_stopping_patience = int(cfg.get('early_stopping_patience', 8))
        self.early_stopping_min_delta = float(cfg.get('early_stopping_min_delta', 1e-4))
        self.early_stopping_warmup_epochs = int(cfg.get('early_stopping_warmup_epochs', 5))
        self.early_stopping_metric = str(cfg.get('early_stopping_metric', 'mAP'))
        self.early_stopping_mode = str(cfg.get('early_stopping_mode', 'max'))
        ap50_floor = cfg.get('best_checkpoint_ap50_floor', None)
        self.best_checkpoint_ap50_floor = (
            None if ap50_floor is None else float(ap50_floor)
        )
        self.bce_loss_weight = float(cfg.get('bce_loss_weight', 0.3))
        self.dice_loss_weight = float(cfg.get('dice_loss_weight', 0.3))
        self.focal_loss_weight = float(cfg.get('focal_loss_weight', 0.2))
        self.iou_loss_weight = float(cfg.get('iou_loss_weight', 0.2))
        self.focal_alpha = float(cfg.get('focal_alpha', 0.75))
        self.focal_gamma = float(cfg.get('focal_gamma', 2.0))
        self.validation_threshold = float(cfg.get('validation_threshold', 0.5))
        self.loss_weight_sum = sum((
            self.bce_loss_weight,
            self.dice_loss_weight,
            self.focal_loss_weight,
            self.iou_loss_weight,
        ))
        if self.validate_every_epoch < 1:
            raise ValueError('validate_every_epoch must be at least 1')
        if self.early_stopping_patience < 0:
            raise ValueError('early_stopping_patience cannot be negative')
        if self.early_stopping_mode not in ('min', 'max'):
            raise ValueError("early_stopping_mode must be 'min' or 'max'")
        if (
            self.best_checkpoint_ap50_floor is not None
            and not 0.0 <= self.best_checkpoint_ap50_floor <= 1.0
        ):
            raise ValueError('best_checkpoint_ap50_floor must be in [0, 1]')
        if min(
            self.bce_loss_weight,
            self.dice_loss_weight,
            self.focal_loss_weight,
            self.iou_loss_weight,
        ) < 0 or self.loss_weight_sum <= 0:
            raise ValueError('loss weights must be non-negative and have a positive sum')
        if not 0.0 <= self.focal_alpha <= 1.0:
            raise ValueError('focal_alpha must be in [0, 1]')
        if self.focal_gamma < 0:
            raise ValueError('focal_gamma cannot be negative')
        if not 0.0 <= self.validation_threshold <= 1.0:
            raise ValueError('validation_threshold must be in [0, 1]')

        self.device = kwargs['device'] if 'device' in kwargs else 'cpu'
        self.save_dir = kwargs['save_dir'] if 'save_dir' in kwargs else '/tmp'
        self.eval_dir = kwargs.get(
            'eval_dir', os.path.join(os.path.dirname(self.save_dir), 'eval')
        )
        self.is_main = kwargs['is_main'] if 'is_main' in kwargs else True
        self.is_distributed = kwargs['is_distributed'] if 'is_distributed' in kwargs else False
        if self.early_stopping_patience and self.val_dataloader is None:
            raise ValueError('early stopping requires a validation dataloader')
        if self.validate_before_training and self.val_dataloader is None:
            raise ValueError('pre-training validation requires a validation dataloader')
        if (
            self.best_checkpoint_ap50_floor is not None
            and self.val_dataloader is None
        ):
            raise ValueError(
                'best_checkpoint_ap50_floor requires a validation dataloader'
            )

        self.best_val_metric = (
            float('inf') if self.early_stopping_mode == 'min' else -float('inf')
        )
        self.best_step = 0
        self.best_epoch = 0
        self.best_constraint_value = None
        self.has_best_checkpoint = False
        self.bad_validation_count = 0

        self.step = 1
        self.start_epoch = 1
        self.resume_step = self._load_and_sync_parameters()

        params = []
        nparams = []
        params_by_multiplier = {}
        for n, p in model.named_parameters():
            if p.requires_grad:
                params.append(p)
                nparams.append(p.nelement())
                normalized_name = n[7:] if n.startswith('module.') else n
                matches = [
                    (pattern, multiplier)
                    for pattern, multiplier in self.lr_multipliers.items()
                    if pattern in normalized_name
                ]
                if len(matches) > 1:
                    raise ValueError(
                        f'{normalized_name} matches multiple lr_multipliers: '
                        f'{[pattern for pattern, _ in matches]}'
                    )
                multiplier = matches[0][1] if matches else 1.0
                params_by_multiplier.setdefault(multiplier, []).append(p)
                if self.is_main:
                    logger.info(
                        f'Add {n} {p.shape} for mask refinement optimization '
                        f'at lr={self.lr * multiplier:g}.'
                    )

        unmatched_patterns = [
            pattern for pattern in self.lr_multipliers
            if not any(
                pattern in (name[7:] if name.startswith('module.') else name)
                for name, parameter in model.named_parameters()
                if parameter.requires_grad
            )
        ]
        if unmatched_patterns:
            raise ValueError(
                f'lr_multipliers matched no trainable parameters: {unmatched_patterns}'
            )
        if self.base_freeze_steps and set(params_by_multiplier) == {1.0}:
            raise ValueError(
                'base_freeze_steps requires at least one non-base LR multiplier '
                'so some parameters can train during warmup'
            )

        if self.is_main:
            logger.info(f'{len(params)} parameters for mask refinement optimization.')
            logger.info(f'Total model size is {(sum(nparams) / 1e6):.2f} M.')
            logger.info(
                'Mask loss weights: '
                f'BCE={self.bce_loss_weight:g}, Dice={self.dice_loss_weight:g}, '
                f'Focal={self.focal_loss_weight:g}, IoU={self.iou_loss_weight:g}; '
                f'focal alpha={self.focal_alpha:g}, gamma={self.focal_gamma:g}'
            )
            if self.val_dataloader is not None:
                constraint_message = ''
                if self.best_checkpoint_ap50_floor is not None:
                    constraint_message = (
                        f'; require val/AP50 >= '
                        f'{self.best_checkpoint_ap50_floor:g} for best-checkpoint eligibility'
                    )
                logger.info(
                    f'Validate every {self.validate_every_epoch} epoch(s); monitor '
                    f'val/{self.early_stopping_metric} ({self.early_stopping_mode}), '
                    f'patience={self.early_stopping_patience}, '
                    f'min_delta={self.early_stopping_min_delta:g}, '
                    f'warmup={self.early_stopping_warmup_epochs} epoch(s)'
                    f'{constraint_message}.'
                )

        optimizer_groups = [
            {
                'params': group_params,
                'lr': self.lr * multiplier,
                'initial_lr': self.lr * multiplier,
                'freeze_until_step': (
                    self.base_freeze_steps if multiplier == 1.0 else 0
                ),
            }
            for multiplier, group_params in sorted(params_by_multiplier.items())
        ]
        self.optimizer = torch.optim.AdamW(
            optimizer_groups, lr=self.lr, weight_decay=self.weight_decay
        )

        if self.resume_step:
            self.step = self.resume_step + 1
            self.start_epoch = self.resume_step // len(self.dataloader) + 1
            self._load_optimizer_state()
            self._load_early_stopping_state()

    def _load_and_sync_parameters(self):
        """Load model checkpoint."""
        def parse_resume_step_from_filename(path):
            filename = os.path.basename(path)
            match = re.search(r'model(\d+)\.pt$', filename)
            return int(match.group(1)) if match else 0
        
        resume_step = 0
        if self.resume_checkpoint:
            resume_step = parse_resume_step_from_filename(self.resume_checkpoint)
            load_ckpt(self.model, self.resume_checkpoint)
            if self.is_main:
                logger.info(f"Loading mask refinement model from checkpoint: {self.resume_checkpoint}...")
            
        return resume_step

    def _load_optimizer_state(self):
        """Load optimizer state."""
        checkpoint_dir = os.path.dirname(self.resume_checkpoint)
        opt_checkpoint = os.path.join(checkpoint_dir, "mask_refinement_opt.pt")
        if not os.path.exists(opt_checkpoint):
            # Backward compatibility with the original filename.
            opt_checkpoint = os.path.join(checkpoint_dir, "opt.pt")

        if os.path.exists(opt_checkpoint):
            self.optimizer.load_state_dict(
                torch.load(opt_checkpoint)
            )
            if self.is_main:
                logger.info(f"Loading optimizer state from checkpoint: {opt_checkpoint}...")

    def _load_early_stopping_state(self):
        state_path = os.path.join(
            os.path.dirname(self.resume_checkpoint), "early_stopping_state.pt"
        )
        if not os.path.exists(state_path):
            return
        state = torch.load(state_path, map_location='cpu')
        state_step = int(state.get('step', 0))
        if state_step > self.resume_step:
            if self.is_main:
                logger.warning(
                    f"Ignoring early stopping state from step {state_step} because "
                    f"the resumed checkpoint is older (step {self.resume_step})."
                )
            return
        state_monitor = state.get('monitor')
        state_mode = state.get('mode')
        if (
            state_monitor != self.early_stopping_metric
            or state_mode != self.early_stopping_mode
        ):
            if self.is_main:
                logger.warning(
                    'Ignoring early stopping state for '
                    f'{state_monitor!r} ({state_mode!r}); current monitor is '
                    f'{self.early_stopping_metric!r} ({self.early_stopping_mode!r}).'
                )
            return
        state_ap50_floor = state.get('best_checkpoint_ap50_floor')
        if state_ap50_floor is not None:
            state_ap50_floor = float(state_ap50_floor)
        if state_ap50_floor != self.best_checkpoint_ap50_floor:
            raise ValueError(
                'Cannot resume with a different best_checkpoint_ap50_floor: '
                f'checkpoint state uses {state_ap50_floor!r}, current config uses '
                f'{self.best_checkpoint_ap50_floor!r}. Start a new run with init_ckpt '
                'instead of resume_ckpt.'
            )
        if self.best_checkpoint_ap50_floor is not None:
            state_validation_threshold = state.get(
                'best_checkpoint_validation_threshold'
            )
            if state_validation_threshold is None:
                raise ValueError(
                    'Cannot resume an AP50-constrained run whose state does not '
                    'record best_checkpoint_validation_threshold. Start a new run '
                    'with init_ckpt instead of resume_ckpt.'
                )
            state_validation_threshold = float(state_validation_threshold)
            if state_validation_threshold != self.validation_threshold:
                raise ValueError(
                    'Cannot resume an AP50-constrained run with a different '
                    'validation_threshold: checkpoint state uses '
                    f'{state_validation_threshold!r}, current config uses '
                    f'{self.validation_threshold!r}. Start a new run with init_ckpt '
                    'instead of resume_ckpt.'
                )
        if (
            state_monitor == 'mAP'
            and state.get('metric_versions', {}).get('mAP')
            != MAP_METRIC_VERSION
        ):
            if self.is_main:
                logger.warning(
                    'Ignoring early stopping state produced by the legacy mAP '
                    'calculation; the next validation will establish a new best.'
                )
            return
        self.best_val_metric = float(state.get('best_val_metric', self.best_val_metric))
        self.best_step = int(state.get('best_step', self.best_step))
        self.best_epoch = int(state.get('best_epoch', self.best_epoch))
        self.best_constraint_value = state.get(
            'best_constraint_value', self.best_constraint_value
        )
        if self.best_constraint_value is not None:
            self.best_constraint_value = float(self.best_constraint_value)
        self.has_best_checkpoint = bool(
            state.get('has_best_checkpoint', math.isfinite(self.best_val_metric))
        )
        self.bad_validation_count = int(
            state.get('bad_validation_count', self.bad_validation_count)
        )
        if self.is_main:
            logger.info(
                f"Restored early stopping state: best {self.early_stopping_metric}="
                f"{self.best_val_metric:.6f} at epoch {self.best_epoch}, step {self.best_step}; "
                f"patience {self.bad_validation_count}/{self.early_stopping_patience}"
            )

    def _anneal_lr(self):
        """Learning rate annealing."""
        if not self.lr_anneal_steps:
            return
        frac_done = self.step / self.lr_anneal_steps
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = (
                param_group.get('initial_lr', self.lr)
                * max(0.0, 1 - frac_done)
            )

    def _clear_warmup_gradients(self):
        """Keep base optimizer state untouched during VGGT-only warmup."""
        for param_group in self.optimizer.param_groups:
            if self.step <= int(param_group.get('freeze_until_step', 0)):
                for parameter in param_group['params']:
                    parameter.grad = None

    def _saved_model_state(self):
        saved_state_dict = {}
        model_state_dict = self.model.state_dict()

        for key in model_state_dict:
            if 'clip_model' in key or 'text_model' in key or 'bert_model' in key:
                continue
            saved_state_dict[key] = model_state_dict[key]
        return saved_state_dict

    def _save(self, checkpoint_step=None):
        """Save a numbered model checkpoint and the latest optimizer state."""
        checkpoint_step = self.step if checkpoint_step is None else checkpoint_step
        with open(os.path.join(self.save_dir, f"mask_refinement_model{checkpoint_step:06d}.pt"), "wb") as f:
            torch.save(self._saved_model_state(), f)

        with open(os.path.join(self.save_dir, f"mask_refinement_opt.pt"), "wb") as f:
            torch.save(self.optimizer.state_dict(), f)
        
        if self.is_main:
            logger.info(f'Mask refinement model saved! [Step: {checkpoint_step:06d}]')

    def _save_best(self, epoch, validation_metrics):
        checkpoint_path = os.path.join(self.save_dir, 'mask_refinement_model_best.pt')
        with open(checkpoint_path, 'wb') as f:
            torch.save(self._saved_model_state(), f)
        with open(os.path.join(self.save_dir, 'best_checkpoint.json'), 'w') as f:
            metadata = {
                'checkpoint': os.path.abspath(checkpoint_path),
                'step': self.step - 1,
                'epoch': epoch,
                'monitor': self.early_stopping_metric,
                'mode': self.early_stopping_mode,
                'value': validation_metrics[self.early_stopping_metric],
                'metric_versions': {'mAP': MAP_METRIC_VERSION},
                'validation_metrics': validation_metrics,
            }
            if self.best_checkpoint_ap50_floor is not None:
                metadata['constraints'] = {
                    'AP50': {
                        'minimum': self.best_checkpoint_ap50_floor,
                        'value': validation_metrics['AP50'],
                        'validation_threshold': self.validation_threshold,
                    }
                }
            json.dump(metadata, f, indent=2)
        logger.info(
            f"Saved new best checkpoint at epoch {epoch}, step {self.step - 1}: "
            f"val/{self.early_stopping_metric}="
            f"{validation_metrics[self.early_stopping_metric]:.6f}"
        )

    def _save_early_stopping_state(self):
        torch.save(
            {
                'step': self.step - 1,
                'best_val_metric': self.best_val_metric,
                'best_step': self.best_step,
                'best_epoch': self.best_epoch,
                'best_constraint_value': self.best_constraint_value,
                'has_best_checkpoint': self.has_best_checkpoint,
                'bad_validation_count': self.bad_validation_count,
                'monitor': self.early_stopping_metric,
                'mode': self.early_stopping_mode,
                'best_checkpoint_ap50_floor': self.best_checkpoint_ap50_floor,
                'best_checkpoint_validation_threshold': (
                    self.validation_threshold
                    if self.best_checkpoint_ap50_floor is not None
                    else None
                ),
                'metric_versions': {'mAP': MAP_METRIC_VERSION},
            },
            os.path.join(self.save_dir, 'early_stopping_state.pt'),
        )

    def _compute_mask_refinement_loss(self, pred_mask, gt_mask, point_cloud, initial_mask):
        """Compute composite mask refinement loss.

        Args:
            pred_mask: predicted mask logits [B, N]
            gt_mask: ground truth mask [B, N]
            point_cloud: point cloud coords [B, N, 3]
            initial_mask: initial mask [B, N]
        """
        batch_size, num_points = pred_mask.shape

        mask_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_mask, gt_mask.float()
        )
        pred_probs = torch.sigmoid(pred_mask)
        dice_loss = self._compute_dice_loss(pred_probs, gt_mask.float())
        focal_loss = self._compute_focal_loss(
            pred_mask,
            gt_mask.float(),
            alpha=self.focal_alpha,
            gamma=self.focal_gamma,
        )
        iou_loss = self._compute_iou_loss(pred_probs, gt_mask.float())

        total_loss = (
            self.bce_loss_weight * mask_loss
            + self.dice_loss_weight * dice_loss
            + self.focal_loss_weight * focal_loss
            + self.iou_loss_weight * iou_loss
        ) / self.loss_weight_sum
        
        stats = self._compute_mask_stats(pred_probs, gt_mask.float())

        return {
            'loss': total_loss,
            'mask_loss': mask_loss,
            'dice_loss': dice_loss,
            'focal_loss': focal_loss,
            'iou_loss': iou_loss,
            'pred_mean': stats['pred_mean'],
            'gt_mean': stats['gt_mean'],
            'pred_std': stats['pred_std'],
            'gt_std': stats['gt_std'],
            'positive_ratio': stats['positive_ratio']
        }
    
    def _compute_focal_loss(self, pred_logits, gt_mask, alpha=0.25, gamma=2.0):
        """Compute numerically stable binary focal loss from logits."""
        bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            pred_logits, gt_mask, reduction='none'
        )
        pt = torch.exp(-bce_loss)
        alpha_weight = alpha * gt_mask + (1 - alpha) * (1 - gt_mask)
        focal_loss = alpha_weight * (1 - pt).pow(gamma) * bce_loss
        return focal_loss.mean()
    
    def _compute_iou_loss(self, pred_probs, gt_mask):
        """Compute IoU loss."""
        intersection = torch.sum(pred_probs * gt_mask, dim=1)
        union = torch.sum(pred_probs, dim=1) + torch.sum(gt_mask, dim=1) - intersection
        
        iou = (intersection + 1e-6) / (union + 1e-6)
        iou_loss = 1.0 - torch.mean(iou)
        
        return iou_loss
    
    def _compute_mask_stats(self, pred_probs, gt_mask):
        """Compute mask statistics for debugging."""
        return {
            'pred_mean': torch.mean(pred_probs),
            'gt_mean': torch.mean(gt_mask),
            'pred_std': torch.std(pred_probs),
            'gt_std': torch.std(gt_mask),
            'positive_ratio': torch.mean(gt_mask)
        }

    def _compute_dice_loss(self, pred_probs, gt_mask):
        """Compute Dice loss (Dice = 2*|A∩B| / (|A|+|B|))."""
        intersection = torch.sum(pred_probs * gt_mask, dim=1)
        union = torch.sum(pred_probs, dim=1) + torch.sum(gt_mask, dim=1)
        
        dice = (2.0 * intersection + 1e-6) / (union + 1e-6)
        dice_loss = 1.0 - torch.mean(dice)
        
        return dice_loss

    def _compute_batch_losses(self, data, return_outputs=False):
        x = data['pred_mask_local'].to(self.device).unsqueeze(-1)
        x_kwargs = {}
        for key in data:
            if key.startswith('c_'):
                if torch.is_tensor(data[key]):
                    x_kwargs[key] = data[key].to(self.device)
                else:
                    x_kwargs[key] = data[key]
        gt_mask = data['gt_mask_local'].to(self.device)
        pred_mask = self.model(x, **x_kwargs).squeeze(-1)
        losses = self._compute_mask_refinement_loss(
            pred_mask, gt_mask, x_kwargs['c_pc_xyz'], x.squeeze(-1)
        )
        if return_outputs:
            outputs = [
                (pred_mask[index], gt_mask[index])
                for index in range(pred_mask.shape[0])
            ]
            return losses, outputs
        return losses

    @staticmethod
    def _empty_validation_results():
        """Return the schema written by Segment3DEvaluator.save()."""
        return {
            key: []
            for key in (
                'visit_id', 'annot_id', 'pred_count', 'gt_count',
                'Prc', 'mAP', 'AP25', 'AP50',
                'Rec', 'mAR', 'AR25', 'AR50', 'mIoU',
            )
        }

    def _compute_validation_segmentation_metrics(
        self, frame_outputs, visit_ids, annot_ids
    ):
        """Compute continuous AP and hard-mask evaluator metrics per frame."""
        metric_sums = {
            key: torch.zeros((), device=self.device, dtype=torch.float64)
            for key in (
                'Prc', 'mAP', 'AP25', 'AP50',
                'Rec', 'mAR', 'AR25', 'AR50', 'mIoU',
            )
        }
        recall_thresholds = torch.linspace(
            0.5, 0.95, 10, device=self.device, dtype=torch.float32
        )
        results = self._empty_validation_results()

        for index, (pred_logits, gt_mask) in enumerate(frame_outputs):
            pred_probability = torch.sigmoid(pred_logits.reshape(-1))
            pred_mask = pred_probability > self.validation_threshold
            gt_mask = gt_mask.to(self.device).reshape(-1) > 0.5

            true_positive = torch.logical_and(pred_mask, gt_mask).sum().to(torch.float32)
            pred_positive = pred_mask.sum().to(torch.float32)
            gt_positive = gt_mask.sum().to(torch.float32)
            union = torch.logical_or(pred_mask, gt_mask).sum().to(torch.float32)

            precision = torch.where(
                pred_positive > 0,
                true_positive / pred_positive.clamp_min(1),
                torch.zeros_like(true_positive),
            )
            recall = torch.where(
                gt_positive > 0,
                true_positive / gt_positive.clamp_min(1),
                torch.zeros_like(true_positive),
            )
            iou = torch.where(
                union > 0,
                true_positive / union.clamp_min(1),
                torch.zeros_like(true_positive),
            )
            average_precision = compute_average_precision(
                gt_mask, pred_probability
            )
            ap25 = (precision >= 0.25).to(torch.float32)
            ap50 = (precision >= 0.50).to(torch.float32)
            mean_ar = (
                recall >= recall_thresholds
            ).to(torch.float32).mean()
            ar25 = (recall >= 0.25).to(torch.float32)
            ar50 = (recall >= 0.50).to(torch.float32)

            metric_sums['Prc'] += precision
            metric_sums['mAP'] += average_precision
            metric_sums['AP25'] += ap25
            metric_sums['AP50'] += ap50
            metric_sums['Rec'] += recall
            metric_sums['mAR'] += mean_ar
            metric_sums['AR25'] += ar25
            metric_sums['AR50'] += ar50
            metric_sums['mIoU'] += iou

            results['visit_id'].append(str(visit_ids[index]))
            results['annot_id'].append(str(annot_ids[index]))
            results['pred_count'].append(int(pred_positive.item()))
            results['gt_count'].append(int(gt_positive.item()))
            for key, value in (
                ('Prc', precision),
                ('mAP', average_precision),
                ('AP25', ap25),
                ('AP50', ap50),
                ('Rec', recall),
                ('mAR', mean_ar),
                ('AR25', ar25),
                ('AR50', ar50),
                ('mIoU', iou),
            ):
                results[key].append(float(value.item()))

        return metric_sums, results

    def _gather_validation_results(self, local_results):
        """Gather rank-local rows in dataset order and remove sampler padding."""
        if not self.is_distributed:
            return local_results
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError('distributed validation requires an initialized process group')

        gathered_results = [None] * dist.get_world_size()
        dist.all_gather_object(gathered_results, local_results)

        combined = self._empty_validation_results()
        max_rank_samples = max(
            len(results['visit_id']) for results in gathered_results
        )
        for sample_index in range(max_rank_samples):
            for rank_results in gathered_results:
                if sample_index >= len(rank_results['visit_id']):
                    continue
                for key in combined:
                    combined[key].append(rank_results[key][sample_index])

        dataset_size = len(self.val_dataloader.dataset)
        for key in combined:
            del combined[key][dataset_size:]
        return combined

    def _write_validation_evaluation(self, checkpoint_step, results):
        """Write the result tree produced by eval_all_checkpoints.py."""
        checkpoint_name = f'mask_refinement_model{checkpoint_step:06d}.pt'
        checkpoint_path = os.path.abspath(os.path.join(self.save_dir, checkpoint_name))
        checkpoint_output = os.path.join(
            self.eval_dir, os.path.splitext(checkpoint_name)[0]
        )
        test_dir = os.path.join(checkpoint_output, 'test-' + time_str(Y=False))
        viz_dir = os.path.join(test_dir, 'viz')
        os.makedirs(viz_dir, exist_ok=True)

        results_path = os.path.join(test_dir, 'results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f)
        with open(os.path.join(test_dir, 'metadata.json'), 'w') as f:
            json.dump(
                {
                    'checkpoint': checkpoint_path,
                    'checkpoint_step': checkpoint_step,
                    'metric_versions': {'mAP': MAP_METRIC_VERSION},
                },
                f,
                indent=2,
            )
        with open(os.path.join(test_dir, 'test.log'), 'w') as f:
            f.write('Evaluation produced during in-training validation.\n')
            f.write(f'Load checkpoint from {checkpoint_path}\n')
            f.write(f'Save results to {results_path}\n')
        logger.info(f'Save evaluator-compatible validation results to {results_path}')
        return results_path

    @staticmethod
    def _batch_size(data):
        return data['pred_mask_local'].shape[0]

    def _validate(self, epoch):
        """Compute validation metrics and retain evaluator-compatible rows."""
        self.model.eval()
        totals = None
        sample_count = torch.zeros((), device=self.device, dtype=torch.float64)
        local_results = self._empty_validation_results()

        with torch.no_grad():
            for data in self.val_dataloader:
                losses, frame_outputs = self._compute_batch_losses(
                    data, return_outputs=True
                )
                segmentation_metrics, batch_results = (
                    self._compute_validation_segmentation_metrics(
                        frame_outputs,
                        data['c_visit_id'],
                        data['c_desc_id'],
                    )
                )
                batch_size = self._batch_size(data)
                if totals is None:
                    totals = {
                        key: torch.zeros((), device=self.device, dtype=torch.float64)
                        for key in losses
                    }
                for key, value in losses.items():
                    totals[key] += value.detach().to(torch.float64) * batch_size
                for key in local_results:
                    local_results[key].extend(batch_results[key])
                sample_count += batch_size

        if totals is None:
            raise ValueError('validation dataloader produced no batches')

        loss_names = list(totals)
        reduced = torch.stack([totals[key] for key in loss_names] + [sample_count])
        if self.is_distributed:
            if not dist.is_available() or not dist.is_initialized():
                raise RuntimeError('distributed validation requires an initialized process group')
            dist.all_reduce(reduced, op=dist.ReduceOp.SUM)

        reduced_sample_count = reduced[-1].item()
        if reduced_sample_count == 0:
            raise ValueError('validation dataset contains no samples')
        metrics = {
            key: (reduced[index] / reduced[-1]).item()
            for index, key in enumerate(loss_names)
        }
        validation_results = self._gather_validation_results(local_results)
        total_samples = len(validation_results['visit_id'])
        if total_samples == 0:
            raise ValueError('validation dataset contains no samples')
        for key in segmentation_metrics:
            metrics[key] = sum(validation_results[key]) / total_samples

        if self.early_stopping_metric not in metrics:
            raise KeyError(
                f"early_stopping_metric={self.early_stopping_metric!r} is unavailable; "
                f"choose one of {sorted(metrics)}"
            )

        if self.is_main:
            logger.info(
                f"[VALIDATION] ==> Epoch: {epoch:3d} | Step: {self.step - 1:7d} | "
                f"Frames: {int(total_samples):5d} | Loss: {metrics['loss']:8.5f} | "
                f"BCE: {metrics['mask_loss']:6.4f} | Dice: {metrics['dice_loss']:6.4f} | "
                f"Focal: {metrics['focal_loss']:6.4f} | "
                f"SoftIoU: {1.0 - metrics['iou_loss']:6.4f} | "
                f"Prc: {metrics['Prc']:6.4f} | Rec: {metrics['Rec']:6.4f} | "
                f"mIoU: {metrics['mIoU']:6.4f} | "
                f"mAP: {metrics['mAP']:6.4f} | mAR: {metrics['mAR']:6.4f} | "
                f"AP25: {metrics['AP25']:6.4f} | AP50: {metrics['AP50']:6.4f} | "
                f"AR25: {metrics['AR25']:6.4f} | AR50: {metrics['AR50']:6.4f}"
            )
            Board().write({
                'step': self.step - 1,
                'val/epoch': epoch,
                **{f'val/{key}': value for key, value in metrics.items()},
                'val/soft_iou': 1.0 - metrics['iou_loss'],
            })

        self.model.train()
        return metrics, validation_results

    def _update_early_stopping(self, epoch, validation_metrics):
        current = validation_metrics[self.early_stopping_metric]
        if not math.isfinite(current):
            raise FloatingPointError(
                f"validation metric {self.early_stopping_metric} is not finite: {current}"
            )
        constraint_value = validation_metrics.get('AP50')
        if self.best_checkpoint_ap50_floor is not None:
            if constraint_value is None:
                raise KeyError(
                    'best_checkpoint_ap50_floor is configured, but validation '
                    'metrics do not contain AP50'
                )
            if not math.isfinite(constraint_value):
                raise FloatingPointError(
                    f'validation metric AP50 is not finite: {constraint_value}'
                )
            constraint_satisfied = (
                constraint_value >= self.best_checkpoint_ap50_floor
            )
        else:
            constraint_satisfied = True

        if self.early_stopping_mode == 'min':
            primary_improved = (
                current < self.best_val_metric - self.early_stopping_min_delta
            )
        else:
            primary_improved = (
                current > self.best_val_metric + self.early_stopping_min_delta
            )
        improved = constraint_satisfied and primary_improved

        if improved:
            self.best_val_metric = current
            self.best_step = self.step - 1
            self.best_epoch = epoch
            self.best_constraint_value = (
                float(constraint_value)
                if self.best_checkpoint_ap50_floor is not None
                else None
            )
            self.has_best_checkpoint = True
            self.bad_validation_count = 0
            if self.is_main:
                self._save_best(epoch, validation_metrics)
        elif (
            self.has_best_checkpoint
            and epoch >= self.early_stopping_warmup_epochs
        ):
            self.bad_validation_count += 1

        if self.is_main:
            self._save_early_stopping_state()
            if self.has_best_checkpoint:
                constraint_summary = ''
                if self.best_checkpoint_ap50_floor is not None:
                    constraint_summary = (
                        f', val/AP50={self.best_constraint_value:.6f} '
                        f'(floor {self.best_checkpoint_ap50_floor:.6f})'
                    )
                logger.info(
                    f"[EARLY STOPPING] best val/{self.early_stopping_metric}="
                    f"{self.best_val_metric:.6f}{constraint_summary} at epoch "
                    f"{self.best_epoch}, step {self.best_step}; patience "
                    f"{self.bad_validation_count}/{self.early_stopping_patience}"
                )
            else:
                logger.info(
                    '[EARLY STOPPING] no best checkpoint is eligible yet: '
                    f'val/AP50 must be >= {self.best_checkpoint_ap50_floor:.6f}; '
                    'patience remains inactive.'
                )

        return (
            self.early_stopping_patience > 0
            and epoch >= self.early_stopping_warmup_epochs
            and self.bad_validation_count >= self.early_stopping_patience
        )

    def _warn_if_no_eligible_checkpoint(self):
        if (
            self.is_main
            and self.best_checkpoint_ap50_floor is not None
            and not self.has_best_checkpoint
        ):
            logger.warning(
                'Training ended without an eligible best checkpoint: no '
                f'validation reached AP50 >= {self.best_checkpoint_ap50_floor:.6f} '
                f'at validation_threshold={self.validation_threshold:.6f}. Numbered '
                'checkpoints were retained, but mask_refinement_model_best.pt was '
                'not created by this run.'
            )

    def run_loop(self):
        """Mask refinement training loop."""
        if self.validate_before_training and not self.resume_checkpoint:
            if self.is_main:
                logger.info(
                    '[VALIDATION] ==> Evaluate initialization before optimization.'
                )
            validation_metrics, validation_results = self._validate(epoch=0)
            if self.is_main:
                self._save(checkpoint_step=0)
                self._write_validation_evaluation(0, validation_results)
            self._update_early_stopping(0, validation_metrics)
            if self.is_distributed:
                dist.barrier()

        for epoch in range(self.start_epoch, self.max_epochs + 1):
            self.model.train()

            if self.is_distributed:
                self.dataloader.sampler.set_epoch(epoch)

            reached_max_steps = False
            for it, data in enumerate(self.dataloader):
                self.optimizer.zero_grad()
                losses = self._compute_batch_losses(data)

                total_loss = losses['loss']
                total_loss.backward()
                self._clear_warmup_gradients()
                self.optimizer.step()
                self._anneal_lr()

                if self.is_main and self.step % self.log_every_step == 0:
                    losses_dict = {key: losses[key].item() for key in losses}
                    
                    logger.info(
                        f"[MASK REFINEMENT] ==> Epoch: {epoch:3d} | Iter: {it+1:5d} | "
                        f"Step: {self.step:7d} | Loss: {losses_dict['loss']:8.5f} | "
                        f"BCE: {losses_dict['mask_loss']:6.4f} | Dice: {losses_dict['dice_loss']:6.4f} | "
                        f"Focal: {losses_dict['focal_loss']:6.4f} | IoU: {losses_dict['iou_loss']:6.4f} | "
                        f"PosRatio: {losses_dict['positive_ratio']:4.3f} | "
                        f"PredMean: {losses_dict['pred_mean']:4.3f} | GtMean: {losses_dict['gt_mean']:4.3f}"
                    )

                    write_dict = {
                        'step': self.step, 
                        'train/epoch': epoch,
                        'train/total_loss': total_loss.item()
                    }
                    
                    for key in losses_dict:
                        write_dict[f'train/{key}'] = losses_dict[key]
                    
                    Board().write(write_dict)

                self.step += 1
                if self.step > self.max_steps:
                    reached_max_steps = True
                    break

            should_stop = False
            should_validate = (
                self.val_dataloader is not None
                and (epoch % self.validate_every_epoch == 0 or reached_max_steps)
            )
            if should_validate:
                validation_metrics, validation_results = self._validate(epoch)
                checkpoint_step = self.step - 1
                if self.is_main:
                    self._save(checkpoint_step=checkpoint_step)
                    self._write_validation_evaluation(
                        checkpoint_step, validation_results
                    )
                should_stop = self._update_early_stopping(epoch, validation_metrics)
                if self.is_distributed:
                    dist.barrier()

            if should_stop:
                if self.is_main:
                    logger.info(
                        f"Early stopping at epoch {epoch}, step {self.step - 1}; "
                        f"best checkpoint was epoch {self.best_epoch}, step {self.best_step}."
                    )
                return
            if reached_max_steps:
                self._warn_if_no_eligible_checkpoint()
                return
