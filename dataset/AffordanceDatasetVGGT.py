import os

import numpy as np
import torch
from loguru import logger

from dataset.AffordanceDataset import AffordanceDataset


class AffordanceDatasetVGGT(AffordanceDataset):
    """AffordanceDataset plus per-point VGGT features and reliability signals.

    Expects a mirrored ``vggt_feat.npy`` tree produced by
    ``pipeline/step7b_vggt_feats/extract_vggt_features.py``. ``vggt_feat_root``
    is the common root containing ``train/`` and ``val/``; each split mirrors
    the relative processed-sample layout. Each sample must contain aligned
    feature, confidence, and view-count files; incomplete samples are dropped.

    Only the processed-branch modes are supported (``use_sam2``, ``use_sam2_1``,
    ``use_processed_final_train``), since those are the ones with per-sample
    directories the features live in.

    Also normalizes keys for the refinement loops: ``pred_mask_local`` is
    aliased to ``x`` (the initial lifted mask) when absent, and
    ``c_visit_id``/``c_desc_id`` are filled from the data item.
    """

    def __init__(self, *args, vggt_feat_name='vggt_feat.npy',
                 vggt_conf_name='vggt_conf.npy',
                 vggt_view_count_name='vggt_view_count.npy',
                 vggt_feat_root=None, require_vggt=True, load_vggt=True,
                 load_reliability=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.vggt_feat_name = vggt_feat_name
        self.vggt_conf_name = vggt_conf_name
        self.vggt_view_count_name = vggt_view_count_name
        self.vggt_feat_root = vggt_feat_root
        self.load_vggt = load_vggt
        self.load_reliability = load_reliability

        if not self.data_items or 'base_path' not in self.data_items[0]:
            raise ValueError(
                'AffordanceDatasetVGGT requires a processed-branch mode '
                '(use_sam2 / use_sam2_1 / use_processed_final_train) with per-sample directories.'
            )

        if require_vggt:
            num_total = len(self.data_items)
            self.data_items = [item for item in self.data_items
                               if all(os.path.exists(path)
                                      for path in self._vggt_paths(item).values())]
            num_kept = len(self.data_items)
            if num_kept < num_total:
                logger.warning(
                    f'AffordanceDatasetVGGT: kept {num_kept}/{num_total} samples with '
                    'complete VGGT feature/confidence/view-count files; run step7b '
                    'extraction for the missing ones.'
                )
            if num_kept == 0:
                raise ValueError(
                    f'No sample under {self.processed_dir} has a complete VGGT feature '
                    'bundle; re-run pipeline/step7b_vggt_feats/'
                    'extract_vggt_features.py.'
                )

    def _vggt_path(self, data_item, filename):
        base_path = data_item['base_path']
        if self.vggt_feat_root is not None:
            rel = os.path.relpath(base_path, self.processed_dir)
            return os.path.join(
                self.vggt_feat_root, self.split, rel, filename
            )
        return os.path.join(base_path, filename)

    def _vggt_paths(self, data_item):
        return {
            'feat': self._vggt_path(data_item, self.vggt_feat_name),
            'conf': self._vggt_path(data_item, self.vggt_conf_name),
            'view_count': self._vggt_path(data_item, self.vggt_view_count_name),
        }

    def _feat_path(self, data_item):
        """Backward-compatible helper for callers that only need the feature path."""
        return self._vggt_path(data_item, self.vggt_feat_name)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data_item = self.data_items[idx]

        if not self.load_vggt:
            return data

        paths = self._vggt_paths(data_item)
        feat = np.load(paths['feat']).astype(np.float32)
        num_points = data['c_pc_xyz'].shape[0]
        if feat.shape[0] != num_points:
            raise ValueError(
                f"{paths['feat']}: {feat.shape[0]} features for "
                f'{num_points} points; re-run step7b extraction for this sample.'
            )
        data['c_vggt_feat'] = torch.from_numpy(feat)
        if self.load_reliability:
            conf = np.load(paths['conf']).astype(np.float32)
            view_count = np.load(paths['view_count']).astype(np.float32)
            if conf.shape != (num_points,):
                raise ValueError(
                    f"{paths['conf']}: expected shape ({num_points},), got {conf.shape}"
                )
            if view_count.shape != (num_points,):
                raise ValueError(
                    f"{paths['view_count']}: expected shape ({num_points},), "
                    f'got {view_count.shape}'
                )
            data['c_vggt_conf'] = torch.from_numpy(conf)
            data['c_vggt_view_count'] = torch.from_numpy(view_count)

        if 'pred_mask_local' not in data and 'x' in data:
            data['pred_mask_local'] = data['x']
        if 'c_visit_id' not in data:
            data['c_visit_id'] = data_item.get('visit_id', '')
        if 'c_desc_id' not in data:
            data['c_desc_id'] = data_item.get('desc_id', '')

        return data
