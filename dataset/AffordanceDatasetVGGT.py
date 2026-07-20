import os

import numpy as np
import torch
from loguru import logger

from dataset.AffordanceDataset import AffordanceDataset


class AffordanceDatasetVGGT(AffordanceDataset):
    """AffordanceDataset plus per-point VGGT features (``c_vggt_feat``).

    Expects a ``vggt_feat.npy`` ([N, feat_dim], float16/32) in each sample
    directory, produced by ``pipeline/step7b_vggt_feats/extract_vggt_features.py``.
    Samples without a feature file are dropped at init.

    Only the processed-branch modes are supported (``use_sam2``, ``use_sam2_1``,
    ``use_processed_final_train``), since those are the ones with per-sample
    directories the features live in.

    Also normalizes keys for the refinement loops: ``pred_mask_local`` is
    aliased to ``x`` (the initial lifted mask) when absent, and
    ``c_visit_id``/``c_desc_id`` are filled from the data item.
    """

    def __init__(self, *args, vggt_feat_name='vggt_feat.npy', vggt_feat_root=None,
                 require_vggt=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.vggt_feat_name = vggt_feat_name
        self.vggt_feat_root = vggt_feat_root

        if not self.data_items or 'base_path' not in self.data_items[0]:
            raise ValueError(
                'AffordanceDatasetVGGT requires a processed-branch mode '
                '(use_sam2 / use_sam2_1 / use_processed_final_train) with per-sample directories.'
            )

        if require_vggt:
            num_total = len(self.data_items)
            self.data_items = [item for item in self.data_items
                               if os.path.exists(self._feat_path(item))]
            num_kept = len(self.data_items)
            if num_kept < num_total:
                logger.warning(
                    f'AffordanceDatasetVGGT: kept {num_kept}/{num_total} samples with '
                    f'{self.vggt_feat_name}; run step7b extraction for the missing ones.'
                )
            if num_kept == 0:
                raise ValueError(
                    f'No sample under {self.processed_dir} has {self.vggt_feat_name}; '
                    'run pipeline/step7b_vggt_feats/extract_vggt_features.py first.'
                )

    def _feat_path(self, data_item):
        base_path = data_item['base_path']
        if self.vggt_feat_root is not None:
            rel = os.path.relpath(base_path, self.processed_dir)
            return os.path.join(self.vggt_feat_root, rel, self.vggt_feat_name)
        return os.path.join(base_path, self.vggt_feat_name)

    def __getitem__(self, idx):
        data = super().__getitem__(idx)
        data_item = self.data_items[idx]

        feat = np.load(self._feat_path(data_item)).astype(np.float32)
        num_points = data['c_pc_xyz'].shape[0]
        if feat.shape[0] != num_points:
            raise ValueError(
                f'{self._feat_path(data_item)}: {feat.shape[0]} features for '
                f'{num_points} points; re-run step7b extraction for this sample.'
            )
        data['c_vggt_feat'] = torch.from_numpy(feat)

        if 'pred_mask_local' not in data and 'x' in data:
            data['pred_mask_local'] = data['x']
        if 'c_visit_id' not in data:
            data['c_visit_id'] = data_item.get('visit_id', '')
        if 'c_desc_id' not in data:
            data['c_desc_id'] = data_item.get('desc_id', '')

        return data
