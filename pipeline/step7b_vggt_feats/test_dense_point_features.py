import unittest
from unittest import mock
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from pipeline.step7b_vggt_feats.extract_vggt_features import (
    compute_source_frame_features,
    feature_view_weights,
    parse_args,
    point_head_feature_dim,
    sample_point_head_latents,
    select_source_sequence,
    source_frame_spec,
)


class _FakePointHead(nn.Module):
    """Minimal chunked head that exercises the final-predictor hook."""

    def __init__(self):
        super().__init__()
        self.scratch = nn.Module()
        self.scratch.output_conv2 = nn.Sequential(nn.Conv2d(3, 4, 1))

    def forward(self, _tokens, images, _patch_start_idx=None, patch_start_idx=None):
        del _patch_start_idx, patch_start_idx
        num_frames = images.shape[1]
        for start in range(0, num_frames, 2):
            latent = torch.stack([
                torch.full((3, 2, 2), float(frame_idx))
                for frame_idx in range(start, min(start + 2, num_frames))
            ])
            self.scratch.output_conv2(latent)
        prediction = torch.zeros(1, num_frames, 2, 2, 3)
        confidence = torch.ones(1, num_frames, 2, 2)
        return prediction, confidence


class DensePointFeatureTest(unittest.TestCase):

    def test_source_frame_is_the_default_and_has_distinct_outputs(self):
        argv = [
            'extract_vggt_features.py',
            '--split', 'train',
            '--processed_dir', 'unused',
        ]
        with mock.patch('sys.argv', argv):
            args = parse_args()
        self.assertEqual(args.frame_mode, 'source')
        self.assertEqual(args.feature_name, 'vggt_dpt_source_feat.npy')
        self.assertEqual(args.confidence_name, 'vggt_source_conf.npy')
        self.assertEqual(args.view_count_name, 'vggt_source_view_count.npy')

    def test_uniform_visit_average_filename_is_preserved(self):
        argv = [
            'extract_vggt_features.py',
            '--split', 'train',
            '--processed_dir', 'unused',
            '--frame_mode', 'visit_average',
            '--view_aggregation', 'uniform',
        ]
        with mock.patch('sys.argv', argv):
            args = parse_args()
        self.assertEqual(args.feature_source, 'dpt')
        self.assertEqual(args.feature_name, 'vggt_dpt_feat_uniform.npy')

    def test_uniform_aggregation_keeps_confidence_out_of_feature_weights(self):
        confidence = torch.tensor([1.0, 3.0])
        torch.testing.assert_close(
            feature_view_weights(confidence, 'uniform'),
            torch.ones_like(confidence),
        )
        torch.testing.assert_close(
            feature_view_weights(confidence, 'confidence'),
            torch.log1p(confidence),
        )

    def test_samples_frames_across_internal_dpt_chunks(self):
        point_head = _FakePointHead()
        center = torch.zeros(1, 1, 1, 2)

        features, confidence = sample_point_head_latents(
            point_head,
            tokens_list=[],
            images=torch.zeros(1, 3, 3, 2, 2),
            patch_start_idx=0,
            sample_grids=[center, None, center],
        )

        self.assertEqual(point_head_feature_dim(point_head), 3)
        self.assertEqual([tuple(item.shape) for item in features], [
            (1, 3), (0, 3), (1, 3),
        ])
        torch.testing.assert_close(features[0], torch.zeros(1, 3))
        torch.testing.assert_close(features[2], torch.full((1, 3), 2.0))
        self.assertEqual(tuple(confidence.shape), (1, 3, 2, 2))
        self.assertFalse(point_head.scratch.output_conv2._forward_pre_hooks)

    def test_source_sequence_keeps_source_first_and_uses_nearby_context(self):
        frames = [f'frame_{idx}' for idx in range(8)]
        self.assertEqual(
            select_source_sequence(frames, 'frame_3', 5, context_stride=1),
            ['frame_3', 'frame_2', 'frame_4', 'frame_1', 'frame_5'],
        )

    def test_source_frame_is_recovered_from_processed_sample_path(self):
        self.assertEqual(
            source_frame_spec(
                '/processed/train/420683/42445132/'
                'description__frame_5424.954__mask_0',
                '/processed/train',
            ),
            ('420683', '42445132', '5424.954'),
        )

    def test_source_feature_uses_only_frame_zero_descriptor(self):
        class FakeAggregator(nn.Module):
            def forward(self, images):
                batch, frames = images.shape[:2]
                tokens = torch.zeros(batch, frames, 1, 2)
                return [tokens], 0

        point_head = _FakePointHead()
        args = SimpleNamespace(
            feature_source='dpt',
            vis_thres=0.25,
        )
        source_data = (
            torch.zeros(3, 2, 2),
            np.eye(4),
            (2.0, 2.0, 1.0, 1.0, 0.0, 0.0),
            torch.ones(2, 2),
        )
        features, view_count, confidence = compute_source_frame_features(
            FakeAggregator(),
            point_head,
            point_head,
            proj_t=None,
            feature_dim=3,
            source_data=source_data,
            context_images=[torch.ones(3, 2, 2)],
            points=np.asarray([[0.5, 0.5, 1.0]], dtype=np.float32),
            args=args,
            device='cpu',
        )
        np.testing.assert_allclose(features, np.zeros((1, 3)))
        np.testing.assert_allclose(view_count, np.ones(1))
        np.testing.assert_allclose(confidence, np.ones(1))


if __name__ == '__main__':
    unittest.main()
