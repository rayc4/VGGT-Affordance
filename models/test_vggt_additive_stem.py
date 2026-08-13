import unittest
from unittest import mock

import torch
from omegaconf import OmegaConf

from models.cdm_vggt import CDMVGGTAdditiveStem, VGGTAdditiveStemAdapter


class VGGTAdditiveStemAdapterTest(unittest.TestCase):

    def test_zero_initialization_preserves_base_and_projection_gets_gradient(self):
        torch.manual_seed(3)
        adapter = VGGTAdditiveStemAdapter(feat_dim=4, stem_dim=3)
        feature = torch.randn(1, 4, 4)
        confidence = torch.tensor([[1.0, 2.0, 8.0, 4.0]])
        view_count = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        residual = adapter(feature, confidence, view_count)
        torch.testing.assert_close(residual, torch.zeros_like(residual))

        target = torch.randn_like(residual)
        (residual * target).sum().backward()
        self.assertGreater(adapter.projection.weight.grad.abs().sum().item(), 0.0)

    def test_scalar_gate_is_monotonic_and_masks_unseen_points(self):
        adapter = VGGTAdditiveStemAdapter(feat_dim=4, stem_dim=3)
        feature = torch.randn(1, 4, 4)
        confidence = torch.tensor([[1.0, 2.0, 8.0, 100.0]])
        view_count = torch.tensor([[1.0, 1.0, 1.0, 0.0]])

        gate = adapter.confidence_gate(feature, confidence, view_count)
        self.assertLess(gate[0, 0].item(), gate[0, 1].item())
        self.assertLess(gate[0, 1].item(), gate[0, 2].item())
        self.assertEqual(gate[0, 3].item(), 0.0)

        with torch.no_grad():
            adapter.projection.weight.fill_(1.0)
        residual = adapter(feature, confidence, view_count)
        torch.testing.assert_close(residual[0, 3], torch.zeros(3))

    def test_parameterization_has_no_hidden_mlp(self):
        adapter = VGGTAdditiveStemAdapter(feat_dim=128, stem_dim=32)
        trainable = sum(
            parameter.numel() for parameter in adapter.parameters()
            if parameter.requires_grad
        )
        self.assertEqual(trainable, 4354)

    def test_joint_finetuning_keeps_scene_batchnorm_statistics_fixed(self):
        class FakeStem(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.stride = 1
                self.linear = torch.nn.Linear(1, 32, bias=False)
                self.bn = torch.nn.BatchNorm1d(32)

        class FakeSceneModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = torch.nn.Sequential(FakeStem())
                self.extra_bn = torch.nn.BatchNorm1d(32)

        cfg = OmegaConf.create({
            'input_feats': 1,
            'vggt_feat_dim': 128,
            'stem_dim': 32,
            'freeze_base': False,
            'freeze_batchnorm': True,
            'scene_model': {
                'name': 'PointTransformerSeg',
                'num_points': 8,
                'pretrained_weight': None,
                'freeze': False,
            },
        })
        scene_model = FakeSceneModel()
        with mock.patch(
            'models.cdm_vggt.load_scene_model', return_value=scene_model
        ):
            model = CDMVGGTAdditiveStem(cfg)

        model.train()
        self.assertTrue(model.scene_model.training)
        self.assertTrue(model.contact_layer.training)
        self.assertTrue(model.vggt_adapter.training)
        self.assertFalse(model.scene_model.enc1[0].bn.training)
        self.assertFalse(model.scene_model.extra_bn.training)
        self.assertTrue(all(
            parameter.requires_grad for parameter in model.parameters()
        ))


if __name__ == '__main__':
    unittest.main()
