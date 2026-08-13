import json
import math
import os
import tempfile
import unittest

import torch

from utils.metrics import MAP_METRIC_VERSION
from utils.training import SimpleMaskRefinementTrainLoop


class _Config(dict):
    __getattr__ = dict.__getitem__


def _make_loop(*, ap50_floor=None, patience=0, warmup=0, is_main=False,
               save_dir=None):
    cfg = _Config({
        'lr': 1e-4,
        'max_steps': 10,
        'log_every_step': 100,
        'resume_ckpt': '',
        'weight_decay': 0.0,
        'lr_anneal_steps': 0,
        'early_stopping_metric': 'mAP',
        'early_stopping_mode': 'max',
        'early_stopping_patience': patience,
        'early_stopping_min_delta': 0.0,
        'early_stopping_warmup_epochs': warmup,
        'best_checkpoint_ap50_floor': ap50_floor,
    })
    return SimpleMaskRefinementTrainLoop(
        cfg=cfg,
        model=torch.nn.Linear(1, 1),
        dataloader=[None],
        val_dataloader=[None],
        device='cpu',
        save_dir=save_dir or tempfile.mkdtemp(),
        is_main=is_main,
        is_distributed=False,
    )


class ConstrainedCheckpointSelectionTest(unittest.TestCase):

    def test_highest_map_is_selected_only_among_floor_eligible_epochs(self):
        loop = _make_loop(ap50_floor=0.23)

        loop._update_early_stopping(0, {'mAP': 0.30, 'AP50': 0.22})
        self.assertFalse(loop.has_best_checkpoint)
        self.assertEqual(loop.best_val_metric, -math.inf)

        loop.step = 2
        loop._update_early_stopping(1, {'mAP': 0.24, 'AP50': 0.23})
        self.assertTrue(loop.has_best_checkpoint)
        self.assertEqual(loop.best_val_metric, 0.24)
        self.assertEqual(loop.best_step, 1)

        loop.step = 3
        loop._update_early_stopping(2, {'mAP': 0.23, 'AP50': 0.40})
        self.assertEqual(loop.best_val_metric, 0.24)
        self.assertEqual(loop.best_step, 1)

        loop.step = 4
        loop._update_early_stopping(3, {'mAP': 0.26, 'AP50': 0.231})
        self.assertEqual(loop.best_val_metric, 0.26)
        self.assertEqual(loop.best_step, 3)
        self.assertEqual(loop.best_constraint_value, 0.231)

    def test_patience_starts_only_after_first_eligible_checkpoint(self):
        loop = _make_loop(ap50_floor=0.23, patience=2, warmup=0)

        self.assertFalse(
            loop._update_early_stopping(0, {'mAP': 0.30, 'AP50': 0.22})
        )
        self.assertEqual(loop.bad_validation_count, 0)

        self.assertFalse(
            loop._update_early_stopping(1, {'mAP': 0.24, 'AP50': 0.23})
        )
        self.assertEqual(loop.bad_validation_count, 0)

        self.assertFalse(
            loop._update_early_stopping(2, {'mAP': 0.25, 'AP50': 0.22})
        )
        self.assertEqual(loop.bad_validation_count, 1)
        self.assertTrue(
            loop._update_early_stopping(3, {'mAP': 0.26, 'AP50': 0.22})
        )

    def test_null_floor_preserves_unconstrained_selection(self):
        loop = _make_loop(ap50_floor=None)
        loop._update_early_stopping(1, {'mAP': 0.20, 'AP50': 0.0})
        self.assertTrue(loop.has_best_checkpoint)
        self.assertEqual(loop.best_val_metric, 0.20)

    def test_saved_metadata_and_state_record_constraint(self):
        with tempfile.TemporaryDirectory() as save_dir:
            loop = _make_loop(
                ap50_floor=0.23, is_main=True, save_dir=save_dir
            )
            loop.step = 8
            loop._update_early_stopping(
                4, {'mAP': 0.25, 'AP50': 0.24}
            )

            with open(os.path.join(save_dir, 'best_checkpoint.json')) as handle:
                metadata = json.load(handle)
            self.assertEqual(metadata['step'], 7)
            self.assertEqual(metadata['constraints']['AP50'], {
                'minimum': 0.23,
                'value': 0.24,
                'validation_threshold': 0.5,
            })

            state = torch.load(
                os.path.join(save_dir, 'early_stopping_state.pt'),
                map_location='cpu',
            )
            self.assertEqual(state['best_checkpoint_ap50_floor'], 0.23)
            self.assertEqual(
                state['best_checkpoint_validation_threshold'], 0.5
            )
            self.assertEqual(state['best_constraint_value'], 0.24)
            self.assertTrue(state['has_best_checkpoint'])

    def test_missing_or_nonfinite_ap50_is_rejected(self):
        loop = _make_loop(ap50_floor=0.23)
        with self.assertRaisesRegex(KeyError, 'do not contain AP50'):
            loop._update_early_stopping(1, {'mAP': 0.2})
        with self.assertRaisesRegex(FloatingPointError, 'AP50 is not finite'):
            loop._update_early_stopping(
                1, {'mAP': 0.2, 'AP50': float('nan')}
            )

    def test_resume_rejects_different_constraint_threshold(self):
        with tempfile.TemporaryDirectory() as save_dir:
            loop = _make_loop(ap50_floor=0.23, save_dir=save_dir)
            state_path = os.path.join(save_dir, 'early_stopping_state.pt')
            torch.save({
                'step': 7,
                'monitor': 'mAP',
                'mode': 'max',
                'metric_versions': {
                    'mAP': MAP_METRIC_VERSION,
                },
                'best_checkpoint_ap50_floor': 0.23,
                'best_checkpoint_validation_threshold': 0.6,
            }, state_path)
            loop.resume_checkpoint = os.path.join(
                save_dir, 'mask_refinement_model000007.pt'
            )
            loop.resume_step = 7
            with self.assertRaisesRegex(
                ValueError, 'different validation_threshold'
            ):
                loop._load_early_stopping_state()

    def test_constraint_state_round_trip(self):
        with tempfile.TemporaryDirectory() as save_dir:
            original = _make_loop(
                ap50_floor=0.23, is_main=True, save_dir=save_dir
            )
            original.step = 8
            original._update_early_stopping(
                4, {'mAP': 0.25, 'AP50': 0.24}
            )

            restored = _make_loop(ap50_floor=0.23, save_dir=save_dir)
            restored.resume_checkpoint = os.path.join(
                save_dir, 'mask_refinement_model000007.pt'
            )
            restored.resume_step = 7
            restored._load_early_stopping_state()

            self.assertTrue(restored.has_best_checkpoint)
            self.assertEqual(restored.best_val_metric, 0.25)
            self.assertEqual(restored.best_constraint_value, 0.24)
            self.assertEqual(restored.best_step, 7)
            self.assertEqual(restored.best_epoch, 4)

    def test_floor_must_be_a_probability(self):
        with self.assertRaisesRegex(
            ValueError, 'best_checkpoint_ap50_floor must be in'
        ):
            _make_loop(ap50_floor=1.01)


if __name__ == '__main__':
    unittest.main()
