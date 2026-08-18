import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


def load_module(name, relative_path, fake_modules=None):
    module_path = Path(__file__).resolve().parents[1] / relative_path
    spec = importlib.util.spec_from_file_location(name, module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, fake_modules or {}):
        spec.loader.exec_module(module)
    return module


transformers = types.ModuleType('transformers')
for name in (
    'AutoModelForCausalLM', 'AutoProcessor', 'GenerationConfig', 'SamModel',
    'SamProcessor',
):
    setattr(transformers, name, object)

numpy = types.ModuleType('numpy')
numpy.array = object

STEP5 = load_module(
    'step5_path_test_target',
    'pipeline/step5_molmo_sam/molmo_sam.py',
    {
        'torch': types.ModuleType('torch'),
        'cv2': types.ModuleType('cv2'),
        'numpy': numpy,
        'transformers': transformers,
    },
)
STEP6 = load_module(
    'step6_path_test_target',
    'pipeline/step6_molmo_merge/molmo_merge.py',
    {'numpy': numpy},
)


class SplitOutputRootTest(unittest.TestCase):
    def test_step5_appends_split_to_both_roots(self):
        self.assertEqual(
            STEP5.resolve_io_directories('/step4/new', '/step5/new', 'val'),
            ('/step4/new/val', '/step5/new/val'),
        )

    def test_step5_defaults_keep_legacy_layout(self):
        self.assertEqual(
            STEP5.resolve_io_directories(None, None, 'train'),
            (
                'pipeline/step4_crop_images/seg_image_output/'
                'point_clipwithaffordance_output/train',
                'pipeline/step5_molmo_sam/molmo_output/train',
            ),
        )

    def test_step5_reads_affordance_from_explicit_root(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            affordance_dir = Path(temp_dir) / 'val'
            affordance_dir.mkdir()
            (affordance_dir / 'visit-1_affordance.json').write_text(
                json.dumps([{'desc_id': 'desc-1', 'affordance': 'handle'}])
            )
            self.assertEqual(
                STEP5.get_affordance_info(
                    'val', 'visit-1', 'video-1', 'desc-1', temp_dir
                ),
                'handle',
            )

    def test_step6_appends_split_to_all_roots(self):
        args = argparse.Namespace(
            cropinfo_root='/step4/new',
            molmo_root='/step5/new',
            merge_root='/step6/new',
            split='val',
        )
        self.assertEqual(
            STEP6.resolve_io_directories(args),
            ('/step4/new/val', '/step5/new/val', '/step6/new/val'),
        )

    def test_step6_defaults_keep_legacy_layout(self):
        args = argparse.Namespace(
            cropinfo_root=None,
            molmo_root=None,
            merge_root=None,
            split='train',
        )
        self.assertEqual(
            STEP6.resolve_io_directories(args),
            (
                'pipeline/step4_crop_images/seg_image_output/'
                'point_clipwithaffordance_output/train',
                'pipeline/step5_molmo_sam/molmo_output/train',
                'pipeline/step6_molmo_merge/molmo_merge_output/train',
            ),
        )


if __name__ == '__main__':
    unittest.main()
