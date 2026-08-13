import argparse
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock


def load_step4_module():
    torch = types.ModuleType('torch')
    torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    torchvision = types.ModuleType('torchvision')
    transforms = types.ModuleType('torchvision.transforms')
    functional = types.ModuleType('torchvision.transforms.functional')
    transforms.functional = functional
    torchvision.transforms = transforms

    fake_modules = {
        'torch': torch,
        'torchvision': torchvision,
        'torchvision.transforms': transforms,
        'torchvision.transforms.functional': functional,
    }
    module_path = Path(__file__).with_name('qwen_seg_image.py')
    spec = importlib.util.spec_from_file_location('step4_io_test_target', module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


STEP4 = load_step4_module()


class Step4DirectoryTest(unittest.TestCase):
    def test_explicit_roots_gain_split_directories(self):
        args = argparse.Namespace(
            input_root='/points',
            output_root='/crops/new',
            data_root='/legacy/points',
            split='val',
        )
        self.assertEqual(
            STEP4.resolve_io_directories(args),
            ('/points/val', '/crops/new/val'),
        )

    def test_legacy_data_root_still_appends_split(self):
        args = argparse.Namespace(
            input_root=None,
            output_root=None,
            data_root='/points/legacy',
            split='val',
        )
        input_dir, output_dir = STEP4.resolve_io_directories(args)
        self.assertEqual(input_dir, '/points/legacy/val')
        self.assertTrue(output_dir.endswith('/seg_image_output/legacy/val'))

    def test_resolved_output_does_not_gain_an_extra_split_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            input_dir = root / 'points'
            output_dir = root / 'new crops'
            visit_dir = input_dir / 'visit-1'
            visit_dir.mkdir(parents=True)
            point_data = [{
                'desc_id': 'desc-1',
                'frame_results': [{
                    'object_found': True,
                    'coordinates': {'x': 10, 'y': 20},
                    'image_name': 'frame.jpg',
                }],
            }]
            (visit_dir / 'video-1_point.json').write_text(json.dumps(point_data))

            calls = []

            def record_crop(*args, **kwargs):
                calls.append((args, kwargs))

            with mock.patch.object(STEP4, 'crop_image_gpu', record_crop):
                STEP4.process_all_images(
                    str(input_dir), 'val', 512, 512, str(output_dir),
                    '/raw', 'cpu',
                )

            self.assertEqual(len(calls), 1)
            self.assertEqual(
                calls[0][0][5],
                str(output_dir / 'visit-1' / 'video-1' / 'desc-1'),
            )


if __name__ == '__main__':
    unittest.main()
