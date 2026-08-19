import importlib.util
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


def load_step3_module():
    transformers = types.ModuleType('transformers')
    transformers.Qwen3VLForConditionalGeneration = object
    transformers.AutoProcessor = object

    qwen_vl_utils = types.ModuleType('qwen_vl_utils')
    qwen_vl_utils.process_vision_info = lambda messages: ([], [])

    torch = types.ModuleType('torch')
    tqdm_module = types.ModuleType('tqdm')
    tqdm_module.tqdm = lambda values, **kwargs: values

    fake_modules = {
        'transformers': transformers,
        'qwen_vl_utils': qwen_vl_utils,
        'torch': torch,
        'tqdm': tqdm_module,
    }
    module_path = Path(__file__).with_name('qwen_point.py')
    spec = importlib.util.spec_from_file_location('step3_path_test_target', module_path)
    module = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, fake_modules):
        spec.loader.exec_module(module)
    return module


STEP3 = load_step3_module()


class Step3OutputDirectoryTest(unittest.TestCase):
    def test_explicit_output_root_gains_split_directory(self):
        self.assertEqual(
            STEP3.resolve_output_dir('/input', '/new/points', 'val'),
            '/new/points/val',
        )

    def test_default_output_keeps_legacy_layout(self):
        self.assertEqual(
            STEP3.resolve_output_dir(
                '/input/clipwithaffordance_output', None, 'train'
            ),
            'pipeline/step3_point_prediction/'
            'point_clipwithaffordance_output/train',
        )


if __name__ == '__main__':
    unittest.main()
