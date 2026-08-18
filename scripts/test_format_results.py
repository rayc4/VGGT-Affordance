import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts import format_results


def write_results(path, score):
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({'mAP': [score]}))


class CheckpointResolutionTest(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.root = Path(self.temp_dir.name)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_resolve_numbered_checkpoint(self):
        checkpoint = (self.root / 'run' / 'ckpt'
                      / 'mask_refinement_model000145.pt')
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        results = (self.root / 'run' / 'eval' / 'mask_refinement_model000145'
                   / 'test-1' / 'results.json')
        write_results(results, 0.25)

        self.assertEqual(format_results.resolve(str(checkpoint)), str(results))

    def test_resolve_best_checkpoint_to_numbered_results(self):
        checkpoint_dir = self.root / 'run' / 'ckpt'
        checkpoint_dir.mkdir(parents=True)
        best_checkpoint = checkpoint_dir / 'mask_refinement_model_best.pt'
        best_checkpoint.touch()
        (checkpoint_dir / 'mask_refinement_model000145.pt').touch()
        (checkpoint_dir / 'best_checkpoint.json').write_text(
            json.dumps({'step': 145})
        )
        results = (self.root / 'run' / 'eval' / 'mask_refinement_model000145'
                   / 'test-1' / 'results.json')
        write_results(results, 0.25)

        self.assertEqual(
            format_results.resolve(str(best_checkpoint)), str(results)
        )

    def test_resolve_checkpoint_using_metadata(self):
        checkpoint = self.root / 'run' / 'ckpt' / 'model.pt'
        checkpoint.parent.mkdir(parents=True)
        checkpoint.touch()
        result_dir = self.root / 'run' / 'eval' / 'test-1'
        results = result_dir / 'results.json'
        write_results(results, 0.25)
        (result_dir / 'metadata.json').write_text(json.dumps({
            'checkpoint': str(checkpoint.resolve()),
            'checkpoint_step': None,
        }))

        self.assertEqual(format_results.resolve(str(checkpoint)), str(results))


if __name__ == '__main__':
    unittest.main()
