import unittest

import torch

from utils.metrics import compute_average_precision


class AveragePrecisionTest(unittest.TestCase):
    def test_perfect_ranking(self):
        target = torch.tensor([0, 1, 0, 1])
        scores = torch.tensor([0.1, 0.9, 0.2, 0.8])
        self.assertAlmostEqual(
            compute_average_precision(target, scores).item(), 1.0
        )

    def test_imperfect_ranking(self):
        target = torch.tensor([1, 0, 1, 0])
        scores = torch.tensor([0.9, 0.8, 0.7, 0.1])
        # Precision at the positive ranks is 1/1 and 2/3.
        expected = (1.0 + 2.0 / 3.0) / 2.0
        self.assertAlmostEqual(
            compute_average_precision(target, scores).item(), expected, places=6
        )

    def test_tied_scores_are_grouped(self):
        target = torch.tensor([1, 0])
        scores = torch.tensor([0.5, 0.5])
        self.assertAlmostEqual(
            compute_average_precision(target, scores).item(), 0.5
        )

    def test_empty_target_is_zero(self):
        target = torch.zeros(3)
        scores = torch.tensor([0.9, 0.2, 0.1])
        self.assertEqual(compute_average_precision(target, scores).item(), 0.0)

    def test_batch_returns_per_frame_values(self):
        target = torch.tensor([[1, 0], [1, 0]])
        scores = torch.tensor([[0.9, 0.1], [0.1, 0.9]])
        expected = torch.tensor([1.0, 0.5])
        torch.testing.assert_close(
            compute_average_precision(target, scores), expected
        )

    def test_shape_mismatch_is_rejected(self):
        with self.assertRaises(ValueError):
            compute_average_precision(torch.ones(2), torch.ones(3))


if __name__ == '__main__':
    unittest.main()
