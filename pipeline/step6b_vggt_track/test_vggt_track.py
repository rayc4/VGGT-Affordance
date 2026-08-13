import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from pipeline.step6b_vggt_track.vggt_track import (
    FrameInfo,
    build_parser,
    preprocess_images,
    run,
    sample_mask_points,
    select_context_frames,
    warp_mask_from_tracks,
)


class FakeTracker:
    """Return stationary, fully confident tracks for integration testing."""

    def track(self, images: torch.Tensor, query_points_xy: np.ndarray):
        frame_count = images.shape[0]
        tracks = np.repeat(query_points_xy[None, :, :], frame_count, axis=0)
        quality = np.ones((frame_count, len(query_points_xy)), dtype=np.float32)
        return tracks, quality, quality


class FrameSelectionTest(unittest.TestCase):
    def test_selects_nearest_frames_within_window(self):
        frames = [
            FrameInfo(f"video_{stamp:.1f}", stamp, Path(f"{stamp:.1f}.jpg"))
            for stamp in (0.0, 0.1, 0.2, 0.6, 2.0)
        ]
        selected = select_context_frames(frames, "video_0.2", 3, 0.5)
        self.assertEqual(
            [frame.stem for frame in selected],
            ["video_0.1", "video_0.0", "video_0.6"],
        )
        selected = select_context_frames(
            frames, "video_0.2", 2, 0.5, excluded_stems={"video_0.1"}
        )
        self.assertEqual(
            [frame.stem for frame in selected], ["video_0.0", "video_0.6"]
        )


class CoordinateAndWarpTest(unittest.TestCase):
    def test_preprocess_coordinate_transform_round_trips(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            image_path = Path(temporary_dir) / "portrait.jpg"
            Image.new("RGB", (30, 40), "white").save(image_path)
            images, transforms = preprocess_images([image_path], model_size=518)
            self.assertEqual(tuple(images.shape), (1, 3, 518, 518))
            points = np.asarray([[0.0, 0.0], [29.0, 39.0], [12.5, 7.5]])
            restored = transforms[0].model_to_original(
                transforms[0].original_to_model(points)
            )
            np.testing.assert_allclose(restored, points, atol=1e-5)

    def test_sampling_is_bounded_and_uses_only_mask_pixels(self):
        mask = np.zeros((20, 30), dtype=np.uint8)
        mask[3:15, 7:24] = 1
        points = sample_mask_points(mask, 19)
        self.assertEqual(points.shape, (19, 2))
        self.assertTrue(np.all(mask[points[:, 1].astype(int), points[:, 0].astype(int)]))
        np.testing.assert_array_equal(points, sample_mask_points(mask, 19))

    def test_dense_mask_follows_uniform_track_translation(self):
        mask = np.zeros((16, 20), dtype=np.uint8)
        mask[4:10, 5:12] = 1
        queries = sample_mask_points(mask, 12)
        translated = queries + np.asarray([3.0, 2.0], dtype=np.float32)
        warped = warp_mask_from_tracks(
            mask, queries, translated, mask.shape, close_radius=0
        )
        expected = np.zeros_like(mask)
        expected[6:12, 8:15] = 1
        np.testing.assert_array_equal(warped, expected)


class OutputCompatibilityTest(unittest.TestCase):
    def test_run_copies_sources_and_adds_step6_compatible_frames(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            data_root = root / "data"
            raw_dir = (
                data_root
                / "train_val_set"
                / "visit"
                / "video"
                / "hires_wide"
            )
            raw_dir.mkdir(parents=True)
            for timestamp in ("1.0", "1.1", "1.2"):
                Image.new("RGB", (30, 20), (10, 20, 30)).save(
                    raw_dir / f"video_{timestamp}.jpg"
                )

            input_dir = root / "step6" / "val" / "visit" / "video" / "description"
            input_dir.mkdir(parents=True)
            mask = np.zeros((20, 30), dtype=np.uint8)
            mask[5:13, 9:20] = 1
            np.savez_compressed(
                input_dir / "video_1.0_mask_data.npz", masks=mask[None]
            )
            Image.new("RGB", (30, 20), "red").save(
                input_dir / "video_1.0_mask_000.jpg"
            )

            output_root = root / "step6b"
            args = build_parser().parse_args(
                [
                    "--split",
                    "val",
                    "--data_root",
                    str(data_root),
                    "--step6_root",
                    str(root / "step6"),
                    "--output_root",
                    str(output_root),
                    "--context_frames",
                    "2",
                    "--max_query_points",
                    "32",
                    "--min_track_points",
                    "4",
                    "--min_track_fraction",
                    "0.1",
                    "--min_propagated_pixels",
                    "4",
                    "--mask_close_radius",
                    "0",
                ]
            )
            stats = run(args, tracker=FakeTracker())

            result_dir = output_root / "val" / "visit" / "video" / "description"
            npz_files = sorted(result_dir.glob("*_mask_data.npz"))
            self.assertEqual(len(npz_files), 3)
            self.assertEqual(stats.source_frames, 1)
            self.assertEqual(stats.propagated_frames, 2)
            for path in npz_files:
                with np.load(path, allow_pickle=False) as data:
                    self.assertEqual(data.files, ["masks"])
                    self.assertEqual(data["masks"].ndim, 3)
                    self.assertEqual(data["masks"].shape[1:], (20, 30))
            self.assertEqual(len(list(result_dir.glob("*_mask_000.jpg"))), 3)

            resumed = run(args, tracker=FakeTracker())
            self.assertEqual(resumed.descriptions, 0)
            self.assertEqual(resumed.skipped_descriptions, 1)


if __name__ == "__main__":
    unittest.main()
