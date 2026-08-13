"""Step 6b: expand Step 6 masks to nearby frames with VGGT pixel tracks.

Each Step 6 frame is placed at index zero of a VGGT sequence.  Pixels sampled
from each source mask are used as VGGT track queries.  Confident tracks warp
the dense source mask into nearby frames, producing a superset with the same
directory, NPZ, and visualization format as Step 6.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Protocol, Sequence

import cv2
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


REPO_ROOT = Path(__file__).resolve().parents[2]
VGGT_ROOT = REPO_ROOT / "vggt"
STEP6_ROOT_BASE = Path("pipeline/step6_molmo_merge/molmo_merge_output")
OUTPUT_ROOT_BASE = Path("pipeline/step6b_vggt_track/vggt_track_output")
MASK_SUFFIX = "_mask_data.npz"
FRAME_PATTERN = re.compile(r"^(?P<video>.+)_(?P<timestamp>-?\d+(?:\.\d+)?)$")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


@dataclass(frozen=True)
class FrameInfo:
    stem: str
    timestamp: float
    path: Path


@dataclass(frozen=True)
class ImageTransform:
    """Coordinate transform between an original frame and VGGT input pixels."""

    original_width: int
    original_height: int
    resized_width: int
    resized_height: int
    pad_left: int
    pad_top: int
    model_size: int

    def original_to_model(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float32).copy()
        points[..., 0] = (
            points[..., 0] * self.resized_width / self.original_width
            + self.pad_left
        )
        points[..., 1] = (
            points[..., 1] * self.resized_height / self.original_height
            + self.pad_top
        )
        return points

    def model_to_original(self, points_xy: np.ndarray) -> np.ndarray:
        points = np.asarray(points_xy, dtype=np.float32).copy()
        points[..., 0] = (
            (points[..., 0] - self.pad_left)
            * self.original_width
            / self.resized_width
        )
        points[..., 1] = (
            (points[..., 1] - self.pad_top)
            * self.original_height
            / self.resized_height
        )
        return points


@dataclass(frozen=True)
class MaskQueries:
    mask_index: int
    start: int
    stop: int
    source_points_xy: np.ndarray
    source_mask: np.ndarray


@dataclass(frozen=True)
class Proposal:
    mask: np.ndarray
    score: float
    source_stem: str
    source_mask_index: int


@dataclass
class RunStats:
    descriptions: int = 0
    skipped_descriptions: int = 0
    source_frames: int = 0
    propagated_frames: int = 0
    propagated_masks: int = 0

    def add(self, other: "RunStats") -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


class TrackBackend(Protocol):
    def track(
        self, images: torch.Tensor, query_points_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return tracks [S,N,2], visibility [S,N], confidence [S,N]."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 6b: propagate full-image masks with VGGT pixel tracks"
    )
    parser.add_argument("--split", required=True, choices=("train", "val"))
    parser.add_argument("--data_root", default="scenefun3d")
    parser.add_argument(
        "--input_root", "--input-root", "--step6_root", dest="step6_root",
        default=None,
        help="Step 6 input root; <split> is appended",
    )
    parser.add_argument(
        "--output_root",
        default=None,
        help="Output root; <split> is appended",
    )
    parser.add_argument("--hf_model", default="facebook/VGGT-1B")
    parser.add_argument("--ckpt", default=None, help="Local VGGT checkpoint")
    parser.add_argument(
        "--local_files_only",
        "--local-files-only",
        action="store_true",
        help="Require the Hugging Face checkpoint to be cached locally",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--context_frames",
        type=int,
        default=8,
        help="Maximum nearby frames added to each source-frame VGGT sequence",
    )
    parser.add_argument(
        "--context_window_seconds",
        type=float,
        default=3.0,
        help="Only consider frames this many seconds from the source",
    )
    parser.add_argument("--model_size", type=int, default=518)
    parser.add_argument(
        "--max_query_points",
        type=int,
        default=512,
        help="Maximum sampled pixels per source mask",
    )
    parser.add_argument(
        "--query_chunk_size",
        type=int,
        default=256,
        help="Track-head query chunk size; VGGT image tokens are reused",
    )
    parser.add_argument("--visibility_threshold", type=float, default=0.5)
    parser.add_argument("--confidence_threshold", type=float, default=0.5)
    parser.add_argument("--min_track_points", type=int, default=8)
    parser.add_argument("--min_track_fraction", type=float, default=0.25)
    parser.add_argument("--min_propagated_pixels", type=int, default=8)
    parser.add_argument(
        "--mask_close_radius",
        type=int,
        default=1,
        help="Closing radius used to repair subpixel rasterization holes (0 disables)",
    )
    parser.add_argument(
        "--dedupe_iou",
        type=float,
        default=0.9,
        help="Drop a propagated proposal when its IoU with a better one is this high",
    )
    parser.add_argument("--visit_id", default=None)
    parser.add_argument("--video_id", default=None)
    parser.add_argument("--desc_id", default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace completed Step 6b description directories",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, --num_shards), and num_shards >= 1")
    for name in (
        "context_frames",
        "max_query_points",
        "query_chunk_size",
        "min_track_points",
        "min_propagated_pixels",
        "mask_close_radius",
    ):
        if getattr(args, name) < 0:
            parser.error(f"--{name} must be non-negative")
    if args.max_query_points == 0 or args.query_chunk_size == 0:
        parser.error("--max_query_points and --query_chunk_size must be positive")
    if args.model_size <= 0 or args.model_size % 14:
        parser.error("--model_size must be a positive multiple of VGGT's patch size 14")
    if args.context_window_seconds <= 0:
        parser.error("--context_window_seconds must be positive")
    for name in (
        "visibility_threshold",
        "confidence_threshold",
        "min_track_fraction",
        "dedupe_iou",
    ):
        if not 0.0 <= getattr(args, name) <= 1.0:
            parser.error(f"--{name} must be in [0, 1]")


def parse_frame(path: Path, video_id: str) -> FrameInfo | None:
    match = FRAME_PATTERN.match(path.stem)
    if match is None or match.group("video") != video_id:
        return None
    return FrameInfo(path.stem, float(match.group("timestamp")), path)


def enumerate_video_frames(video_dir: Path, video_id: str) -> list[FrameInfo]:
    if not video_dir.is_dir():
        return []
    frames = []
    for path in video_dir.iterdir():
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
            frame = parse_frame(path, video_id)
            if frame is not None:
                frames.append(frame)
    return sorted(frames, key=lambda frame: (frame.timestamp, frame.stem))


def select_context_frames(
    frames: Sequence[FrameInfo],
    source_stem: str,
    count: int,
    window_seconds: float,
    excluded_stems: set[str] | None = None,
) -> list[FrameInfo]:
    source = next((frame for frame in frames if frame.stem == source_stem), None)
    if source is None:
        return []
    excluded = excluded_stems or set()
    candidates = [
        frame
        for frame in frames
        if frame.stem != source_stem
        and frame.stem not in excluded
        and abs(frame.timestamp - source.timestamp) <= window_seconds
    ]
    candidates.sort(
        key=lambda frame: (
            abs(frame.timestamp - source.timestamp),
            frame.timestamp,
            frame.stem,
        )
    )
    return candidates[:count]


def _resized_dimension(value: float, patch_size: int, maximum: int) -> int:
    rounded = int(round(value / patch_size)) * patch_size
    return min(maximum, max(patch_size, rounded))


def preprocess_images(
    image_paths: Sequence[Path], model_size: int = 518
) -> tuple[torch.Tensor, list[ImageTransform]]:
    """Apply VGGT's official pad preprocessing and retain inverse transforms."""

    tensors = []
    transforms = []
    for path in image_paths:
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            if width >= height:
                resized_width = model_size
                resized_height = _resized_dimension(
                    height * model_size / width, 14, model_size
                )
            else:
                resized_height = model_size
                resized_width = _resized_dimension(
                    width * model_size / height, 14, model_size
                )
            resized = image.resize(
                (resized_width, resized_height), Image.Resampling.BICUBIC
            )
        pad_left = (model_size - resized_width) // 2
        pad_top = (model_size - resized_height) // 2
        canvas = Image.new("RGB", (model_size, model_size), (255, 255, 255))
        canvas.paste(resized, (pad_left, pad_top))
        array = np.asarray(canvas, dtype=np.float32) / 255.0
        tensors.append(torch.from_numpy(array).permute(2, 0, 1))
        transforms.append(
            ImageTransform(
                original_width=width,
                original_height=height,
                resized_width=resized_width,
                resized_height=resized_height,
                pad_left=pad_left,
                pad_top=pad_top,
                model_size=model_size,
            )
        )
    if not tensors:
        raise ValueError("At least one image is required")
    return torch.stack(tensors), transforms


def resize_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if binary.shape != shape:
        binary = cv2.resize(
            binary, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return binary


def sample_mask_points(mask: np.ndarray, max_points: int) -> np.ndarray:
    """Deterministically sample representative (x, y) pixels from a mask."""

    pixels_yx = np.argwhere(np.asarray(mask) > 0)
    if not len(pixels_yx):
        return np.empty((0, 2), dtype=np.float32)
    if len(pixels_yx) > max_points:
        indices = np.floor(
            np.arange(max_points, dtype=np.float64) * len(pixels_yx) / max_points
        ).astype(np.int64)
        pixels_yx = pixels_yx[indices]
    return pixels_yx[:, ::-1].astype(np.float32)


def warp_mask_from_tracks(
    source_mask: np.ndarray,
    source_points_xy: np.ndarray,
    target_points_xy: np.ndarray,
    target_shape: tuple[int, int],
    close_radius: int = 1,
    pixel_chunk_size: int = 4096,
) -> np.ndarray:
    """Warp every foreground pixel by its nearest confident query's motion."""

    source_mask = np.asarray(source_mask) > 0
    source_yx = np.argwhere(source_mask)
    output = np.zeros(target_shape, dtype=np.uint8)
    if not len(source_yx) or not len(source_points_xy):
        return output

    source_height, source_width = source_mask.shape
    target_height, target_width = target_shape
    source_scale = np.asarray(
        [max(source_width - 1, 1), max(source_height - 1, 1)], dtype=np.float32
    )
    target_scale = np.asarray(
        [max(target_width - 1, 1), max(target_height - 1, 1)], dtype=np.float32
    )
    queries = np.asarray(source_points_xy, dtype=np.float32) / source_scale
    targets = np.asarray(target_points_xy, dtype=np.float32) / target_scale
    motion = targets - queries
    source_xy = source_yx[:, ::-1].astype(np.float32)
    source_normalized = source_xy / source_scale

    for start in range(0, len(source_normalized), pixel_chunk_size):
        chunk = source_normalized[start : start + pixel_chunk_size]
        distances = np.sum((chunk[:, None, :] - queries[None, :, :]) ** 2, axis=2)
        nearest = np.argmin(distances, axis=1)
        warped = (chunk + motion[nearest]) * target_scale
        warped = np.rint(warped).astype(np.int64)
        inside = (
            (warped[:, 0] >= 0)
            & (warped[:, 0] < target_width)
            & (warped[:, 1] >= 0)
            & (warped[:, 1] < target_height)
        )
        output[warped[inside, 1], warped[inside, 0]] = 1

    if close_radius:
        size = 2 * close_radius + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        output = cv2.morphologyEx(output, cv2.MORPH_CLOSE, kernel)
    return output


def mask_iou(first: np.ndarray, second: np.ndarray) -> float:
    first_binary = np.asarray(first) > 0
    second_binary = np.asarray(second) > 0
    union = np.count_nonzero(first_binary | second_binary)
    if not union:
        return 1.0
    return float(np.count_nonzero(first_binary & second_binary) / union)


def dedupe_proposals(
    proposals: Iterable[Proposal], iou_threshold: float
) -> list[Proposal]:
    kept = []
    for proposal in sorted(
        proposals,
        key=lambda item: (-item.score, item.source_stem, item.source_mask_index),
    ):
        if any(mask_iou(proposal.mask, other.mask) >= iou_threshold for other in kept):
            continue
        kept.append(proposal)
    return kept


def save_overlay(image_path: Path, mask: np.ndarray, output_path: Path) -> None:
    with Image.open(image_path) as opened:
        image = np.asarray(opened.convert("RGB")).copy()
    binary = resize_mask(mask, image.shape[:2]) > 0
    image[binary] = (255, 0, 0)
    Image.fromarray(image).save(output_path)


def load_masks(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "masks" not in data:
            raise ValueError(f"Step 6 file has no 'masks' array: {path}")
        masks = np.asarray(data["masks"])
    if masks.ndim != 3:
        raise ValueError(f"Step 6 masks must have shape [M,H,W], got {masks.shape}: {path}")
    return masks


def build_mask_queries(
    masks: np.ndarray,
    image_shape: tuple[int, int],
    max_query_points: int,
) -> tuple[np.ndarray, list[MaskQueries]]:
    all_points = []
    records = []
    offset = 0
    for mask_index, mask in enumerate(masks):
        resized = resize_mask(mask, image_shape)
        points = sample_mask_points(resized, max_query_points)
        if not len(points):
            continue
        all_points.append(points)
        records.append(
            MaskQueries(
                mask_index=mask_index,
                start=offset,
                stop=offset + len(points),
                source_points_xy=points,
                source_mask=resized,
            )
        )
        offset += len(points)
    if not all_points:
        return np.empty((0, 2), dtype=np.float32), records
    return np.concatenate(all_points, axis=0), records


def valid_track_mask(
    tracks_xy: np.ndarray,
    visibility: np.ndarray,
    confidence: np.ndarray,
    target_shape: tuple[int, int],
    visibility_threshold: float,
    confidence_threshold: float,
) -> np.ndarray:
    height, width = target_shape
    tracks = np.asarray(tracks_xy)
    return (
        np.isfinite(tracks).all(axis=1)
        & (tracks[:, 0] >= 0)
        & (tracks[:, 0] < width)
        & (tracks[:, 1] >= 0)
        & (tracks[:, 1] < height)
        & (np.asarray(visibility) >= visibility_threshold)
        & (np.asarray(confidence) >= confidence_threshold)
    )


class VGGTTrackBackend:
    """Persistent VGGT aggregator and tracking head used by one worker."""

    def __init__(
        self,
        hf_model: str,
        checkpoint: str | None,
        local_files_only: bool,
        device: str,
        query_chunk_size: int,
    ) -> None:
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("Step 6b requires CUDA, but torch.cuda.is_available() is false")
        self.device = torch.device(device)
        self.query_chunk_size = query_chunk_size
        if str(VGGT_ROOT) not in sys.path:
            sys.path.insert(0, str(VGGT_ROOT))
        from vggt.models.vggt import VGGT

        if checkpoint:
            model = VGGT(
                enable_camera=False,
                enable_point=False,
                enable_depth=False,
                enable_track=True,
            )
            state = torch.load(checkpoint, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model" in state:
                state = state["model"]
            if isinstance(state, dict) and "state_dict" in state:
                state = state["state_dict"]
            if not isinstance(state, dict):
                raise ValueError(f"Unsupported VGGT checkpoint payload: {checkpoint}")
            if state and all(key.startswith("module.") for key in state):
                state = {key.removeprefix("module."): value for key, value in state.items()}
            model.load_state_dict(state, strict=False)
        else:
            model = VGGT.from_pretrained(
                hf_model, local_files_only=local_files_only
            )
            model.camera_head = None
            model.point_head = None
            model.depth_head = None
        if model.track_head is None:
            raise ValueError("The selected VGGT checkpoint has no tracking head")
        self.aggregator = model.aggregator.to(self.device).eval()
        self.track_head = model.track_head.to(self.device).eval()
        del model

    def _autocast(self):
        if self.device.type != "cuda":
            return contextlib.nullcontext()
        major, _ = torch.cuda.get_device_capability(self.device)
        dtype = torch.bfloat16 if major >= 8 else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def track(
        self, images: torch.Tensor, query_points_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        images_device = images.unsqueeze(0).to(self.device, non_blocking=True)
        query_tensor = torch.from_numpy(
            np.asarray(query_points_xy, dtype=np.float32)
        ).unsqueeze(0).to(self.device)
        track_chunks = []
        visibility_chunks = []
        confidence_chunks = []
        with torch.inference_mode(), self._autocast():
            tokens, patch_start_idx = self.aggregator(images_device)
            for start in range(0, query_tensor.shape[1], self.query_chunk_size):
                query_chunk = query_tensor[:, start : start + self.query_chunk_size]
                track_list, visibility, confidence = self.track_head(
                    tokens,
                    images=images_device,
                    patch_start_idx=patch_start_idx,
                    query_points=query_chunk,
                )
                track_chunks.append(track_list[-1])
                visibility_chunks.append(visibility)
                if confidence is None:
                    confidence = torch.ones_like(visibility)
                confidence_chunks.append(confidence)
        tracks = torch.cat(track_chunks, dim=2)[0].float().cpu().numpy()
        visibility = torch.cat(visibility_chunks, dim=2)[0].float().cpu().numpy()
        confidence = torch.cat(confidence_chunks, dim=2)[0].float().cpu().numpy()
        return tracks, visibility, confidence


def _raw_video_dir(data_root: Path, split: str, visit_id: str, video_id: str) -> Path:
    split_dir = "train_val_set" if split in {"train", "val"} else "test_set"
    return data_root / split_dir / visit_id / video_id / "hires_wide"


def _copy_step6_artifacts(source_dir: Path, output_dir: Path) -> None:
    for source in source_dir.iterdir():
        if source.is_file() and (
            source.name.endswith(MASK_SUFFIX) or source.suffix.lower() in IMAGE_SUFFIXES
        ):
            shutil.copy2(source, output_dir / source.name)


def _publish_description(temp_dir: Path, final_dir: Path, overwrite: bool) -> None:
    if final_dir.exists():
        if not overwrite:
            raise FileExistsError(final_dir)
        shutil.rmtree(final_dir)
    os.replace(temp_dir, final_dir)


def process_description(
    args: argparse.Namespace,
    tracker: TrackBackend,
    source_dir: Path,
    output_dir: Path,
    visit_id: str,
    video_id: str,
) -> RunStats:
    stats = RunStats(descriptions=1)
    seed_files = sorted(source_dir.glob(f"*{MASK_SUFFIX}"))
    stats.source_frames = len(seed_files)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.step6b-", dir=output_dir.parent)
    )
    try:
        _copy_step6_artifacts(source_dir, temp_dir)
        raw_dir = _raw_video_dir(Path(args.data_root), args.split, visit_id, video_id)
        frames = enumerate_video_frames(raw_dir, video_id)
        frame_by_stem = {frame.stem: frame for frame in frames}
        source_stems = {
            path.name[: -len(MASK_SUFFIX)] for path in seed_files
        }
        proposals_by_frame: dict[str, list[Proposal]] = {}

        for seed_path in seed_files:
            source_stem = seed_path.name[: -len(MASK_SUFFIX)]
            source_frame = frame_by_stem.get(source_stem)
            if source_frame is None:
                print(f"Warning: raw source frame not found for {seed_path}")
                continue
            context_frames = select_context_frames(
                frames,
                source_stem,
                args.context_frames,
                args.context_window_seconds,
                excluded_stems=source_stems,
            )
            if not context_frames:
                continue
            image_paths = [source_frame.path] + [frame.path for frame in context_frames]
            images, transforms = preprocess_images(image_paths, args.model_size)
            source_shape = (
                transforms[0].original_height,
                transforms[0].original_width,
            )
            try:
                masks = load_masks(seed_path)
            except ValueError as error:
                print(f"Warning: {error}")
                continue
            query_points, mask_queries = build_mask_queries(
                masks, source_shape, args.max_query_points
            )
            if not len(query_points):
                continue
            model_queries = transforms[0].original_to_model(query_points)
            tracks_model, visibility, confidence = tracker.track(images, model_queries)
            expected_shape = (len(image_paths), len(query_points), 2)
            if tracks_model.shape != expected_shape:
                raise ValueError(
                    f"VGGT returned tracks {tracks_model.shape}; expected {expected_shape}"
                )

            for sequence_index, target_frame in enumerate(context_frames, start=1):
                if target_frame.stem in source_stems:
                    continue
                target_transform = transforms[sequence_index]
                target_shape = (
                    target_transform.original_height,
                    target_transform.original_width,
                )
                target_tracks = target_transform.model_to_original(
                    tracks_model[sequence_index]
                )
                for record in mask_queries:
                    selection = slice(record.start, record.stop)
                    mask_tracks = target_tracks[selection]
                    mask_visibility = visibility[sequence_index, selection]
                    mask_confidence = confidence[sequence_index, selection]
                    valid = valid_track_mask(
                        mask_tracks,
                        mask_visibility,
                        mask_confidence,
                        target_shape,
                        args.visibility_threshold,
                        args.confidence_threshold,
                    )
                    valid_count = int(np.count_nonzero(valid))
                    required_count = max(
                        args.min_track_points,
                        int(np.ceil(len(valid) * args.min_track_fraction)),
                    )
                    if valid_count < required_count:
                        continue
                    propagated = warp_mask_from_tracks(
                        record.source_mask,
                        record.source_points_xy[valid],
                        mask_tracks[valid],
                        target_shape,
                        close_radius=args.mask_close_radius,
                    )
                    if np.count_nonzero(propagated) < args.min_propagated_pixels:
                        continue
                    score = float(
                        np.mean(
                            np.sqrt(
                                mask_visibility[valid] * mask_confidence[valid]
                            )
                        )
                    )
                    proposals_by_frame.setdefault(target_frame.stem, []).append(
                        Proposal(
                            mask=propagated,
                            score=score,
                            source_stem=source_stem,
                            source_mask_index=record.mask_index,
                        )
                    )

        for target_stem, proposals in sorted(proposals_by_frame.items()):
            kept = dedupe_proposals(proposals, args.dedupe_iou)
            if not kept:
                continue
            masks = np.stack([proposal.mask for proposal in kept]).astype(np.uint8)
            np.savez_compressed(temp_dir / f"{target_stem}{MASK_SUFFIX}", masks=masks)
            image_path = frame_by_stem[target_stem].path
            for mask_index, mask in enumerate(masks):
                save_overlay(
                    image_path,
                    mask,
                    temp_dir / f"{target_stem}_mask_{mask_index:03d}.jpg",
                )
            stats.propagated_frames += 1
            stats.propagated_masks += len(masks)

        _publish_description(temp_dir, output_dir, args.overwrite)
    finally:
        if temp_dir.exists():
            shutil.rmtree(temp_dir)
    return stats


def discover_jobs(
    args: argparse.Namespace, step6_root: Path, output_root: Path
) -> tuple[list[tuple[Path, Path, str, str]], int]:
    visits = sorted(path for path in step6_root.iterdir() if path.is_dir())
    if args.visit_id:
        visits = [path for path in visits if path.name == args.visit_id]
    visits = visits[args.shard :: args.num_shards]
    jobs = []
    skipped = 0
    for visit_dir in visits:
        for video_dir in sorted(path for path in visit_dir.iterdir() if path.is_dir()):
            if args.video_id and video_dir.name != args.video_id:
                continue
            for desc_dir in sorted(path for path in video_dir.iterdir() if path.is_dir()):
                if args.desc_id and desc_dir.name != args.desc_id:
                    continue
                output_dir = (
                    output_root / visit_dir.name / video_dir.name / desc_dir.name
                )
                if output_dir.exists() and not args.overwrite:
                    skipped += 1
                    continue
                jobs.append((desc_dir, output_dir, visit_dir.name, video_dir.name))
    return jobs, skipped


def run(args: argparse.Namespace, tracker: TrackBackend | None = None) -> RunStats:
    step6_root = Path(args.step6_root or STEP6_ROOT_BASE) / args.split
    output_root = Path(args.output_root or OUTPUT_ROOT_BASE) / args.split
    if not step6_root.is_dir():
        raise FileNotFoundError(f"Step 6 split input not found: {step6_root}")
    input_resolved = step6_root.resolve()
    output_resolved = output_root.resolve()
    if (
        input_resolved == output_resolved
        or input_resolved in output_resolved.parents
        or output_resolved in input_resolved.parents
    ):
        raise ValueError("--step6_root and --output_root must be disjoint trees")
    output_root.mkdir(parents=True, exist_ok=True)

    jobs, skipped = discover_jobs(args, step6_root, output_root)
    stats = RunStats(skipped_descriptions=skipped)
    print(
        f"Step 6b shard {args.shard}/{args.num_shards}: "
        f"{len(jobs)} pending, {skipped} already complete"
    )
    if not jobs:
        return stats
    if tracker is None:
        tracker = VGGTTrackBackend(
            hf_model=args.hf_model,
            checkpoint=args.ckpt,
            local_files_only=args.local_files_only,
            device=args.device,
            query_chunk_size=args.query_chunk_size,
        )
    for source_dir, output_dir, visit_id, video_id in tqdm(
        jobs, desc="description"
    ):
        result = process_description(
            args, tracker, source_dir, output_dir, visit_id, video_id
        )
        stats.add(result)
    return stats


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(parser, args)
    stats = run(args)
    print(
        "Step 6b complete: "
        f"descriptions={stats.descriptions}, "
        f"skipped={stats.skipped_descriptions}, "
        f"source_frames={stats.source_frames}, "
        f"new_frames={stats.propagated_frames}, "
        f"new_masks={stats.propagated_masks}"
    )


if __name__ == "__main__":
    main()
