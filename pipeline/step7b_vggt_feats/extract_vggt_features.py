"""Step 7b: lift dense VGGT point-head features to per-point 3D features.

For every processed sample directory (any directory containing
``filtered_point_cloud.ply``), this script computes per-point VGGT features and
reliability signals. Source mode writes ``vggt_dpt_source_feat.npy`` ([N, 128],
float16), ``vggt_source_conf.npy`` ([N], float32),
``vggt_source_view_count.npy`` ([N], uint16), and a reproducibility manifest in
a separate mirrored tree. By default that is
``<processed_sam2>/vggt_features/<split>``; override it with
``--feat_out_root``.

The default ``--frame_mode source`` preserves the frame identity of the lifted
2D mask. For every sample, its encoded source frame is placed at VGGT sequence
position zero, nearby frames provide context, and only the source frame's dense
latent is sampled at the exact projected pixel for each 3D point. Samples that
share a source frame reuse one VGGT forward.

``--frame_mode visit_average`` retains the legacy behavior: uniformly sample
frames from every visit video, lift the dense latent from every visible view,
and average those descriptors into one visit-level feature field.

Example:
    python pipeline/step7b_vggt_feats/extract_vggt_features.py \
        --data_root scenefun3d --split train \
        --processed_dir data/processed_sam2/train

    python pipeline/step7b_vggt_feats/extract_vggt_features.py \
        --data_root scenefun3d --split val \
        --processed_dir data/processed_sam2/val
"""
import argparse
import glob
import json
import os
import re
import sys

import cv2
import numpy as np
import open3d as o3d
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
VGGT_ROOT = os.path.join(REPO_ROOT, 'vggt')
for _p in (VGGT_ROOT, REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vggt.models.vggt import VGGT  # noqa: E402

PATCH_SIZE = 14
TOKEN_DIM = 2048  # 2 * embed_dim of VGGT-1B aggregator
TARGET_SIZE = 518
PATCH_CACHE_VERSION = 2
DPT_CACHE_VERSION = 1
DPT_FEATURE_NAME = 'vggt_dpt_feat.npy'
DPT_UNIFORM_FEATURE_NAME = 'vggt_dpt_feat_uniform.npy'
PATCH_FEATURE_NAME = 'vggt_feat.npy'
PATCH_UNIFORM_FEATURE_NAME = 'vggt_feat_uniform.npy'
SOURCE_DPT_FEATURE_NAME = 'vggt_dpt_source_feat.npy'
SOURCE_PATCH_FEATURE_NAME = 'vggt_source_feat.npy'
SOURCE_CONFIDENCE_NAME = 'vggt_source_conf.npy'
SOURCE_VIEW_COUNT_NAME = 'vggt_source_view_count.npy'
SOURCE_METADATA_NAME = 'vggt_source_meta.json'
SOURCE_CACHE_VERSION = 1
SOURCE_DIR_RE = re.compile(r'__frame_(?P<frame>.+?)__mask_[^/]+$')


def parse_args():
    parser = argparse.ArgumentParser(
        description='Lift dense VGGT features to per-point 3D features'
    )
    parser.add_argument('--data_root', type=str, default='scenefun3d',
                        help='SceneFun3D root containing train_val_set/ (e.g. scenefun3d)')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Split of the visits (train/val -> train_val_set, test -> test_set)')
    parser.add_argument('--processed_dir', type=str, required=True,
                        help='Split-specific sample tree, e.g. data/processed_sam2/train')
    parser.add_argument('--feat_out_root', type=str, default=None,
                        help='Split-specific mirror root for VGGT feature/reliability arrays '
                             '(default: <processed parent>/vggt_features/<split>)')
    parser.add_argument('--cache_dir', type=str, default='outputs/vggt_feat_cache',
                        help='Per-visit feature cache directory')
    parser.add_argument('--hf_model', type=str, default='facebook/VGGT-1B')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Local VGGT checkpoint (overrides --hf_model)')
    parser.add_argument('--frame_mode', choices=['source', 'visit_average'],
                        default='source',
                        help='source (default): make each lifted mask source frame '
                             'VGGT frame 0 and save only its exact-pixel descriptor; '
                             'visit_average: legacy visit-wide multi-view averaging')
    parser.add_argument('--feature_source', choices=['dpt', 'patch'], default='dpt',
                        help='Per-pixel descriptor source: the point-head dense latent '
                             '(recommended/default) or legacy final patch tokens')
    parser.add_argument('--view_aggregation', choices=['confidence', 'uniform'],
                        default='confidence',
                        help='Average visible-view features with log1p(point confidence) '
                             'weights (default) or uniform weights; confidence is saved '
                             'as a reliability input in both cases')
    parser.add_argument('--feature_name', type=str, default=None,
                        help='Output feature filename (default is derived from '
                             '--frame_mode, --feature_source, and --view_aggregation)')
    parser.add_argument('--confidence_name', type=str, default=None,
                        help='Output confidence filename (source mode defaults to '
                             f'{SOURCE_CONFIDENCE_NAME}; visit-average mode keeps '
                             'vggt_conf.npy)')
    parser.add_argument('--view_count_name', type=str, default=None,
                        help='Output view-count filename (source mode defaults to '
                             f'{SOURCE_VIEW_COUNT_NAME}; visit-average mode keeps '
                             'vggt_view_count.npy)')
    parser.add_argument('--metadata_name', type=str, default=SOURCE_METADATA_NAME,
                        help='Per-sample source-extraction manifest filename')
    parser.add_argument('--out_dim', type=int, default=256,
                        help='Legacy patch-token feature dim after fixed random '
                             'orthonormal projection; ignored for DPT features '
                             f'(0 or >={TOKEN_DIM} keeps raw patch tokens)')
    parser.add_argument('--frames_per_video', type=int, default=24,
                        help='Frames sampled per video in legacy visit-average mode')
    parser.add_argument('--chunk_size', type=int, default=12,
                        help='Frames per VGGT forward. In source mode this is one '
                             'source frame plus up to chunk_size-1 context frames')
    parser.add_argument('--context_stride', type=int, default=1,
                        help='Temporal frame stride for source-mode context selection')
    parser.add_argument('--confidence_head', choices=['point', 'depth'], default='point',
                        help='VGGT confidence head used for view weighting (default: point)')
    parser.add_argument('--vis_thres', type=float, default=0.25,
                        help='Depth visibility threshold (|d - z| <= thres * d), as in fusion_util')
    parser.add_argument('--pose_time_thresh', type=float, default=0.1,
                        help='Max |frame_ts - pose_ts| in seconds for nearest-pose lookup')
    parser.add_argument('--visit_id', type=str, default=None, help='Process a single visit')
    parser.add_argument('--require_nonempty_gt', action='store_true',
                        help='Skip samples with an empty gt_mask_local.npy (useful '
                             'for the filtered training split)')
    parser.add_argument('--num_shards', type=int, default=1,
                        help='Split visits deterministically across this many workers')
    parser.add_argument('--shard', type=int, default=0,
                        help='Worker shard index in [0, num_shards)')
    parser.add_argument('--seed', type=int, default=2023, help='Seed for the projection matrix')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overwrite', action='store_true',
                        help='Recompute existing per-sample feature/reliability arrays')
    parser.add_argument('--overwrite_cache', action='store_true', help='Recompute per-visit caches')
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error('--num_shards must be at least 1')
    if not 0 <= args.shard < args.num_shards:
        parser.error('--shard must be in [0, num_shards)')
    if args.visit_id is not None and (args.num_shards != 1 or args.shard != 0):
        parser.error('--visit_id cannot be combined with multi-worker sharding')
    if args.frames_per_video < 1:
        parser.error('--frames_per_video must be at least 1')
    if args.chunk_size < 1:
        parser.error('--chunk_size must be at least 1')
    if args.context_stride < 1:
        parser.error('--context_stride must be at least 1')
    if args.feature_name is None:
        if args.frame_mode == 'source':
            args.feature_name = (
                SOURCE_DPT_FEATURE_NAME if args.feature_source == 'dpt'
                else SOURCE_PATCH_FEATURE_NAME
            )
        else:
            default_feature_names = {
                ('dpt', 'confidence'): DPT_FEATURE_NAME,
                ('dpt', 'uniform'): DPT_UNIFORM_FEATURE_NAME,
                ('patch', 'confidence'): PATCH_FEATURE_NAME,
                ('patch', 'uniform'): PATCH_UNIFORM_FEATURE_NAME,
            }
            args.feature_name = default_feature_names[
                (args.feature_source, args.view_aggregation)
            ]
    if args.confidence_name is None:
        args.confidence_name = (
            SOURCE_CONFIDENCE_NAME if args.frame_mode == 'source'
            else 'vggt_conf.npy'
        )
    if args.view_count_name is None:
        args.view_count_name = (
            SOURCE_VIEW_COUNT_NAME if args.frame_mode == 'source'
            else 'vggt_view_count.npy'
        )
    for option, filename in (
            ('--feature_name', args.feature_name),
            ('--confidence_name', args.confidence_name),
            ('--view_count_name', args.view_count_name),
            ('--metadata_name', args.metadata_name)):
        if os.path.basename(filename) != filename:
            parser.error(f'{option} must be a filename, not a path')
    return args


def find_sample_dirs(processed_dir):
    sample_dirs = []
    for dirpath, _dirnames, filenames in os.walk(processed_dir):
        if 'filtered_point_cloud.ply' in filenames:
            sample_dirs.append(dirpath)
    return sorted(sample_dirs)


def source_frame_spec(sample_dir, processed_dir):
    """Return ``(visit_id, video_id, frame_timestamp)`` for a lifted sample.

    Newer lift results record ``video_id`` and ``frame_id`` in
    ``mask_result.json``. Existing processed_sam2 data encodes the same values
    in ``<visit>/<video>/<description>__frame_<timestamp>__mask_<index>``; keep
    that layout as a backward-compatible fallback.
    """
    rel_parts = os.path.relpath(sample_dir, processed_dir).split(os.sep)
    if len(rel_parts) < 3:
        raise ValueError(
            f'Expected <visit>/<video>/<sample> below {processed_dir}, got '
            f'{os.path.relpath(sample_dir, processed_dir)!r}'
        )

    match = SOURCE_DIR_RE.search(rel_parts[-1])
    if match is not None:
        return rel_parts[0], rel_parts[1], match.group('frame')

    metadata = {}
    metadata_path = os.path.join(sample_dir, 'mask_result.json')
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
    visit_id = str(metadata.get('visit_id') or rel_parts[0])
    video_id = str(metadata.get('video_id') or rel_parts[1])
    frame_ts = metadata.get('frame_id')
    if frame_ts in (None, ''):
        raise ValueError(
            f'Cannot recover source frame from {sample_dir}; expected frame_id '
            'in mask_result.json or __frame_<timestamp>__mask_ in the directory name'
        )
    return visit_id, video_id, str(frame_ts)


def frame_timestamp(frame_path, video_id):
    stem = os.path.splitext(os.path.basename(frame_path))[0]
    prefix = f'{video_id}_'
    if stem.startswith(prefix):
        return stem[len(prefix):]
    if '_' not in stem:
        raise ValueError(f'Cannot parse frame timestamp from {frame_path}')
    return stem.split('_', 1)[1]


def find_source_frame(frame_paths, video_id, frame_ts):
    expected_name = f'{video_id}_{frame_ts}.jpg'
    for path in frame_paths:
        if os.path.basename(path) == expected_name:
            return path
    matches = [
        path for path in frame_paths
        if frame_timestamp(path, video_id) == str(frame_ts)
    ]
    if len(matches) == 1:
        return matches[0]
    raise FileNotFoundError(
        f'Expected exactly one source RGB frame {expected_name}, found {len(matches)}'
    )


def select_source_sequence(frame_paths, source_path, sequence_size,
                           context_stride=1):
    """Put the source first, followed by temporally nearest context frames."""
    if source_path not in frame_paths:
        raise ValueError(f'Source frame is not in the video frame list: {source_path}')
    if sequence_size <= 1:
        return [source_path]

    source_idx = frame_paths.index(source_path)
    chosen = []
    used = {source_idx}
    radius = 1
    while len(chosen) < sequence_size - 1:
        added = False
        for idx in (source_idx - radius * context_stride,
                    source_idx + radius * context_stride):
            if 0 <= idx < len(frame_paths) and idx not in used:
                chosen.append(frame_paths[idx])
                used.add(idx)
                added = True
                if len(chosen) == sequence_size - 1:
                    break
        if not added and (source_idx - radius * context_stride < 0
                          and source_idx + radius * context_stride >= len(frame_paths)):
            break
        radius += 1

    # Near a boundary, a stride can exhaust one side before filling the
    # sequence. Backfill with the nearest unused frames so context count stays
    # stable whenever the video is long enough.
    if len(chosen) < sequence_size - 1:
        remaining = sorted(
            (idx for idx in range(len(frame_paths)) if idx not in used),
            key=lambda idx: (abs(idx - source_idx), idx),
        )
        chosen.extend(frame_paths[idx] for idx in remaining[
            :sequence_size - 1 - len(chosen)
        ])
    return [source_path, *chosen]


def load_projection(cache_dir, out_dim, seed):
    """Fixed seeded orthonormal projection so features are comparable across scenes/runs."""
    if out_dim <= 0 or out_dim >= TOKEN_DIM:
        return None
    os.makedirs(cache_dir, exist_ok=True)
    proj_path = os.path.join(cache_dir, f'vggt_rand_proj_{TOKEN_DIM}x{out_dim}_seed{seed}.npy')
    if os.path.exists(proj_path):
        return np.load(proj_path).astype(np.float32)
    rng = np.random.default_rng(seed)
    mat = rng.standard_normal((TOKEN_DIM, out_dim)).astype(np.float64)
    q, _ = np.linalg.qr(mat)
    q = q.astype(np.float32)
    # Multiple GPU workers may initialize the same deterministic projection at
    # startup. Write a process-local temporary file and publish it atomically so
    # no worker can observe a partially written .npy.
    tmp_path = f'{proj_path}.tmp.{os.getpid()}.npy'
    np.save(tmp_path, q)
    os.replace(tmp_path, proj_path)
    print(f'Saved projection matrix to {proj_path}')
    return q


def load_vggt_feature_model(args, device):
    if args.ckpt:
        model = VGGT()
        state = torch.load(args.ckpt, map_location='cpu')
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        model.load_state_dict(state, strict=False)
    else:
        model = VGGT.from_pretrained(args.hf_model)
    aggregator = model.aggregator
    point_head = model.point_head
    if point_head is None:
        raise ValueError('VGGT checkpoint has no point head for dense DPT features')
    confidence_head = model.point_head if args.confidence_head == 'point' else model.depth_head
    if confidence_head is None:
        raise ValueError(f'VGGT checkpoint has no {args.confidence_head} confidence head')
    # Camera and tracking heads are not needed for feature extraction.
    del model
    aggregator = aggregator.to(device).eval()
    point_head = point_head.to(device).eval()
    if confidence_head is point_head:
        confidence_head = point_head
    else:
        confidence_head = confidence_head.to(device).eval()
    return aggregator, point_head, confidence_head


def point_head_feature_dim(point_head):
    """Width of the pretrained latent consumed by the point prediction head."""
    output_conv2 = point_head.scratch.output_conv2
    for module in output_conv2.modules():
        if isinstance(module, torch.nn.Conv2d):
            return module.in_channels
    raise ValueError('Unable to infer the VGGT point-head latent dimension')


def feature_view_weights(confidence, aggregation):
    """Return weights used to combine a point's descriptors across views."""
    if aggregation == 'uniform':
        return torch.ones_like(confidence)
    if aggregation == 'confidence':
        # VGGT confidence is positive and unbounded. log1p retains its ordering
        # while preventing an extreme score from dominating the feature mean.
        return torch.log1p(confidence)
    raise ValueError(f'Unknown view aggregation: {aggregation!r}')


@torch.no_grad()
def sample_point_head_latents(point_head, tokens_list, images, patch_start_idx,
                              sample_grids):
    """Sample the pretrained dense latent just before point-head XYZ decoding.

    ``DPTHead`` normally returns only XYZ and confidence. A forward pre-hook on
    its final prediction stack exposes the exact full-resolution latent without
    modifying the vendored VGGT package. Sampling inside the hook avoids
    retaining an ``[S, 128, H, W]`` tensor after the point-head forward.

    Args:
        sample_grids: One normalized ``grid_sample`` grid per frame. Each is
            shaped ``[1, 1, K, 2]``; ``None`` represents a frame with no visible
            scan points.

    Returns:
        A list of ``[K, C]`` sampled feature tensors and point confidence maps.
    """
    batch_size, num_frames = images.shape[:2]
    if batch_size != 1:
        raise ValueError('Step 7b dense feature sampling expects batch size 1')
    if len(sample_grids) != num_frames:
        raise ValueError(
            f'Expected {num_frames} sampling grids, got {len(sample_grids)}'
        )

    sampled_features = [None] * num_frames
    next_frame = 0
    expected_dim = point_head_feature_dim(point_head)

    def sample_latent(_module, inputs):
        nonlocal next_frame
        latent = inputs[0]
        if latent.ndim != 4 or latent.shape[1] != expected_dim:
            raise ValueError(
                'Unexpected point-head latent shape: '
                f'{tuple(latent.shape)} (expected [S, {expected_dim}, H, W])'
            )
        chunk_frames = latent.shape[0]
        if next_frame + chunk_frames > num_frames:
            raise ValueError('Point head emitted more frame features than expected')

        for local_idx in range(chunk_frames):
            frame_idx = next_frame + local_idx
            grid = sample_grids[frame_idx]
            if grid is None:
                sampled_features[frame_idx] = latent.new_empty((0, expected_dim))
            else:
                sampled_features[frame_idx] = F.grid_sample(
                    latent[local_idx:local_idx + 1],
                    grid,
                    mode='bilinear',
                    align_corners=False,
                )[0, :, 0].T
        next_frame += chunk_frames

    handle = point_head.scratch.output_conv2.register_forward_pre_hook(sample_latent)
    try:
        point_prediction, point_confidence = point_head(
            tokens_list,
            images=images,
            patch_start_idx=patch_start_idx,
        )
    finally:
        handle.remove()
    del point_prediction

    if next_frame != num_frames or any(feature is None for feature in sampled_features):
        raise RuntimeError(
            f'Point head exposed dense features for {next_frame}/{num_frames} frames'
        )
    return sampled_features, point_confidence


def load_trajectory(traj_path):
    """Parse hires_poses.traj into {timestamp: 4x4 camera-to-world}, as in step7."""
    poses = {}
    with open(traj_path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) != 7:
                continue
            ts = tokens[0]
            angle_axis = np.asarray([float(t) for t in tokens[1:4]])
            r_w_to_p = cv2.Rodrigues(angle_axis)[0]
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = r_w_to_p
            extrinsics[:3, 3] = [float(t) for t in tokens[4:7]]
            poses[ts] = np.linalg.inv(extrinsics)
    return poses


def nearest_pose(frame_ts, poses, time_thresh):
    if frame_ts in poses:
        return poses[frame_ts]
    closest = min(poses.keys(), key=lambda k: abs(float(k) - float(frame_ts)))
    if abs(float(closest) - float(frame_ts)) > time_thresh:
        return None
    return poses[closest]


def read_intrinsics(intrinsics_dir, video_id, frame_ts):
    path = os.path.join(intrinsics_dir, f'{video_id}_{frame_ts}.pincam')
    if not os.path.exists(path):
        candidates = [f for f in os.listdir(intrinsics_dir) if frame_ts in f and f.endswith('.pincam')]
        if not candidates:
            return None
        path = os.path.join(intrinsics_dir, candidates[0])
    w, h, fx, fy, hw, hh = np.loadtxt(path)
    return w, h, fx, fy, hw, hh


def preprocess_image(rgb, max_size=TARGET_SIZE):
    """Resize so the longer side is ~max_size with both dims multiples of 14; range [0, 1]."""
    h0, w0 = rgb.shape[:2]
    scale = max_size / max(w0, h0)
    new_w = max(PATCH_SIZE, int(round(w0 * scale / PATCH_SIZE)) * PATCH_SIZE)
    new_h = max(PATCH_SIZE, int(round(h0 * scale / PATCH_SIZE)) * PATCH_SIZE)
    resized = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    img = torch.from_numpy(resized).float().permute(2, 0, 1) / 255.0
    return img


def load_preprocessed_rgb(frame_path):
    bgr = cv2.imread(frame_path)
    if bgr is None:
        return None
    return preprocess_image(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))


def load_source_frame_data(frame_path, video_id, poses, intrinsics_dir,
                           depth_dir, pose_time_thresh, device):
    """Load the RGB tensor and exact geometry needed to lift its pixels."""
    frame_ts = frame_timestamp(frame_path, video_id)
    pose = nearest_pose(frame_ts, poses, pose_time_thresh)
    if pose is None:
        raise ValueError(
            f'No pose within {pose_time_thresh}s for source frame {frame_path}'
        )
    intrinsics = read_intrinsics(intrinsics_dir, video_id, frame_ts)
    if intrinsics is None:
        raise FileNotFoundError(f'No intrinsics for source frame {frame_path}')
    depth_path = os.path.join(depth_dir, f'{video_id}_{frame_ts}.png')
    depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise FileNotFoundError(f'No readable depth for source frame {frame_path}')
    image = load_preprocessed_rgb(frame_path)
    if image is None:
        raise FileNotFoundError(f'No readable RGB for source frame {frame_path}')
    depth_t = torch.from_numpy(depth.astype(np.float32) / 1000.0).to(device)
    return image, pose, intrinsics, depth_t


def project_points(points_t, cam2world, intrinsics, depth_t, vis_thres, device):
    """Project points into a frame; return continuous pixel coords and a visibility mask.

    Mirrors fusion_util.PointCloudToImageMapper.compute_mapping: in-bounds check
    plus depth-based occlusion test |d - z| <= vis_thres * d.
    """
    w0, h0, fx, fy, cx, cy = intrinsics
    w2c = torch.from_numpy(np.linalg.inv(cam2world)).float().to(device)
    p_cam = points_t @ w2c[:3, :3].T + w2c[:3, 3]
    z = p_cam[:, 2]
    safe_z = torch.where(z.abs() > 1e-6, z, torch.full_like(z, 1e-6))
    u = p_cam[:, 0] * fx / safe_z + cx
    v = p_cam[:, 1] * fy / safe_z + cy

    valid = (z > 1e-2) & (u >= 0) & (u < w0) & (v >= 0) & (v < h0)
    if depth_t is not None:
        dh, dw = depth_t.shape
        ui = (u * dw / w0).long().clamp(0, dw - 1)
        vi = (v * dh / h0).long().clamp(0, dh - 1)
        d = depth_t[vi, ui]
        valid &= (d > 0) & ((d - z).abs() <= vis_thres * d)
    return u, v, valid


@torch.no_grad()
def compute_source_frame_features(aggregator, point_head, confidence_head,
                                  proj_t, feature_dim, source_data,
                                  context_images, points, args, device):
    """Lift only frame-zero pixels while later VGGT frames provide context."""
    source_image, pose, intrinsics, depth_t = source_data
    image_list = [source_image, *context_images]
    images = torch.stack(image_list).to(device)
    points_t = torch.from_numpy(points).float().to(device)

    u, v, valid = project_points(
        points_t, pose, intrinsics, depth_t, args.vis_thres, device
    )
    idx = valid.nonzero(as_tuple=True)[0]
    features = torch.zeros(
        len(points), feature_dim, dtype=torch.float32, device=device
    )
    confidence_out = torch.zeros(len(points), dtype=torch.float32, device=device)
    view_count = torch.zeros(len(points), dtype=torch.float32, device=device)
    if idx.numel() == 0:
        return (features.cpu().numpy(), view_count.cpu().numpy(),
                confidence_out.cpu().numpy())

    w0, h0 = intrinsics[0], intrinsics[1]
    gx = 2.0 * (u[idx] + 0.5) / w0 - 1.0
    gy = 2.0 * (v[idx] + 0.5) / h0 - 1.0
    source_grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
    sample_grids = [source_grid] + [None] * len(context_images)

    autocast_dtype = torch.bfloat16 if (
        device.startswith('cuda') and torch.cuda.get_device_capability()[0] >= 8
    ) else torch.float16
    with torch.autocast(
            device_type=device.split(':')[0], dtype=autocast_dtype,
            enabled=device.startswith('cuda')):
        tokens_list, patch_start_idx = aggregator(images.unsqueeze(0))

    batched_images = images.unsqueeze(0)
    if args.feature_source == 'dpt':
        with torch.autocast(device_type=device.split(':')[0], enabled=False):
            sampled_features, point_confidence = sample_point_head_latents(
                point_head,
                tokens_list,
                batched_images,
                patch_start_idx,
                sample_grids,
            )
        sampled_feat = sampled_features[0]
        if confidence_head is point_head:
            confidence = point_confidence
        else:
            del point_confidence
            with torch.autocast(device_type=device.split(':')[0], enabled=False):
                confidence_pred, confidence = confidence_head(
                    tokens_list,
                    images=batched_images,
                    patch_start_idx=patch_start_idx,
                )
            del confidence_pred
    else:
        with torch.autocast(device_type=device.split(':')[0], enabled=False):
            confidence_pred, confidence = confidence_head(
                tokens_list,
                images=batched_images,
                patch_start_idx=patch_start_idx,
            )
        del confidence_pred
        _, _, img_h, img_w = images.shape
        hp, wp = img_h // PATCH_SIZE, img_w // PATCH_SIZE
        source_tokens = tokens_list[-1][0][0, patch_start_idx:, :].float()
        if proj_t is not None:
            source_tokens = source_tokens @ proj_t
        source_map = source_tokens.reshape(hp, wp, -1).permute(2, 0, 1)[None]
        sampled_feat = F.grid_sample(
            source_map, source_grid, mode='bilinear', align_corners=False
        )[0, :, 0].T

    source_confidence_map = confidence[0, 0][None, None].float()
    sampled_confidence = F.grid_sample(
        source_confidence_map,
        source_grid,
        mode='bilinear',
        align_corners=False,
    )[0, 0, 0]
    sampled_confidence = torch.nan_to_num(
        sampled_confidence, nan=1.0, posinf=1e6, neginf=1.0
    ).clamp_min(1.0)

    features[idx] = sampled_feat
    confidence_out[idx] = sampled_confidence
    view_count[idx] = 1.0
    return (features.cpu().numpy(), view_count.cpu().numpy(),
            confidence_out.cpu().numpy())


@torch.no_grad()
def compute_visit_features(aggregator, point_head, confidence_head, proj_t,
                           feature_dim, visit_dir, points, args, device):
    """Return features, view counts, and mean VGGT confidence for scan points."""
    points_t = torch.from_numpy(points).float().to(device)
    feat_sum = torch.zeros(
        len(points), feature_dim, dtype=torch.float32, device=device
    )
    feat_weight_sum = torch.zeros(len(points), dtype=torch.float32, device=device)
    conf_sum = torch.zeros(len(points), dtype=torch.float32, device=device)
    view_cnt = torch.zeros(len(points), dtype=torch.float32, device=device)

    autocast_dtype = torch.bfloat16 if (device.startswith('cuda') and
                                        torch.cuda.get_device_capability()[0] >= 8) else torch.float16

    video_ids = sorted(
        d for d in os.listdir(visit_dir)
        if os.path.isdir(os.path.join(visit_dir, d, 'hires_wide'))
    )
    for video_id in video_ids:
        video_dir = os.path.join(visit_dir, video_id)
        traj_path = os.path.join(video_dir, 'hires_poses.traj')
        if not os.path.exists(traj_path):
            tqdm.write(f'  [skip video] no trajectory: {traj_path}')
            continue
        poses = load_trajectory(traj_path)
        if not poses:
            continue

        frames = sorted(glob.glob(os.path.join(video_dir, 'hires_wide', '*.jpg')))
        if not frames:
            continue
        sel = np.unique(np.linspace(0, len(frames) - 1, args.frames_per_video).astype(int))
        frames = [frames[i] for i in sel]

        intrinsics_dir = os.path.join(video_dir, 'hires_wide_intrinsics')
        depth_dir = os.path.join(video_dir, 'hires_depth')

        prepared = []  # (image tensor, pose, intrinsics, depth tensor)
        for frame_path in frames:
            frame_ts = os.path.basename(frame_path).rsplit('.jpg', 1)[0].split('_')[1]
            pose = nearest_pose(frame_ts, poses, args.pose_time_thresh)
            if pose is None:
                continue
            intrinsics = read_intrinsics(intrinsics_dir, video_id, frame_ts)
            if intrinsics is None:
                continue
            depth_path = os.path.join(depth_dir, f'{video_id}_{frame_ts}.png')
            if not os.path.exists(depth_path):
                continue
            depth = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
            if depth is None:
                continue
            depth_t = torch.from_numpy(depth.astype(np.float32) / 1000.0).to(device)
            bgr = cv2.imread(frame_path)
            if bgr is None:
                continue
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            prepared.append((preprocess_image(rgb), pose, intrinsics, depth_t))

        for start in range(0, len(prepared), args.chunk_size):
            chunk = prepared[start:start + args.chunk_size]
            images = torch.stack([c[0] for c in chunk]).to(device)  # [S, 3, H, W]
            _, _, img_h, img_w = images.shape
            hp, wp = img_h // PATCH_SIZE, img_w // PATCH_SIZE

            # Projection is computed before the VGGT heads so the DPT hook can
            # sample each full-resolution latent immediately and release it.
            frame_samples = []  # (visible scan-point indices, normalized grid)
            for _, pose, intrinsics, depth_t in chunk:
                u, v, valid = project_points(
                    points_t, pose, intrinsics, depth_t, args.vis_thres, device
                )
                idx = valid.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    frame_samples.append((idx, None))
                    continue
                w0, h0 = intrinsics[0], intrinsics[1]
                gx = 2.0 * (u[idx] + 0.5) / w0 - 1.0
                gy = 2.0 * (v[idx] + 0.5) / h0 - 1.0
                grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
                frame_samples.append((idx, grid))

            with torch.autocast(device_type=device.split(':')[0], dtype=autocast_dtype,
                                enabled=device.startswith('cuda')):
                tokens_list, patch_start_idx = aggregator(images.unsqueeze(0))

            batched_images = images.unsqueeze(0)
            if args.feature_source == 'dpt':
                with torch.autocast(device_type=device.split(':')[0], enabled=False):
                    sampled_features, point_confidence = sample_point_head_latents(
                        point_head,
                        tokens_list,
                        batched_images,
                        patch_start_idx,
                        [grid for _, grid in frame_samples],
                    )
                if confidence_head is point_head:
                    confidence = point_confidence
                    del point_confidence
                else:
                    del point_confidence
                    with torch.autocast(device_type=device.split(':')[0], enabled=False):
                        confidence_pred, confidence = confidence_head(
                            tokens_list,
                            images=batched_images,
                            patch_start_idx=patch_start_idx,
                        )
                    del confidence_pred
                token_maps = None
            else:
                with torch.autocast(device_type=device.split(':')[0], enabled=False):
                    confidence_pred, confidence = confidence_head(
                        tokens_list,
                        images=batched_images,
                        patch_start_idx=patch_start_idx,
                    )
                del confidence_pred
                tokens = tokens_list[-1][0][:, patch_start_idx:, :].float()
                if proj_t is not None:
                    tokens = tokens @ proj_t
                token_maps = tokens.reshape(
                    len(chunk), hp, wp, -1
                ).permute(0, 3, 1, 2)
                sampled_features = None

            confidence_maps = confidence[0].unsqueeze(1).float()  # [S, 1, H, W]
            del tokens_list

            for si, (idx, grid) in enumerate(frame_samples):
                if idx.numel() == 0:
                    continue
                if sampled_features is not None:
                    sampled_feat = sampled_features[si]
                else:
                    sampled_feat = F.grid_sample(
                        token_maps[si:si + 1],
                        grid,
                        mode='bilinear',
                        align_corners=False,
                    )[0, :, 0].T
                sampled_conf = F.grid_sample(
                    confidence_maps[si:si + 1], grid, mode='bilinear', align_corners=False
                )[0, 0, 0]  # [K]
                sampled_conf = torch.nan_to_num(
                    sampled_conf, nan=1.0, posinf=1e6, neginf=1.0
                ).clamp_min(1.0)
                feature_weight = feature_view_weights(
                    sampled_conf, args.view_aggregation
                )
                feat_sum[idx] += sampled_feat * feature_weight.unsqueeze(-1)
                feat_weight_sum[idx] += feature_weight
                conf_sum[idx] += sampled_conf
                view_cnt[idx] += 1.0
                del sampled_feat, sampled_conf, feature_weight

            # Release per-chunk tensors before the next aggregator forward to
            # avoid retaining the full-resolution confidence map at peak memory.
            del images, batched_images, confidence, confidence_maps
            del token_maps, sampled_features, frame_samples
            if args.feature_source == 'patch':
                del tokens

    feats = feat_sum / feat_weight_sum.clamp(min=1e-6).unsqueeze(-1)
    mean_conf = conf_sum / view_cnt.clamp(min=1.0)
    return feats.cpu().numpy(), view_cnt.cpu().numpy(), mean_conf.cpu().numpy()


def source_extraction_signature(args, feature_dim, visit_id, video_id,
                                frame_ts):
    return {
        'version': SOURCE_CACHE_VERSION,
        'frame_mode': 'source',
        'visit_id': str(visit_id),
        'video_id': str(video_id),
        'frame_id': str(frame_ts),
        'feature_source': args.feature_source,
        'feature_dim': int(feature_dim),
        'confidence_head': args.confidence_head,
        'chunk_size': int(args.chunk_size),
        'context_stride': int(args.context_stride),
        'vis_thres': float(args.vis_thres),
        'pose_time_thresh': float(args.pose_time_thresh),
        'model': os.path.abspath(args.ckpt) if args.ckpt else args.hf_model,
        'projection_dim': (
            int(args.out_dim) if args.feature_source == 'patch' else None
        ),
        'projection_seed': (
            int(args.seed) if args.feature_source == 'patch' else None
        ),
    }


def sample_output_dir(sample_dir, args):
    rel = os.path.relpath(sample_dir, args.processed_dir)
    return os.path.join(args.feat_out_root, rel)


def source_output_paths(sample_dir, args):
    out_dir = sample_output_dir(sample_dir, args)
    return {
        'dir': out_dir,
        'feat': os.path.join(out_dir, args.feature_name),
        'conf': os.path.join(out_dir, args.confidence_name),
        'view_count': os.path.join(out_dir, args.view_count_name),
        'metadata': os.path.join(out_dir, args.metadata_name),
    }


def source_outputs_complete(paths, num_points, feature_dim, signature):
    required = (paths['feat'], paths['conf'], paths['view_count'], paths['metadata'])
    if not all(os.path.exists(path) for path in required):
        return False
    try:
        feature = np.load(paths['feat'], mmap_mode='r')
        confidence = np.load(paths['conf'], mmap_mode='r')
        view_count = np.load(paths['view_count'], mmap_mode='r')
        with open(paths['metadata']) as f:
            metadata = json.load(f)
    except (OSError, ValueError, json.JSONDecodeError):
        return False
    return (
        feature.shape == (num_points, feature_dim)
        and confidence.shape == (num_points,)
        and view_count.shape == (num_points,)
        and metadata.get('signature') == signature
    )


def write_source_outputs(paths, feature, confidence, view_count, metadata):
    os.makedirs(paths['dir'], exist_ok=True)
    np.save(paths['feat'], feature.astype(np.float16))
    np.save(paths['conf'], confidence.astype(np.float32))
    np.save(paths['view_count'], view_count.astype(np.uint16))
    tmp_metadata = f"{paths['metadata']}.tmp.{os.getpid()}"
    with open(tmp_metadata, 'w') as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
        f.write('\n')
    os.replace(tmp_metadata, paths['metadata'])


def prepare_source_sequence(visit_dir, video_id, frame_ts, args, device):
    video_dir = os.path.join(visit_dir, video_id)
    frame_paths = sorted(glob.glob(os.path.join(video_dir, 'hires_wide', '*.jpg')))
    if not frame_paths:
        raise FileNotFoundError(f'No hires_wide frames under {video_dir}')
    source_path = find_source_frame(frame_paths, video_id, frame_ts)

    traj_path = os.path.join(video_dir, 'hires_poses.traj')
    if not os.path.exists(traj_path):
        raise FileNotFoundError(f'No source video trajectory: {traj_path}')
    poses = load_trajectory(traj_path)
    if not poses:
        raise ValueError(f'No valid poses in {traj_path}')
    source_data = load_source_frame_data(
        source_path,
        video_id,
        poses,
        os.path.join(video_dir, 'hires_wide_intrinsics'),
        os.path.join(video_dir, 'hires_depth'),
        args.pose_time_thresh,
        device,
    )

    ranked_paths = select_source_sequence(
        frame_paths,
        source_path,
        len(frame_paths),
        args.context_stride,
    )[1:]
    context_images = []
    context_paths = []
    for path in ranked_paths:
        image = load_preprocessed_rgb(path)
        if image is None or image.shape != source_data[0].shape:
            continue
        context_images.append(image)
        context_paths.append(path)
        if len(context_images) >= args.chunk_size - 1:
            break
    return source_path, source_data, context_paths, context_images


def process_source_visit(aggregator, point_head, confidence_head, proj_t,
                         feature_dim, visit_id, visit_dir, visit_samples,
                         scan_points, args, device):
    """Materialize source-frame-specific descriptors for one visit."""
    groups = {}
    for sample_dir in visit_samples:
        try:
            parsed_visit, video_id, frame_ts = source_frame_spec(
                sample_dir, args.processed_dir
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            tqdm.write(f'[skip sample {sample_dir}] {exc}')
            continue
        if parsed_visit != visit_id:
            tqdm.write(
                f'[skip sample {sample_dir}] source metadata visit '
                f'{parsed_visit!r} != directory visit {visit_id!r}'
            )
            continue
        groups.setdefault((video_id, frame_ts), []).append(sample_dir)

    tree = cKDTree(scan_points)
    completed = 0
    for (video_id, frame_ts), sample_dirs in tqdm(
            groups.items(), desc=f'{visit_id} source frames', leave=False):
        signature = source_extraction_signature(
            args, feature_dim, visit_id, video_id, frame_ts
        )
        pending = []
        for sample_dir in sample_dirs:
            sample_pcd = o3d.io.read_point_cloud(
                os.path.join(sample_dir, 'filtered_point_cloud.ply')
            )
            sample_points = np.asarray(sample_pcd.points)
            paths = source_output_paths(sample_dir, args)
            if (not args.overwrite and source_outputs_complete(
                    paths, len(sample_points), feature_dim, signature)):
                completed += 1
                continue
            if len(sample_points) == 0:
                tqdm.write(f'[skip sample {sample_dir}] empty point cloud')
                continue
            dist, scan_idx = tree.query(sample_points, workers=-1)
            if dist.max() > 1e-3:
                tqdm.write(
                    f'[warn] {sample_dir}: max NN distance {dist.max():.4f} m '
                    '(points not exactly on the cropped scan)'
                )
            pending.append((sample_dir, paths, scan_idx))
        if not pending:
            continue

        try:
            source_path, source_data, context_paths, context_images = (
                prepare_source_sequence(
                    visit_dir, video_id, frame_ts, args, device
                )
            )
        except (OSError, ValueError) as exc:
            tqdm.write(
                f'[skip source {visit_id}/{video_id}/{frame_ts}] {exc}'
            )
            continue

        unique_scan_idx = np.unique(np.concatenate([
            scan_idx for _, _, scan_idx in pending
        ]))
        group_points = scan_points[unique_scan_idx]
        features, view_count, confidence = compute_source_frame_features(
            aggregator,
            point_head,
            confidence_head,
            proj_t,
            feature_dim,
            source_data,
            context_images,
            group_points,
            args,
            device,
        )
        metadata = {
            'signature': signature,
            'source_frame': os.path.basename(source_path),
            'context_frames': [os.path.basename(path) for path in context_paths],
            'sequence_length': 1 + len(context_paths),
            'descriptor': (
                f'source-frame {args.feature_source} descriptor sampled at each '
                'projected 3D point; '
                'context frames affect VGGT attention but are not averaged'
            ),
        }
        for sample_dir, paths, scan_idx in pending:
            rows = np.searchsorted(unique_scan_idx, scan_idx)
            write_source_outputs(
                paths,
                features[rows],
                confidence[rows],
                view_count[rows],
                metadata,
            )
            completed += 1
    tqdm.write(
        f'[visit {visit_id}] source-frame features ready for '
        f'{completed}/{len(visit_samples)} samples across {len(groups)} frames'
    )


def main():
    args = parse_args()
    device = args.device
    if args.feat_out_root is None:
        processed_parent = os.path.dirname(os.path.normpath(args.processed_dir))
        args.feat_out_root = os.path.join(
            processed_parent, 'vggt_features', args.split
        )
    print(f'VGGT feature output root: {args.feat_out_root}')

    split_dir = 'train_val_set' if args.split in ('train', 'val') else 'test_set'
    raw_root = os.path.join(args.data_root, split_dir)

    sample_dirs = find_sample_dirs(args.processed_dir)
    if not sample_dirs:
        print(f'No sample dirs with filtered_point_cloud.ply under {args.processed_dir}')
        return
    if args.require_nonempty_gt:
        num_before = len(sample_dirs)
        sample_dirs = [
            sample_dir for sample_dir in sample_dirs
            if np.count_nonzero(np.load(
                os.path.join(sample_dir, 'gt_mask_local.npy'), mmap_mode='r'
            )) > 0
        ]
        print(
            f'Non-empty-GT filter: kept {len(sample_dirs)}/{num_before} samples'
        )
        if not sample_dirs:
            print('No samples remain after --require_nonempty_gt; done.')
            return

    by_visit = {}
    for sample_dir in sample_dirs:
        visit_id = os.path.relpath(sample_dir, args.processed_dir).split(os.sep)[0]
        by_visit.setdefault(visit_id, []).append(sample_dir)
    if args.visit_id:
        by_visit = {k: v for k, v in by_visit.items() if k == args.visit_id}
    elif args.num_shards > 1:
        visit_ids = sorted(by_visit)
        assigned = set(visit_ids[args.shard::args.num_shards])
        by_visit = {
            visit_id: samples for visit_id, samples in by_visit.items()
            if visit_id in assigned
        }
    print(
        f'shard {args.shard}/{args.num_shards}: '
        f'{sum(len(v) for v in by_visit.values())} samples across '
        f'{len(by_visit)} visit(s)'
    )
    if not by_visit:
        print('No visits assigned to this shard; done.')
        return

    if args.feature_source == 'patch':
        proj = load_projection(args.cache_dir, args.out_dim, args.seed)
        proj_t = torch.from_numpy(proj).to(device) if proj is not None else None
    else:
        proj_t = None

    aggregator, point_head, confidence_head = load_vggt_feature_model(args, device)
    if args.feature_source == 'dpt':
        feature_dim = point_head_feature_dim(point_head)
    else:
        feature_dim = proj_t.shape[1] if proj_t is not None else TOKEN_DIM
    if args.frame_mode == 'source':
        mode_detail = (
            f'source + up to {args.chunk_size - 1} context frame(s), '
            'no view averaging'
        )
    else:
        mode_detail = f'view aggregation: {args.view_aggregation}'
    print(
        f'Frame mode: {args.frame_mode}, feature source: '
        f'{args.feature_source} ({feature_dim}D), {mode_detail}, '
        f'filename: {args.feature_name}'
    )
    if args.frame_mode == 'source' and args.overwrite_cache:
        print('Note: --overwrite_cache applies only to legacy visit-average mode; '
              'source outputs are controlled by --overwrite and their metadata manifests.')
    os.makedirs(args.cache_dir, exist_ok=True)

    for visit_id, visit_samples in tqdm(by_visit.items(), desc='visits'):
        visit_dir = os.path.join(raw_root, visit_id)
        laser_scan_path = os.path.join(visit_dir, f'{visit_id}_laser_scan.ply')
        crop_mask_path = os.path.join(visit_dir, f'{visit_id}_crop_mask.npy')
        if not (os.path.exists(laser_scan_path) and os.path.exists(crop_mask_path)):
            tqdm.write(f'[skip visit {visit_id}] missing laser scan or crop mask under {visit_dir}')
            continue

        crop_indices = np.where(np.load(crop_mask_path))[0]
        scan = o3d.io.read_point_cloud(laser_scan_path)
        points = np.asarray(scan.points)[crop_indices]

        if args.frame_mode == 'source':
            process_source_visit(
                aggregator,
                point_head,
                confidence_head,
                proj_t,
                feature_dim,
                visit_id,
                visit_dir,
                visit_samples,
                points,
                args,
                device,
            )
            continue

        aggregation_tag = (
            'cw' if args.view_aggregation == 'confidence' else 'uniform'
        )
        if args.feature_source == 'dpt':
            cache_name = (
                f'{visit_id}_vggt_dpt_pointlatent_d{feature_dim}_'
                f'{args.confidence_head}_{aggregation_tag}_v{DPT_CACHE_VERSION}.npz'
            )
        else:
            cache_name = (
                f'{visit_id}_vggt_pointfeat_{args.confidence_head}_d{feature_dim}_'
                f'seed{args.seed}_{aggregation_tag}_v{PATCH_CACHE_VERSION}.npz'
            )
        cache_path = os.path.join(args.cache_dir, cache_name)
        if os.path.exists(cache_path) and not args.overwrite_cache:
            cache = np.load(cache_path)
            feats = cache['feat'].astype(np.float32)
            view_cnt = cache['count']
            mean_conf = cache['conf'].astype(np.float32)
            if (feats.shape != (len(points), feature_dim)
                    or view_cnt.shape != (len(points),)
                    or mean_conf.shape != (len(points),)):
                tqdm.write(f'[visit {visit_id}] stale cache (points changed), recomputing')
                feats = None
        else:
            feats = None
        if feats is None:
            feats, view_cnt, mean_conf = compute_visit_features(
                aggregator,
                point_head,
                confidence_head,
                proj_t,
                feature_dim,
                visit_dir,
                points,
                args,
                device,
            )
            np.savez_compressed(cache_path, feat=feats.astype(np.float16),
                                count=view_cnt.astype(np.uint16),
                                conf=mean_conf.astype(np.float32))
        seen = (view_cnt > 0).mean()
        tqdm.write(f'[visit {visit_id}] {len(points)} scan points, {seen:.1%} seen by >=1 view')
        if seen == 0:
            tqdm.write(f'[visit {visit_id}] no visible points, skipping sample dirs')
            continue

        tree = cKDTree(points)
        for sample_dir in visit_samples:
            rel = os.path.relpath(sample_dir, args.processed_dir)
            out_dir = os.path.join(args.feat_out_root, rel)
            os.makedirs(out_dir, exist_ok=True)
            feat_path = os.path.join(out_dir, args.feature_name)
            conf_path = os.path.join(out_dir, args.confidence_name)
            view_count_path = os.path.join(out_dir, args.view_count_name)
            output_paths = (feat_path, conf_path, view_count_path)
            sample_pcd = o3d.io.read_point_cloud(os.path.join(sample_dir, 'filtered_point_cloud.ply'))
            sample_points = np.asarray(sample_pcd.points)
            if all(os.path.exists(path) for path in output_paths) and not args.overwrite:
                existing_feat = np.load(feat_path, mmap_mode='r')
                existing_conf = np.load(conf_path, mmap_mode='r')
                existing_count = np.load(view_count_path, mmap_mode='r')
                expected_points = len(sample_points)
                if (existing_feat.shape == (expected_points, feature_dim)
                        and existing_conf.shape == (expected_points,)
                        and existing_count.shape == (expected_points,)):
                    continue
                tqdm.write(
                    f'[sample {rel}] stale output shapes; rematerializing '
                    f'{args.feature_name}'
                )
            dist, nn_idx = tree.query(sample_points, workers=-1)
            if dist.max() > 1e-3:
                tqdm.write(f'[warn] {sample_dir}: max NN distance {dist.max():.4f} m '
                           '(points not exactly on the cropped scan)')
            np.save(feat_path, feats[nn_idx].astype(np.float16))
            np.save(conf_path, mean_conf[nn_idx].astype(np.float32))
            np.save(view_count_path, view_cnt[nn_idx].astype(np.uint16))

    print('Done.')


if __name__ == '__main__':
    main()
