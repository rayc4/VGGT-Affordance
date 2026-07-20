"""Step 7b: lift VGGT aggregator tokens to per-point 3D features.

For every processed sample directory (any directory containing
``filtered_point_cloud.ply``), this script computes per-point VGGT features and
saves them as ``vggt_feat.npy`` ([N, out_dim], float16) next to the sample files
(or into a mirror tree via --feat_out_root).

Per visit, features are computed once on the cropped laser scan and cached:
  1. Sample frames from each video of the visit (hires_wide).
  2. Run the VGGT aggregator on the frame sequence; keep last-layer patch
     tokens ([S, P, 2C], 2C=2048), optionally reduced to --out_dim with a fixed
     seeded orthonormal projection (saved to the cache dir for reproducibility).
  3. Project the cropped laser-scan points into each frame with the hires
     poses/intrinsics (same convention as step7), test visibility against
     hires_depth, and bilinearly sample the token grid at the projected pixels.
  4. Average features over all views that see a point (zeros if unseen).
Then per sample directory, points are matched to the cropped scan with a
KD-tree (exact coordinates, sanity-checked) and the features are gathered.

Example:
    python pipeline/step7b_vggt_feats/extract_vggt_features.py \
        --data_root scenefun3d --split train \
        --processed_dir data/processed_data/division_8192/train

    python pipeline/step7b_vggt_feats/extract_vggt_features.py \
        --data_root scenefun3d --split val \
        --processed_dir data/processed_sam2/val
"""
import argparse
import glob
import os
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


def parse_args():
    parser = argparse.ArgumentParser(description='Lift VGGT tokens to per-point 3D features')
    parser.add_argument('--data_root', type=str, default='scenefun3d',
                        help='SceneFun3D root containing train_val_set/ (e.g. scenefun3d)')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'],
                        help='Split of the visits (train/val -> train_val_set, test -> test_set)')
    parser.add_argument('--processed_dir', type=str, required=True,
                        help='Processed sample tree, e.g. data/processed_data/division_8192/train')
    parser.add_argument('--feat_out_root', type=str, default=None,
                        help='Optional mirror tree for vggt_feat.npy (default: next to sample files)')
    parser.add_argument('--cache_dir', type=str, default='outputs/vggt_feat_cache',
                        help='Per-visit feature cache directory')
    parser.add_argument('--hf_model', type=str, default='facebook/VGGT-1B')
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Local VGGT checkpoint (overrides --hf_model)')
    parser.add_argument('--out_dim', type=int, default=256,
                        help='Feature dim after fixed random orthonormal projection '
                             f'(0 or >={TOKEN_DIM} keeps the raw {TOKEN_DIM}-dim tokens)')
    parser.add_argument('--frames_per_video', type=int, default=24)
    parser.add_argument('--chunk_size', type=int, default=12,
                        help='Frames per VGGT forward pass (one sequence per chunk)')
    parser.add_argument('--vis_thres', type=float, default=0.25,
                        help='Depth visibility threshold (|d - z| <= thres * d), as in fusion_util')
    parser.add_argument('--pose_time_thresh', type=float, default=0.1,
                        help='Max |frame_ts - pose_ts| in seconds for nearest-pose lookup')
    parser.add_argument('--visit_id', type=str, default=None, help='Process a single visit')
    parser.add_argument('--seed', type=int, default=2023, help='Seed for the projection matrix')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--overwrite', action='store_true', help='Recompute existing vggt_feat.npy')
    parser.add_argument('--overwrite_cache', action='store_true', help='Recompute per-visit caches')
    return parser.parse_args()


def find_sample_dirs(processed_dir):
    sample_dirs = []
    for dirpath, _dirnames, filenames in os.walk(processed_dir):
        if 'filtered_point_cloud.ply' in filenames:
            sample_dirs.append(dirpath)
    return sorted(sample_dirs)


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
    np.save(proj_path, q)
    print(f'Saved projection matrix to {proj_path}')
    return q


def load_vggt_aggregator(args, device):
    if args.ckpt:
        model = VGGT()
        state = torch.load(args.ckpt, map_location='cpu')
        if isinstance(state, dict) and 'model' in state:
            state = state['model']
        model.load_state_dict(state, strict=False)
    else:
        model = VGGT.from_pretrained(args.hf_model)
    aggregator = model.aggregator
    # the heads (camera/depth/point/track) are not needed for token extraction
    del model
    return aggregator.to(device).eval()


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
def compute_visit_features(aggregator, proj_t, visit_dir, points, args, device):
    """Return ([M, out_dim] float32, [M] view counts) for the cropped scan points."""
    out_dim = proj_t.shape[1] if proj_t is not None else TOKEN_DIM
    points_t = torch.from_numpy(points).float().to(device)
    feat_sum = torch.zeros(len(points), out_dim, dtype=torch.float32, device=device)
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

            with torch.autocast(device_type=device.split(':')[0], dtype=autocast_dtype,
                                enabled=device.startswith('cuda')):
                tokens_list, patch_start_idx = aggregator(images.unsqueeze(0))
            tokens = tokens_list[-1][0][:, patch_start_idx:, :].float()  # [S, Hp*Wp, 2C]
            del tokens_list
            if proj_t is not None:
                tokens = tokens @ proj_t
            token_maps = tokens.reshape(len(chunk), hp, wp, -1).permute(0, 3, 1, 2)  # [S, D, Hp, Wp]

            for si, (_, pose, intrinsics, depth_t) in enumerate(chunk):
                u, v, valid = project_points(points_t, pose, intrinsics, depth_t,
                                             args.vis_thres, device)
                idx = valid.nonzero(as_tuple=True)[0]
                if idx.numel() == 0:
                    continue
                w0, h0 = intrinsics[0], intrinsics[1]
                gx = 2.0 * (u[idx] + 0.5) / w0 - 1.0
                gy = 2.0 * (v[idx] + 0.5) / h0 - 1.0
                grid = torch.stack([gx, gy], dim=-1).view(1, 1, -1, 2)
                sampled = F.grid_sample(token_maps[si:si + 1], grid, mode='bilinear',
                                        align_corners=False)  # [1, D, 1, K]
                feat_sum[idx] += sampled[0, :, 0].T
                view_cnt[idx] += 1.0

    feats = feat_sum / view_cnt.clamp(min=1.0).unsqueeze(-1)
    return feats.cpu().numpy(), view_cnt.cpu().numpy()


def main():
    args = parse_args()
    device = args.device

    split_dir = 'train_val_set' if args.split in ('train', 'val') else 'test_set'
    raw_root = os.path.join(args.data_root, split_dir)

    sample_dirs = find_sample_dirs(args.processed_dir)
    if not sample_dirs:
        print(f'No sample dirs with filtered_point_cloud.ply under {args.processed_dir}')
        return

    by_visit = {}
    for sample_dir in sample_dirs:
        visit_id = os.path.relpath(sample_dir, args.processed_dir).split(os.sep)[0]
        by_visit.setdefault(visit_id, []).append(sample_dir)
    if args.visit_id:
        by_visit = {k: v for k, v in by_visit.items() if k == args.visit_id}
    print(f'{sum(len(v) for v in by_visit.values())} samples across {len(by_visit)} visit(s)')

    proj = load_projection(args.cache_dir, args.out_dim, args.seed)
    proj_t = torch.from_numpy(proj).to(device) if proj is not None else None

    aggregator = load_vggt_aggregator(args, device)
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

        cache_path = os.path.join(args.cache_dir, f'{visit_id}_vggt_pointfeat.npz')
        if os.path.exists(cache_path) and not args.overwrite_cache:
            cache = np.load(cache_path)
            feats, view_cnt = cache['feat'].astype(np.float32), cache['count']
            if feats.shape[0] != len(points):
                tqdm.write(f'[visit {visit_id}] stale cache (points changed), recomputing')
                feats = None
        else:
            feats = None
        if feats is None:
            feats, view_cnt = compute_visit_features(aggregator, proj_t, visit_dir,
                                                     points, args, device)
            np.savez_compressed(cache_path, feat=feats.astype(np.float16),
                                count=view_cnt.astype(np.uint16))
        seen = (view_cnt > 0).mean()
        tqdm.write(f'[visit {visit_id}] {len(points)} scan points, {seen:.1%} seen by >=1 view')
        if seen == 0:
            tqdm.write(f'[visit {visit_id}] no visible points, skipping sample dirs')
            continue

        tree = cKDTree(points)
        for sample_dir in visit_samples:
            if args.feat_out_root is not None:
                rel = os.path.relpath(sample_dir, args.processed_dir)
                out_dir = os.path.join(args.feat_out_root, rel)
                os.makedirs(out_dir, exist_ok=True)
            else:
                out_dir = sample_dir
            out_path = os.path.join(out_dir, 'vggt_feat.npy')
            if os.path.exists(out_path) and not args.overwrite:
                continue

            sample_pcd = o3d.io.read_point_cloud(os.path.join(sample_dir, 'filtered_point_cloud.ply'))
            sample_points = np.asarray(sample_pcd.points)
            dist, nn_idx = tree.query(sample_points, workers=-1)
            if dist.max() > 1e-3:
                tqdm.write(f'[warn] {sample_dir}: max NN distance {dist.max():.4f} m '
                           '(points not exactly on the cropped scan)')
            np.save(out_path, feats[nn_idx].astype(np.float16))

    print('Done.')


if __name__ == '__main__':
    main()
