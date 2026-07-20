import os
import re
import sys
import csv
import importlib.machinery
import importlib.util
import numpy as np
import open3d as o3d
import torch
import cv2
import json
import argparse

# The SceneFun3D toolkit (DataParser, fusion utils) lives under scenefun3d/utils/
# (formerly the removed `scenefun3d_utils` package). TASA also ships its own
# top-level `utils/` package, which would shadow scenefun3d's on sys.path, so bind
# the name `utils` explicitly to the scenefun3d namespace package before importing.
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
_SCENEFUN3D_UTILS = os.path.join(REPO_ROOT, 'scenefun3d', 'utils')
_utils_mod = sys.modules.get('utils')
if _utils_mod is None or _SCENEFUN3D_UTILS not in list(getattr(_utils_mod, '__path__', [])):
    _spec = importlib.machinery.ModuleSpec('utils', loader=None, is_package=True)
    _spec.submodule_search_locations = [_SCENEFUN3D_UTILS]
    sys.modules['utils'] = importlib.util.module_from_spec(_spec)

from utils.data_parser import DataParser  # noqa: E402
from utils.fusion_util import PointCloudToImageMapper  # noqa: E402
from tqdm import tqdm


def load_split_visit_ids(data_root, split):
    split_path = os.path.join(data_root, "benchmark_file_lists", f"{split}_set.csv")
    if not os.path.isfile(split_path):
        raise FileNotFoundError(f"Benchmark split file not found: {split_path}")

    with open(split_path, newline="") as f:
        return {str(row["visit_id"]) for row in csv.DictReader(f)}


class MaskedPointCloudToImageMapper(PointCloudToImageMapper):
    """PointCloudToImageMapper with batched masked mapping.

    The official SceneFun3D toolkit ships only ``compute_mapping``; the batched
    ``compute_multi_masked_mapping`` used by this step lived in the (now removed)
    ``scenefun3d_utils`` package and is vendored here unchanged so results match.
    """

    def compute_multi_masked_mapping(
        self, camera_to_world, coords, mask_list, depth, intrinsic, device
    ):
        """
        Same thing as masked mapping, but batches a list of N masks for better efficiency!

        :param camera_to_world: 4 x 4
        :param coords: N x 3 format
        :param depth: H x W format
        :param intrinsic: 3x3 format
        :return: mapping, N x 3 format, (H,W,mask)
        """
        depth = torch.tensor(depth).to(device)
        mask_list = torch.tensor(mask_list).to(device)
        intrinsic = torch.tensor(intrinsic).to(device)
        camera_to_world = torch.tensor(camera_to_world).to(device)
        mapping = torch.zeros(
            (mask_list.shape[0], 3, coords.shape[0]), dtype=torch.int
        ).to(device)
        coords_new = torch.cat(
            [coords, torch.ones([coords.shape[0], 1]).to(device)], dim=1
        ).transpose(1, 0)
        assert coords_new.shape[0] == 4, "[!] Shape error"

        world_to_camera = torch.linalg.inv(camera_to_world)
        p = torch.matmul(world_to_camera, coords_new)
        p[0] = (p[0] * intrinsic[0][0]) / p[2] + intrinsic[0][2]
        p[1] = (p[1] * intrinsic[1][1]) / p[2] + intrinsic[1][2]
        pi = torch.round(p).to(torch.int)  # simply round the projected coordinates
        inside_mask = (
            (pi[0] >= self.cut_bound)
            * (pi[1] >= self.cut_bound)
            * (pi[0] < self.image_dim[0] - self.cut_bound)
            * (pi[1] < self.image_dim[1] - self.cut_bound)
        )

        for i, mask in enumerate(mask_list):
            _depth = depth.clone()
            _depth = torch.where(mask <= 0, 0, _depth)
            depth_cur = _depth[pi[1][inside_mask], pi[0][inside_mask]]
            occlusion_mask = (
                torch.abs(
                    depth[pi[1][inside_mask], pi[0][inside_mask]] - p[2][inside_mask]
                )
                <= self.vis_thres * depth_cur
            )
            inside_mask_i = inside_mask.clone()
            inside_mask_i[inside_mask == True] = occlusion_mask

            mapping[i][0][inside_mask_i] = pi[1][inside_mask_i]
            mapping[i][1][inside_mask_i] = pi[0][inside_mask_i]
            mapping[i][2][inside_mask_i] = 1

        return mapping.transpose(2, 1).cpu().numpy()

def load_visit_context(parser, data_root, split_dir, visit_id, device):
    """Load and crop a visit's laser scan once, to be reused across all its masks.

    Reading the laser-scan PLY (tens of MB, millions of points) and computing the
    crop is the dominant cost of this step, so it is done once per visit rather
    than once per mask file (previously re-read for every npz).
    """
    laser_scan_path = os.path.join(data_root, split_dir, visit_id, f"{visit_id}_laser_scan.ply")
    if not os.path.exists(laser_scan_path):
        print(f"Point cloud file not found: {laser_scan_path}")
        return None

    pcd = o3d.io.read_point_cloud(laser_scan_path)
    original_points = np.asarray(pcd.points)
    original_colors = np.asarray(pcd.colors)

    crop_indices = parser.get_crop_mask(visit_id, return_indices=True)
    points = original_points[crop_indices]

    proc_pcd = torch.tensor(points).to(device)

    return {
        'original_points': original_points,
        'original_colors': original_colors,
        'crop_indices': crop_indices,
        'proc_pcd': proc_pcd,
    }


def load_video_poses(data_root, split_dir, visit_id, video_id):
    """Parse a video's hires_poses.traj once, to be reused across all its masks."""
    traj_path = os.path.join(data_root, split_dir, visit_id, video_id, "hires_poses.traj")
    if not os.path.exists(traj_path):
        print(f"Trajectory file not found (download incomplete?): {traj_path}")
        return None

    poses = {}
    with open(traj_path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) == 7:
                ts = tokens[0]
                angle_axis = [float(tokens[1]), float(tokens[2]), float(tokens[3])]
                r_w_to_p = cv2.Rodrigues(np.asarray(angle_axis))[0]
                t_w_to_p = np.asarray([float(tokens[4]), float(tokens[5]), float(tokens[6])])
                extrinsics = np.eye(4, 4)
                extrinsics[:3, :3] = r_w_to_p
                extrinsics[:3, -1] = t_w_to_p
                Rt = np.linalg.inv(extrinsics)
                poses[ts] = Rt
    return poses


def process_one_molmo_mask(npz_path, parser, visit_ctx, poses, data_root, split, visit_id, video_id, desc_id, all_results, device):
    split_dir = "train_val_set" if split in ('train', 'val') else "test_set"

    original_points = visit_ctx['original_points']
    original_colors = visit_ctx['original_colors']
    crop_indices = visit_ctx['crop_indices']
    proc_pcd = visit_ctx['proc_pcd']

    npz = np.load(npz_path)
    masks = npz['masks']
    points_2d = npz['points'] if 'points' in npz else None

    print(f"Processing mask file: {npz_path}")
    print(f"Masks: {len(masks)}, 2D points: {len(points_2d) if points_2d is not None else 0}")

    basename = os.path.basename(npz_path)

    if basename.endswith('_crop_mask_data.npz'):
        if basename.startswith('frame'):
            name_part = basename.replace('frame', '').replace('_crop_mask_data.npz', '')
            parts = name_part.split('_')
            if len(parts) >= 3:
                frame_id = parts[-1]
            else:
                print(f"Cannot parse frame_id from: {basename}")
                return
        else:
            name_part = basename.replace('_crop_mask_data.npz', '')
            parts = name_part.split('_')
            if len(parts) >= 2:
                frame_id = parts[-1]
            else:
                print(f"Cannot parse frame_id from: {basename}")
                return
    elif basename.endswith('_mask_data.npz'):
        name_part = basename.replace('_mask_data.npz', '')
        parts = name_part.split('_')
        if len(parts) >= 2:
            frame_id = parts[-1]
        else:
            print(f"Cannot parse frame_id from: {basename}")
            return
    else:
        print(f"Unsupported filename format: {basename} (expected *_crop_mask_data.npz or *_mask_data.npz)")
        return

    depth_dir = os.path.join(data_root, split_dir, visit_id, video_id, "hires_depth")
    depth_path = os.path.join(depth_dir, f"{video_id}_{frame_id}.png")
    if not os.path.exists(depth_path):
        print(f"Depth file not found: {depth_path}")
        return
    
    depth = parser.read_depth_frame(depth_path)
    h, w = depth.shape

    intrinsics_dir = os.path.join(data_root, split_dir, visit_id, video_id, "hires_wide_intrinsics")
    intrinsics_files = [f for f in os.listdir(intrinsics_dir) if f.endswith('.pincam')]
    intrinsic_path = None
    for f in intrinsics_files:
        if frame_id in f:
            intrinsic_path = os.path.join(intrinsics_dir, f)
            break
    
    if intrinsic_path is None:
        print(f"Intrinsics not found: {intrinsics_dir} (frame_id={frame_id})")
        return

    intrinsic = parser.read_camera_intrinsics(intrinsic_path, format="matrix")

    pose = parser.get_nearest_pose(frame_id, poses)
    if pose is None:
        print(f"No suitable camera pose for frame_id: {frame_id}")
        return

    mask_threshold = 0.5

    for mask_idx, mask in enumerate(masks):
        print(f"Processing mask {mask_idx}")

        if mask.shape != (h, w):
            mask = cv2.resize(mask.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
        
        whole_mask = np.ones(depth.shape)

        mask_tensor = torch.tensor(mask).unsqueeze(0).unsqueeze(0).to(torch.float)
        whole_mask_tensor = torch.tensor(whole_mask).unsqueeze(0).unsqueeze(0).to(torch.float)
        if device == "cuda":
            mask_tensor = mask_tensor.cuda()
            whole_mask_tensor = whole_mask_tensor.cuda()
        
        mapper = MaskedPointCloudToImageMapper((w, h))
        mapping_fo = mapper.compute_multi_masked_mapping(
            pose,
            proc_pcd,
            torch.stack([mask_tensor.squeeze(), whole_mask_tensor.squeeze()], dim=0),
            depth,
            intrinsic,
            device
        )

        valid_f = mapping_fo[0, :, -1] == 1
        valid_count = np.count_nonzero(valid_f)

        if valid_count > 0:
            valid_fy = mapping_fo[0, valid_f, 0].astype(int)
            valid_fx = mapping_fo[0, valid_f, 1].astype(int)
            valid_point_indices = np.where(valid_f)[0]

            mask_values = mask[valid_fy, valid_fx]
            mask_points = mask_values > mask_threshold

            if np.sum(mask_points) > 0:
                mask_point_indices = valid_point_indices[mask_points]

                cropped_mask_indices = sorted(mask_point_indices)
                original_mask_indices = [crop_indices[idx] for idx in cropped_mask_indices]

                mask_points_3d = original_points[original_mask_indices]
                mask_colors_3d = original_colors[original_mask_indices]
                mean_xyz = np.mean(mask_points_3d, axis=0).tolist()

                result_data = {
                    "desc_id": desc_id,
                    "desc_text": f"Molmo mask segmentation for {desc_id} (mask {mask_idx})",
                    "annot_id": [f"molmo_mask_{mask_idx}"],
                    "mask_point_num": int(len(original_mask_indices)),
                    "mean_xyz": [float(x) for x in mean_xyz],
                    "original_indices": [int(idx) for idx in original_mask_indices],
                    "frame_id": frame_id,
                    "mask_idx": mask_idx,
                    "source": "molmo_sam",
                    "input_file": npz_path,
                    "visit_id": visit_id,
                    "video_id": video_id
                }

                if points_2d is not None and len(points_2d) > mask_idx:
                    result_data["points_2d"] = points_2d[mask_idx].tolist() if hasattr(points_2d[mask_idx], 'tolist') else points_2d[mask_idx]

                all_results.append(result_data)

                print(f"Processed mask {mask_idx}, points: {len(original_mask_indices):,}")
            else:
                print(f"Mask {mask_idx} has no points above threshold {mask_threshold}")
        else:
            print(f"Mask {mask_idx} has no valid point cloud projection")

    print(f"Done: {npz_path}")


def process_desc_directory(desc_dir, parser, visit_ctx, poses, data_root, split, visit_id, video_id, desc_id, all_results, device):
    if not os.path.exists(desc_dir):
        print(f"Warning: desc dir not found: {desc_dir}")
        return

    npz_files = [f for f in os.listdir(desc_dir) if f.endswith('.npz')]

    if len(npz_files) == 0:
        print(f"Warning: no npz files in {desc_dir}")
        return

    print(f"Processing {desc_id}: {len(npz_files)} npz files")

    for npz_file in npz_files:
        npz_path = os.path.join(desc_dir, npz_file)
        process_one_molmo_mask(npz_path, parser, visit_ctx, poses, data_root, split, visit_id, video_id, desc_id, all_results, device)


def merge_visit_results(output_dir, allowed_visit_ids=None):
    """Concatenate per-visit lift_results_<visit_id>.json files into lift_results_all.json."""
    pattern = re.compile(r'^lift_results_(\d+)\.json$')
    names = []
    for name in os.listdir(output_dir):
        match = pattern.match(name)
        if match and (allowed_visit_ids is None or match.group(1) in allowed_visit_ids):
            names.append(name)
    names.sort()
    merged = []
    for name in names:
        with open(os.path.join(output_dir, name)) as f:
            merged.extend(json.load(f))
    out_path = os.path.join(output_dir, 'lift_results_all.json')
    with open(out_path, 'w') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)
    print(f"Merged {len(names)} visit file(s), {len(merged)} mask(s) -> {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Molmo 2D-to-3D mask lifting tool')
    parser.add_argument('--data_root', type=str, default='scenefun3d', help='Path to data root')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'], help='Dataset split')
    parser.add_argument('--molmo_root', type=str, default=None,
                        help='Split-specific Step 6 input (default: .../molmo_merge_output/<split>)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Split-specific output (default: .../lift_output/<split>)')
    parser.add_argument('--visit_id', type=str, default=None, help='Specific visit_id (default: all)')
    parser.add_argument('--video_id', type=str, default=None, help='Specific video_id (default: all)')
    parser.add_argument('--desc_id', type=str, default=None, help='Specific desc_id (default: all)')
    parser.add_argument('--num_shards', type=int, default=1, help='Split visits across N parallel workers')
    parser.add_argument('--shard', type=int, default=0, help='Shard index in [0, num_shards)')
    parser.add_argument('--overwrite', action='store_true', help='Recompute visits whose lift_results_<visit_id>.json already exists')
    parser.add_argument('--merge_only', action='store_true', help='Only merge per-visit results into lift_results_all.json, then exit')
    args = parser.parse_args()

    if not (0 <= args.shard < args.num_shards):
        parser.error(f'--shard must be in [0, {args.num_shards})')

    if args.molmo_root is None:
        args.molmo_root = os.path.join(
            'pipeline/step6_molmo_merge/molmo_merge_output', args.split
        )
    if args.output_dir is None:
        args.output_dir = os.path.join(
            'pipeline/step7_lift_3d/lift_output', args.split
        )

    if args.merge_only:
        allowed_visit_ids = load_split_visit_ids(args.data_root, args.split)
        merge_visit_results(args.output_dir, allowed_visit_ids)
        return

    data_root = args.data_root
    split = args.split
    molmo_root = args.molmo_root
    output_dir = args.output_dir

    split_dir = "train_val_set" if split in ('train', 'val') else "test_set"
    data_parser = DataParser(os.path.join(data_root, split_dir))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    allowed_visit_ids = load_split_visit_ids(data_root, split)

    if not os.path.isdir(molmo_root):
        raise FileNotFoundError(f"Step 6 split input not found: {molmo_root}")

    if args.visit_id:
        if str(args.visit_id) not in allowed_visit_ids:
            parser.error(
                f"visit_id {args.visit_id} does not belong to benchmark split '{split}'"
            )
        visit_ids = [args.visit_id]
    else:
        input_visit_ids = {
            d for d in os.listdir(molmo_root)
            if os.path.isdir(os.path.join(molmo_root, d))
        }
        ignored_visit_ids = sorted(input_visit_ids - allowed_visit_ids)
        if ignored_visit_ids:
            print(
                f"Ignored {len(ignored_visit_ids)} visit(s) outside benchmark split "
                f"'{split}': {ignored_visit_ids}"
            )
        visit_ids = list(input_visit_ids & allowed_visit_ids)
        visit_ids = sorted(visit_ids)
    visit_ids = visit_ids[args.shard::args.num_shards]

    print(f"Processing {len(visit_ids)} visit_id(s) (shard {args.shard}/{args.num_shards}): {visit_ids}")

    # Runs filtered to a specific video/desc write to a distinct filename and
    # bypass the per-visit checkpoints, so partial results never mask a full run.
    filtered_run = bool(args.video_id or args.desc_id)
    os.makedirs(output_dir, exist_ok=True)

    all_results = []
    total_masks = 0
    visits_done = 0

    for visit_id in tqdm(visit_ids, desc='visit_id'):
        visit_out_path = os.path.join(output_dir, f"lift_results_{visit_id}.json")
        if not filtered_run and not args.overwrite and os.path.exists(visit_out_path):
            print(f"Skipping visit {visit_id}: {visit_out_path} exists (use --overwrite to redo)")
            continue

        visit_dir = os.path.join(molmo_root, visit_id)
        if not os.path.exists(visit_dir):
            print(f"Warning: visit dir not found: {visit_dir}")
            continue

        # Load and crop the visit's laser scan once, reused across all its masks.
        visit_ctx = load_visit_context(data_parser, data_root, split_dir, visit_id, device)
        if visit_ctx is None:
            continue

        visit_results = []

        if args.video_id:
            video_ids = [args.video_id]
        else:
            video_ids = [d for d in os.listdir(visit_dir)
                        if os.path.isdir(os.path.join(visit_dir, d))]

        print(f"Visit {visit_id}: {len(video_ids)} video(s)")

        for video_id in video_ids:
            video_dir = os.path.join(visit_dir, video_id)
            if not os.path.exists(video_dir):
                print(f"Warning: video dir not found: {video_dir}")
                continue

            # Parse the video's trajectory once, reused across all its masks.
            poses = load_video_poses(data_root, split_dir, visit_id, video_id)
            if not poses:
                continue

            if args.desc_id:
                desc_ids = [args.desc_id]
            else:
                desc_ids = [d for d in os.listdir(video_dir)
                           if os.path.isdir(os.path.join(video_dir, d))]

            for desc_id in desc_ids:
                desc_dir = os.path.join(video_dir, desc_id)
                process_desc_directory(desc_dir, data_parser, visit_ctx, poses, data_root, split, visit_id, video_id, desc_id, visit_results, device)

        if filtered_run:
            all_results.extend(visit_results)
        else:
            # Per-visit checkpoint: reruns skip this visit once the file exists.
            with open(visit_out_path, 'w') as f:
                json.dump(visit_results, f, indent=2, ensure_ascii=False)
            print(f"Visit {visit_id}: {len(visit_results)} mask(s) -> {visit_out_path}")

        total_masks += len(visit_results)
        visits_done += 1

        # Release the visit's GPU point cloud before moving to the next visit.
        visit_ctx = None
        if device == "cuda":
            torch.cuda.empty_cache()

    if filtered_run:
        if all_results:
            name_parts = [args.visit_id or 'all', args.video_id, args.desc_id]
            output_filename = 'lift_results_' + '_'.join(p for p in name_parts if p) + '.json'
            output_path = os.path.join(output_dir, output_filename)
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"Done. Processed {len(all_results)} mask(s). Results saved to: {output_path}")
        else:
            print("No valid mask data processed.")
    else:
        print(f"Done. Processed {total_masks} mask(s) across {visits_done} visit(s); per-visit results in {output_dir}")
        if args.num_shards == 1 and not args.visit_id:
            merge_visit_results(output_dir, allowed_visit_ids)
        else:
            print("Sharded/partial run: run with --merge_only afterwards to build lift_results_all.json")

    print("All done.")


if __name__ == '__main__':
    main()
