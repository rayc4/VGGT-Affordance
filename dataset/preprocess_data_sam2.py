import os
import numpy as np
import open3d as o3d
import json
import sys
import argparse
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from dataset.AffordanceDataset import AffordanceDataset


OUTPUT_FILES = (
    "mask_result.json",
    "filtered_point_cloud.ply",
    "pred_mask_global.npy",
    "pred_mask_local.npy",
    "gt_mask_global.npy",
    "gt_mask_local.npy",
)


def outputs_exist(save_dir):
    return all(os.path.exists(os.path.join(save_dir, name)) for name in OUTPUT_FILES)


def load_visit_cache(dataset, split, visit_id, device):
    """Load per-visit assets once (laser scan PLY, crop mask, descriptions,
    annotations) so every description in the visit reuses them instead of
    re-reading the multi-hundred-MB laser scan per description."""
    descriptions = dataset.get_descriptions(visit_id)
    annotations = dataset.get_annotations(visit_id)
    crop_mask = dataset.get_crop_mask(visit_id)
    filtered_idx_list = np.where(crop_mask)[0]

    laser_scan_path = dataset.get_data_asset_path(
        split=split,
        data_asset_identifier="laser_scan_5mm",
        visit_id=visit_id
    )
    laser_scan = o3d.io.read_point_cloud(laser_scan_path)
    cropped_points = np.array(laser_scan.points)[filtered_idx_list]
    cropped_colors = np.array(laser_scan.colors)[filtered_idx_list]

    return {
        "descriptions": descriptions,
        "annotations": annotations,
        "crop_mask": crop_mask,
        "filtered_idx_list": filtered_idx_list,
        "all_points": torch.tensor(cropped_points, device=device, dtype=torch.float32),
        "cropped_colors": cropped_colors,
        "has_colors": cropped_colors.shape[0] > 0,
    }


def grouped_annotation_from_cache(cache, desc_id):
    """Same as AffordanceDataset.get_grouped_annotation (point_mapping=None),
    but reading crop mask / descriptions / annotations from the visit cache."""
    crop_mask = cache["crop_mask"]
    full_mask = np.zeros(crop_mask.shape[0])

    for desc in cache["descriptions"]:
        if desc["desc_id"] == desc_id:
            annot_list = desc["annot_id"]
            break

    for annot in cache["annotations"]:
        if annot["annot_id"] in annot_list and annot["label"] != "exclude":
            idxs = np.asarray(annot["indices"])
            full_mask[idxs] = 1
    return full_mask[crop_mask == 1]


def process_mask_and_pointcloud_from_json(mask_json_path, cache, save_dir, device, radius=0.1, num_points=8192):
    with open(mask_json_path, 'r') as f:
        mask_data = json.load(f)
    original_indices = mask_data['original_indices']
    desc_id = mask_data.get('desc_id', '')
    desc_text = ''
    annot_id = ''

    for desc in cache["descriptions"]:
        if desc['desc_id'] == desc_id:
            desc_text = desc['description']
            annot_id = desc.get('annot_id')
            break
    if desc_text == '' or annot_id == '':
        print(f"error: desc_id {desc_id} not in descriptions, skip")
        exit()
    gt_mask = grouped_annotation_from_cache(cache, desc_id)
    gt_mask = torch.tensor(gt_mask)

    filtered_idx_list = cache["filtered_idx_list"]
    cropped_indices = np.where(np.isin(filtered_idx_list, original_indices))[0]

    if len(cropped_indices) == 0:
        print(f"{mask_json_path} original_indices/cropped_indices empty, skip")
        return
    pred_mask = torch.zeros_like(gt_mask, dtype=torch.uint8)
    pred_mask[cropped_indices] = 1

    all_points = cache["all_points"]
    mask_points = all_points[cropped_indices]
    mean_xyz = mask_points.mean(dim=0)

    distances = torch.norm(all_points - mean_xyz, dim=1)
    within_radius_mask = (distances <= radius)
    if within_radius_mask.sum().item() < num_points:
        remaining_mask = ~within_radius_mask
        remaining_distances = distances[remaining_mask]
        remaining_indices = torch.where(remaining_mask)[0]
        num_to_add = num_points - within_radius_mask.sum().item()
        if num_to_add > 0 and len(remaining_distances) > 0:
            closest_indices = remaining_indices[torch.argsort(remaining_distances)[:num_to_add]]
            within_radius_mask[closest_indices] = True
    if within_radius_mask.sum().item() > num_points:
        selected_indices = torch.where(within_radius_mask)[0]
        selected_distances = distances[selected_indices]
        sorted_indices = torch.argsort(selected_distances)[:num_points]
        closest_indices = selected_indices[sorted_indices]
        new_within_radius_mask = torch.zeros_like(within_radius_mask, dtype=torch.bool, device=device)
        new_within_radius_mask[closest_indices] = True
        within_radius_mask = new_within_radius_mask
    filtered_points = all_points[within_radius_mask].cpu().numpy()
    filtered_point_cloud = o3d.geometry.PointCloud()
    filtered_point_cloud.points = o3d.utility.Vector3dVector(filtered_points)
    if cache["has_colors"]:
        filtered_colors = cache["cropped_colors"][within_radius_mask.cpu().numpy()]
        filtered_point_cloud.colors = o3d.utility.Vector3dVector(filtered_colors)
    os.makedirs(save_dir, exist_ok=True)
    result = {
        "desc_id": desc_id,
        "desc_text": desc_text,
        "annot_id": annot_id,
        "mask_point_num": len(filtered_point_cloud.points),
        "mean_xyz": mean_xyz.cpu().numpy().tolist(),
        "original_indices": np.where(within_radius_mask.cpu().numpy())[0].tolist()
    }
    json_path = os.path.join(save_dir, "mask_result.json")
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Saved json to {json_path}")
    ply_path = os.path.join(save_dir, "filtered_point_cloud.ply")
    o3d.io.write_point_cloud(ply_path, filtered_point_cloud)
    print(f"Saved ply to {ply_path}")
    pred_mask_global_path = os.path.join(save_dir, "pred_mask_global.npy")
    np.save(pred_mask_global_path, pred_mask.cpu().numpy())
    print(f"Saved pred_mask_global to {pred_mask_global_path}")
    pred_mask_local_path = os.path.join(save_dir, "pred_mask_local.npy")
    np.save(pred_mask_local_path, pred_mask[within_radius_mask.cpu().numpy()])
    print(f"Saved pred_mask_local to {pred_mask_local_path}")
    gt_mask_global_path = os.path.join(save_dir, "gt_mask_global.npy")
    np.save(gt_mask_global_path, gt_mask.cpu().numpy())
    print(f"Saved gt_mask_global to {gt_mask_global_path}")
    gt_mask_local_path = os.path.join(save_dir, "gt_mask_local.npy")
    np.save(gt_mask_local_path, gt_mask[within_radius_mask.cpu().numpy()])
    print(f"Saved gt_mask_local to {gt_mask_local_path}")


def main():
    parser = argparse.ArgumentParser(description='Chunk point cloud from mask_result.json')
    parser.add_argument('--root_dir', type=str, default='scenefun3d', help='Data root')
    parser.add_argument('--clip_dir', type=str, default=None,
                        help='Split-specific lift output (default: .../lift_output_organized/<split>)')
    parser.add_argument('--save_dir', type=str, default='scenefun3d/processed_sam2', help='Save root')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val'], help='Split')
    parser.add_argument('--radius', type=float, default=0.15, help='Radius')
    parser.add_argument('--num_points', type=int, default=8192, help='Num points')
    parser.add_argument('--num_shards', type=int, default=1, help='Split visits across N parallel workers')
    parser.add_argument('--shard', type=int, default=0, help='Shard index in [0, num_shards)')
    parser.add_argument('--overwrite', action='store_true', help='Reprocess descs even if all outputs exist')
    args = parser.parse_args()
    if not (0 <= args.shard < args.num_shards):
        parser.error(f'--shard must be in [0, {args.num_shards})')
    if args.clip_dir is None:
        args.clip_dir = os.path.join(
            'pipeline/step7_lift_3d/lift_output_organized', args.split
        )

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    splits = [args.split]
    for split in splits:
        split_dir = args.clip_dir
        if not os.path.exists(split_dir):
            print(f"Dir not found: {split_dir}")
            continue

        dataset = AffordanceDataset(
            root_dir=args.root_dir,
            split=args.split
        )
        allowed_visit_ids = set(dataset.get_visit_id())
        input_visit_ids = {
            d for d in os.listdir(split_dir)
            if os.path.isdir(os.path.join(split_dir, d))
        }
        ignored_visit_ids = sorted(input_visit_ids - allowed_visit_ids)
        if ignored_visit_ids:
            print(
                f"Ignored {len(ignored_visit_ids)} visit(s) outside benchmark split "
                f"'{split}': {ignored_visit_ids}"
            )
        visit_ids = sorted(input_visit_ids & allowed_visit_ids)
        visit_ids = visit_ids[args.shard::args.num_shards]
        print(f"Processing {len(visit_ids)} visit(s) (shard {args.shard}/{args.num_shards})")

        for visit_id in tqdm(visit_ids, desc=split):
            visit_path = os.path.join(split_dir, visit_id)
            # Collect pending descs first so fully-done visits skip the laser scan load.
            tasks = []
            num_done = 0
            for scan_id in sorted(os.listdir(visit_path)):
                scan_path = os.path.join(visit_path, scan_id)
                if not os.path.isdir(scan_path):
                    continue
                for desc_id in sorted(os.listdir(scan_path)):
                    mask_json_path = os.path.join(scan_path, desc_id, 'mask_result.json')
                    if not os.path.exists(mask_json_path):
                        continue
                    save_dir = os.path.join(args.save_dir, split, visit_id, scan_id, desc_id)
                    if not args.overwrite and outputs_exist(save_dir):
                        num_done += 1
                        continue
                    tasks.append((mask_json_path, save_dir))
            if num_done:
                print(f"{visit_id}: skipped {num_done} already-processed desc(s)")
            if not tasks:
                continue

            try:
                cache = load_visit_cache(dataset, split, visit_id, device)
            except Exception as e:
                print(f"Error loading visit {visit_id}: {str(e)}")
                continue

            for mask_json_path, save_dir in tasks:
                try:
                    process_mask_and_pointcloud_from_json(
                        mask_json_path, cache, save_dir, device, radius=args.radius, num_points=args.num_points
                    )
                except Exception as e:
                    print(f"Error processing {mask_json_path}: {str(e)}")
                    continue
            del cache
    print("All done.")

if __name__ == "__main__":
    main()
