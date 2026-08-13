"""Materialize uniformly averaged per-sample VGGT features from legacy caches.

The original Step 7b caches store one uniformly averaged feature array per
cropped visit scan in ``<visit>_vggt_pointfeat.npz``. This utility gathers those
features onto each processed frame crop and writes ``vggt_feat_uniform.npy``
beside the confidence-weighted feature/reliability bundle. It does not run VGGT.
"""

import argparse
import os

import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='scenefun3d')
    parser.add_argument('--split', required=True, choices=['train', 'val', 'test'])
    parser.add_argument('--processed_dir', required=True)
    parser.add_argument('--feat_root', default=None,
                        help='Common output root containing split subdirectories')
    parser.add_argument('--cache_dir', default='outputs/vggt_feat_cache')
    parser.add_argument('--output_name', default='vggt_feat_uniform.npy')
    parser.add_argument('--num_shards', type=int, default=1)
    parser.add_argument('--shard', type=int, default=0)
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        parser.error('--shard must be in [0, --num_shards)')
    if args.feat_root is None:
        processed_root = os.path.dirname(os.path.normpath(args.processed_dir))
        args.feat_root = os.path.join(processed_root, 'vggt_features')
    return args


def find_samples_by_visit(processed_dir):
    by_visit = {}
    for dirpath, _dirnames, filenames in os.walk(processed_dir):
        if 'filtered_point_cloud.ply' not in filenames:
            continue
        rel = os.path.relpath(dirpath, processed_dir)
        visit_id = rel.split(os.sep)[0]
        by_visit.setdefault(visit_id, []).append(dirpath)
    return {visit: sorted(paths) for visit, paths in sorted(by_visit.items())}


def load_cropped_scan(data_root, split, visit_id):
    split_dir = 'train_val_set' if split in ('train', 'val') else 'test_set'
    visit_dir = os.path.join(data_root, split_dir, visit_id)
    scan_path = os.path.join(visit_dir, f'{visit_id}_laser_scan.ply')
    crop_mask_path = os.path.join(visit_dir, f'{visit_id}_crop_mask.npy')
    if not os.path.isfile(scan_path) or not os.path.isfile(crop_mask_path):
        raise FileNotFoundError(f'missing scan or crop mask under {visit_dir}')
    crop_indices = np.flatnonzero(np.load(crop_mask_path))
    scan = o3d.io.read_point_cloud(scan_path)
    return np.asarray(scan.points)[crop_indices]


def main():
    args = parse_args()
    by_visit = find_samples_by_visit(args.processed_dir)
    visit_ids = sorted(by_visit)[args.shard::args.num_shards]
    output_split_root = os.path.join(args.feat_root, args.split)

    print(
        f'shard {args.shard}/{args.num_shards}: {len(visit_ids)} visits, '
        f'output={output_split_root}'
    )
    for visit_id in tqdm(visit_ids, desc=f'{args.split} shard {args.shard}'):
        cache_path = os.path.join(
            args.cache_dir, f'{visit_id}_vggt_pointfeat.npz'
        )
        if not os.path.isfile(cache_path):
            raise FileNotFoundError(f'legacy uniform cache missing: {cache_path}')
        with np.load(cache_path) as cache:
            features = cache['feat']
        if features.ndim != 2 or features.shape[1] != 256:
            raise ValueError(f'{cache_path}: expected feature shape [M, 256]')

        scan_points = load_cropped_scan(args.data_root, args.split, visit_id)
        if features.shape[0] != scan_points.shape[0]:
            raise ValueError(
                f'{cache_path}: {features.shape[0]} features for '
                f'{scan_points.shape[0]} cropped scan points'
            )
        tree = cKDTree(scan_points)

        for sample_dir in by_visit[visit_id]:
            rel = os.path.relpath(sample_dir, args.processed_dir)
            out_dir = os.path.join(output_split_root, rel)
            out_path = os.path.join(out_dir, args.output_name)
            if os.path.isfile(out_path) and not args.overwrite:
                existing = np.load(out_path, mmap_mode='r')
                if existing.shape == (8192, 256) and existing.dtype == np.float16:
                    continue

            sample_cloud = o3d.io.read_point_cloud(
                os.path.join(sample_dir, 'filtered_point_cloud.ply')
            )
            sample_points = np.asarray(sample_cloud.points)
            distances, indices = tree.query(sample_points, workers=-1)
            if distances.size and distances.max() > 1e-3:
                raise ValueError(
                    f'{sample_dir}: max nearest-neighbor distance '
                    f'{distances.max():.4f} m exceeds 1 mm'
                )
            os.makedirs(out_dir, exist_ok=True)
            tmp_path = f'{out_path}.tmp.{os.getpid()}.npy'
            np.save(tmp_path, features[indices].astype(np.float16, copy=False))
            os.replace(tmp_path, out_path)


if __name__ == '__main__':
    main()
