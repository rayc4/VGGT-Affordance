"""Render one processed_sam2 sample as a colored point cloud for inspection.

Colors:
    light grey  surrounding scene (context)
    dark grey   the 8192-point crop the model actually sees
    blue        pred_mask_local  (SAM2 lift, what the crop is centered on)
    green       GT inside the crop  (gt_mask_local; absent on a missed crop)
    red         GT outside the crop (where the annotation really is)

Usage:
    python scripts/inspect_sample.py scenefun3d/processed_sam2/val/<visit>/<scan>/<desc>
    python scripts/inspect_sample.py <dir> --out /tmp/sample.ply   # write instead of show
    python scripts/inspect_sample.py <dir> --context 3.0           # more surrounding scene
"""

import argparse
import json
import os
import sys

import numpy as np
import open3d as o3d

CTX = [0.85, 0.85, 0.85]
CROP = [0.45, 0.45, 0.45]
PRED = [0.10, 0.40, 0.95]
GT_IN = [0.10, 0.75, 0.25]
GT_OUT = [0.95, 0.15, 0.15]


def load_visit(root, split, visit_id):
    base = os.path.join(root, 'train_val_set', visit_id, visit_id)
    scan, crop = base + '_laser_scan.ply', base + '_crop_mask.npy'
    for path in (scan, crop):
        if not os.path.exists(path):
            sys.exit(f'error: missing {path}')
    cloud = o3d.io.read_point_cloud(scan)
    mask = np.load(crop)
    pts = np.asarray(cloud.points)
    if pts.shape[0] != mask.shape[0]:
        sys.exit(f'error: scan has {pts.shape[0]} points but crop mask has {mask.shape[0]}')
    return pts[mask == 1]


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sample_dir', help='processed_sam2/<split>/<visit>/<scan>/<desc>')
    parser.add_argument('--root', default='scenefun3d', help='data root (default: scenefun3d)')
    parser.add_argument('--context', type=float, default=2.0, metavar='M',
                        help='metres of surrounding scene to keep (0 = none, default 2.0)')
    parser.add_argument('--out', metavar='PATH', help='write a .ply instead of opening a window')
    args = parser.parse_args()

    d = args.sample_dir.rstrip(os.sep)
    parts = d.split(os.sep)
    try:
        split, visit_id = parts[-4], parts[-3]
    except IndexError:
        sys.exit('error: expected .../processed_sam2/<split>/<visit>/<scan>/<desc>')

    result = json.load(open(os.path.join(d, 'mask_result.json')))
    gt_global = np.asarray(np.load(os.path.join(d, 'gt_mask_global.npy'), mmap_mode='r'))
    pred_local = np.load(os.path.join(d, 'pred_mask_local.npy'))
    gt_local = np.load(os.path.join(d, 'gt_mask_local.npy'))

    all_pts = load_visit(args.root, split, visit_id)
    if all_pts.shape[0] != gt_global.shape[0]:
        sys.exit(f'error: cropped scan has {all_pts.shape[0]} points, '
                 f'gt_mask_global has {gt_global.shape[0]}')

    crop_idx = np.asarray(result['original_indices'])
    center = np.asarray(result['mean_xyz'])
    gt_idx = np.nonzero(gt_global)[0]
    gt_outside = np.setdiff1d(gt_idx, crop_idx)

    dist = (np.linalg.norm(all_pts[gt_idx] - center, axis=1).min()
            if len(gt_idx) else float('nan'))
    print(f'description : "{result["desc_text"]}"')
    print(f'visit/desc  : {visit_id} / {result["desc_id"]}')
    print(f'crop        : {len(crop_idx)} points, radius '
          f'{np.linalg.norm(all_pts[crop_idx] - center, axis=1).max():.3f} m')
    print(f'pred_local  : {int(pred_local.sum())} points')
    print(f'gt_local    : {int(gt_local.sum())} points'
          + ('   <-- EMPTY: the crop missed the annotation' if gt_local.sum() == 0 else ''))
    print(f'gt_global   : {len(gt_idx)} points')
    print(f'nearest GT point to crop centre: {dist:.3f} m')

    layers = []
    if args.context > 0:
        near = np.linalg.norm(all_pts - center, axis=1) <= args.context
        ctx_idx = np.setdiff1d(np.nonzero(near)[0], crop_idx)
        layers.append((all_pts[ctx_idx], CTX))
    layers.append((all_pts[crop_idx], CROP))
    layers.append((all_pts[crop_idx][pred_local.astype(bool)], PRED))
    layers.append((all_pts[crop_idx][gt_local.astype(bool)], GT_IN))
    layers.append((all_pts[gt_outside], GT_OUT))

    pts = np.vstack([p for p, _ in layers if len(p)])
    colors = np.vstack([np.tile(c, (len(p), 1)) for p, c in layers if len(p)])
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    if args.out:
        o3d.io.write_point_cloud(args.out, cloud)
        print(f'\nWrote {len(pts)} points to {args.out}')
    else:
        print('\nOpening viewer (use --out PATH if you have no display)')
        o3d.visualization.draw_geometries([cloud], window_name=result['desc_text'][:60])


if __name__ == '__main__':
    main()
