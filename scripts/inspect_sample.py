"""Render one processed_sam2 sample as a colored point cloud for inspection.

Colors:
    light grey  surrounding scene (context)
    dark grey   the 8192-point crop the model actually sees
    blue        pred_mask_local  (SAM2 lift, what the crop is centered on)
    green       GT inside the crop  (gt_mask_local; absent on a missed crop)
    cyan        overlap between pred_mask_local and GT
    red         GT outside the crop (where the annotation really is)

Usage:
    python scripts/inspect_sample.py scenefun3d/processed_sam2/val/<visit>/<scan>/<desc>
    python scripts/inspect_sample.py <dir> --out /tmp/sample.ply   # write instead of show
    python scripts/inspect_sample.py <dir> --image /tmp/sample.png # headless three-view preview
    python scripts/inspect_sample.py <dir> --image-2d /tmp/sample_2d.png
    python scripts/inspect_sample.py <dir> --context 3.0           # more surrounding scene
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import open3d as o3d

CTX = [0.85, 0.85, 0.85]
CROP = [0.45, 0.45, 0.45]
PRED = [0.10, 0.40, 0.95]
GT_IN = [0.10, 0.75, 0.25]
OVERLAP = [0.10, 0.80, 0.90]
GT_OUT = [0.95, 0.15, 0.15]
PALETTE = np.asarray([CTX, CROP, PRED, GT_IN, OVERLAP, GT_OUT])


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


def build_cloud(all_pts, crop_idx, center, pred_local, gt_local, gt_outside,
                context):
    """Return uniquely indexed visualization points, colors, and categories."""
    selected = np.zeros(len(all_pts), dtype=bool)
    if context > 0:
        selected |= np.linalg.norm(all_pts - center, axis=1) <= context
    selected[crop_idx] = True
    selected[gt_outside] = True

    category = np.zeros(len(all_pts), dtype=np.uint8)
    category[crop_idx] = 1

    pred = pred_local.astype(bool)
    gt = gt_local.astype(bool)
    category[crop_idx[pred]] = 2
    category[crop_idx[gt]] = 3
    category[crop_idx[pred & gt]] = 4
    category[gt_outside] = 5

    indices = np.flatnonzero(selected)
    return all_pts[indices], PALETTE[category[indices]], category[indices]


def save_image(path, points, categories, title, max_context_points):
    """Write three orthographic projections without requiring a display."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        sys.exit('error: --image requires matplotlib')

    keep = categories != 0
    context_idx = np.flatnonzero(categories == 0)
    if len(context_idx) > max_context_points:
        context_idx = np.random.default_rng(0).choice(
            context_idx, max_context_points, replace=False
        )
    keep[context_idx] = True
    points = points[keep]
    categories = categories[keep]

    projections = [(0, 1, 'X', 'Y', 'top'),
                   (0, 2, 'X', 'Z', 'front'),
                   (1, 2, 'Y', 'Z', 'side')]
    labels = ['context', 'model crop', 'lifted prediction', 'GT in crop',
              'prediction / GT overlap', 'GT outside crop']
    sizes = [0.25, 0.6, 5.0, 5.0, 6.0, 5.0]
    alphas = [0.22, 0.45, 0.9, 0.9, 1.0, 0.9]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for ax, (a, b, xlabel, ylabel, view_name) in zip(axes, projections):
        for category_id in range(len(PALETTE)):
            mask = categories == category_id
            if np.any(mask):
                ax.scatter(points[mask, a], points[mask, b],
                           c=[PALETTE[category_id]], s=sizes[category_id],
                           alpha=alphas[category_id], linewidths=0,
                           rasterized=True)
        ax.set(xlabel=f'{xlabel} (m)', ylabel=f'{ylabel} (m)', title=view_name)
        ax.set_aspect('equal', adjustable='box')
        ax.grid(color='0.9', linewidth=0.5)

    handles = [Line2D([0], [0], marker='o', linestyle='', label=label,
                      markerfacecolor=color, markeredgecolor='none', markersize=7)
               for label, color in zip(labels, PALETTE)]
    fig.legend(handles=handles, loc='outside lower center', ncol=6,
               frameon=False)
    fig.suptitle(title, fontsize=11)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def rodrigues(rotation_vector):
    """Convert an angle-axis vector to a 3x3 rotation matrix."""
    theta = np.linalg.norm(rotation_vector)
    if theta < 1e-12:
        return np.eye(3)
    x, y, z = rotation_vector / theta
    skew = np.asarray([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(theta) * skew + (1 - np.cos(theta)) * (skew @ skew)


def load_source_frame(root, split, visit_id, video_id, frame_id):
    """Load RGB/depth/calibration for a processed sample's source frame."""
    split_dir = 'train_val_set' if split in ('train', 'val') else 'test_set'
    base = os.path.join(root, split_dir, visit_id, video_id)
    stem = f'{video_id}_{frame_id}'
    paths = {
        'rgb': os.path.join(base, 'hires_wide', stem + '.jpg'),
        'depth': os.path.join(base, 'hires_depth', stem + '.png'),
        'intrinsic': os.path.join(base, 'hires_wide_intrinsics', stem + '.pincam'),
        'trajectory': os.path.join(base, 'hires_poses.traj'),
    }
    for name, path in paths.items():
        if not os.path.isfile(path):
            sys.exit(f'error: missing source {name}: {path}')

    try:
        from PIL import Image
    except ImportError:
        sys.exit('error: --image-2d requires Pillow')

    rgb = np.asarray(Image.open(paths['rgb']).convert('RGB'))
    depth = np.asarray(Image.open(paths['depth']), dtype=np.float64) / 1000.0
    width, height, fx, fy, cx, cy = np.loadtxt(paths['intrinsic'])
    if rgb.shape[:2] != depth.shape or rgb.shape[1] != int(width) or rgb.shape[0] != int(height):
        sys.exit(
            f'error: inconsistent source dimensions: RGB {rgb.shape[1]}x{rgb.shape[0]}, '
            f'depth {depth.shape[1]}x{depth.shape[0]}, intrinsics {int(width)}x{int(height)}'
        )

    nearest = None
    target_time = float(frame_id)
    with open(paths['trajectory']) as trajectory:
        for line in trajectory:
            tokens = line.split()
            if len(tokens) != 7:
                continue
            delta = abs(float(tokens[0]) - target_time)
            if nearest is None or delta < nearest[0]:
                nearest = (delta, tokens)
    if nearest is None:
        sys.exit(f'error: no valid poses in {paths["trajectory"]}')

    tokens = nearest[1]
    world_to_camera = np.eye(4)
    world_to_camera[:3, :3] = rodrigues(np.asarray(tokens[1:4], dtype=float))
    world_to_camera[:3, 3] = np.asarray(tokens[4:7], dtype=float)
    intrinsic = np.asarray([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    return rgb, depth, intrinsic, world_to_camera, tokens[0], paths['rgb']


def project_visible_points(points, intrinsic, world_to_camera, depth,
                           visibility_threshold):
    """Project world points and apply the lifting pipeline's depth test."""
    homogeneous = np.column_stack([points, np.ones(len(points))])
    camera = (world_to_camera @ homogeneous.T).T[:, :3]
    z = camera[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.rint(camera[:, 0] * intrinsic[0, 0] / z + intrinsic[0, 2]).astype(int)
        y = np.rint(camera[:, 1] * intrinsic[1, 1] / z + intrinsic[1, 2]).astype(int)

    height, width = depth.shape
    inside = ((z > 0) & (x >= 0) & (y >= 0) & (x < width) & (y < height))
    visible = np.zeros(len(points), dtype=bool)
    inside_idx = np.flatnonzero(inside)
    if len(inside_idx):
        measured_depth = depth[y[inside_idx], x[inside_idx]]
        visible[inside_idx] = (
            (measured_depth > 0)
            & (np.abs(measured_depth - z[inside_idx])
               <= visibility_threshold * measured_depth)
        )
    return np.column_stack([x[visible], y[visible]]), visible


def save_reprojection(path, rgb, pixels, categories, title, point_size):
    """Overlay reprojected mask points on their original RGB frame."""
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        sys.exit('error: --image-2d requires matplotlib')

    labels = {2: 'lifted prediction only', 3: 'GT in crop only',
              4: 'prediction / GT overlap', 5: 'GT outside crop'}
    fig, ax = plt.subplots(figsize=(13.5, 10), constrained_layout=True)
    ax.imshow(rgb)
    for category_id in labels:
        mask = categories == category_id
        if np.any(mask):
            ax.scatter(pixels[mask, 0], pixels[mask, 1],
                       c=[PALETTE[category_id]], s=point_size, alpha=0.85,
                       edgecolors='white', linewidths=0.2)
    handles = [Line2D([0], [0], marker='o', linestyle='', label=label,
                      markerfacecolor=PALETTE[category_id], markeredgecolor='white',
                      markersize=8)
               for category_id, label in labels.items()]
    ax.legend(handles=handles, loc='lower center', bbox_to_anchor=(0.5, 0.01),
              ncol=4, framealpha=0.85)
    ax.set(title=title, xlabel='image x (px)', ylabel='image y (px)')
    ax.set_xlim(0, rgb.shape[1])
    ax.set_ylim(rgb.shape[0], 0)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('sample_dir', help='processed_sam2/<split>/<visit>/<scan>/<desc>')
    parser.add_argument('--root', default='scenefun3d', help='data root (default: scenefun3d)')
    parser.add_argument('--context', type=float, default=2.0, metavar='M',
                        help='metres of surrounding scene to keep (0 = none, default 2.0)')
    output = parser.add_mutually_exclusive_group()
    output.add_argument('--out', metavar='PATH',
                        help='write a .ply instead of opening a window')
    output.add_argument('--image', metavar='PATH',
                        help='write a headless three-view .png instead of opening a window')
    output.add_argument('--image-2d', metavar='PATH',
                        help='reproject mask points onto the original RGB image')
    parser.add_argument('--max-context-points', type=int, default=100_000, metavar='N',
                        help='maximum context points in --image output (default 100000)')
    parser.add_argument('--visibility-threshold', type=float, default=0.25, metavar='F',
                        help='relative depth tolerance for --image-2d (default 0.25)')
    parser.add_argument('--point-size', type=float, default=14.0, metavar='PX',
                        help='marker area for --image-2d (default 14)')
    args = parser.parse_args()

    if args.max_context_points < 0:
        parser.error('--max-context-points must be non-negative')
    if args.visibility_threshold < 0:
        parser.error('--visibility-threshold must be non-negative')
    if args.point_size <= 0:
        parser.error('--point-size must be positive')

    d = args.sample_dir.rstrip(os.sep)
    parts = d.split(os.sep)
    try:
        split, visit_id, video_id, sample_name = parts[-4:]
    except (IndexError, ValueError):
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

    # A 2D reprojection only needs the mask categories; skip the potentially
    # million-point scene context in that mode.
    cloud_context = 0 if args.image_2d else args.context
    pts, colors, categories = build_cloud(
        all_pts, crop_idx, center, pred_local, gt_local, gt_outside, cloud_context
    )
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(pts)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    if args.out:
        if not o3d.io.write_point_cloud(args.out, cloud):
            sys.exit(f'error: could not write {args.out}')
        print(f'\nWrote {len(pts)} points to {args.out}')
    elif args.image:
        pred = pred_local.astype(bool)
        gt = gt_local.astype(bool)
        union = np.logical_or(pred, gt).sum()
        iou = np.logical_and(pred, gt).sum() / union if union else float('nan')
        title = (f'{result["desc_text"]}  |  {visit_id} / {result["desc_id"]}  |  '
                 f'IoU {iou:.3f}')
        save_image(args.image, pts, categories, title, args.max_context_points)
        print(f'\nWrote three-view preview to {args.image}')
    elif args.image_2d:
        frame_match = re.search(r'__frame_(.+?)__mask_\d+$', sample_name)
        frame_id = result.get('frame_id')
        if frame_id is None and frame_match:
            frame_id = frame_match.group(1)
        if frame_id is None:
            sys.exit('error: frame timestamp is absent from mask_result.json and sample name')

        rgb, depth, intrinsic, world_to_camera, pose_time, rgb_path = load_source_frame(
            args.root, split, visit_id, video_id, str(frame_id)
        )
        mask_points = pts[categories >= 2]
        mask_categories = categories[categories >= 2]
        pixels, visible = project_visible_points(
            mask_points, intrinsic, world_to_camera, depth, args.visibility_threshold
        )
        visible_categories = mask_categories[visible]
        counts_3d = {i: int(np.count_nonzero(mask_categories == i)) for i in range(2, 6)}
        counts_2d = {i: int(np.count_nonzero(visible_categories == i)) for i in range(2, 6)}
        print(f'source RGB   : {rgb_path}')
        print(f'frame/pose  : {frame_id} / {pose_time}')
        print(f'visible 2D  : {len(pixels)} / {len(mask_points)} mask points')
        print(f'by category : 3D {counts_3d} -> visible {counts_2d}')
        title = (f'{result["desc_text"]}  |  frame {frame_id}  |  '
                 f'{len(pixels)}/{len(mask_points)} points visible')
        save_reprojection(
            args.image_2d, rgb, pixels, visible_categories, title, args.point_size
        )
        print(f'\nWrote source-frame reprojection to {args.image_2d}')
    else:
        print('\nOpening viewer (use --out PATH if you have no display)')
        o3d.visualization.draw_geometries([cloud], window_name=result['desc_text'][:60])


if __name__ == '__main__':
    main()
