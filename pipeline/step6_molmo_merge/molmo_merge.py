import os
import argparse
import numpy as np
import json
from PIL import Image
from tqdm import tqdm

MOLMO_ROOT_BASE = 'pipeline/step5_molmo_sam/molmo_output'
MOLMO_ROOT = MOLMO_ROOT_BASE
CROPINFO_ROOT = 'pipeline/step4_crop_images/seg_image_output/point_clipwithaffordance_output'
BIGIMG_ROOT = 'path/to/raw_data/val'  # unused: original image path comes from each crop json
MERGE_ROOT_BASE = 'pipeline/step6_molmo_merge/molmo_merge_output'
MERGE_ROOT = MERGE_ROOT_BASE


def resolve_cropinfo_dir(visit_id, video_id, desc_id, splits):
    for split in splits:
        candidate = os.path.join(CROPINFO_ROOT, split, visit_id, video_id, desc_id)
        if os.path.isdir(candidate):
            return candidate
    return None


def visits_in_splits(splits):
    """Set of visit_ids that belong to any of the given crop-info splits."""
    visits = set()
    for split in splits:
        split_dir = os.path.join(CROPINFO_ROOT, split)
        if os.path.isdir(split_dir):
            visits.update(os.listdir(split_dir))
    return visits


def merge_mask_to_bigimg(molmo_mask, crop_info, bigimg_shape):
    mask_big = np.zeros(bigimg_shape, dtype=molmo_mask.dtype)
    l, u, w, h = crop_info['left'], crop_info['upper'], crop_info['width'], crop_info['height']
    if molmo_mask.shape != (h, w):
        molmo_mask = np.array(Image.fromarray(molmo_mask).resize((w, h), resample=Image.NEAREST))
    mask_big[u:u+h, l:l+w] = molmo_mask
    return mask_big


def process_one_desc(visit_id, video_id, desc_id, files, splits):
    npz_files = [f for f in files if f.endswith('_mask_data.npz')]
    if not npz_files:
        return

    print(f"Processing {visit_id}/{video_id}/{desc_id}: {len(npz_files)} npz file(s)")

    cropinfo_dir = resolve_cropinfo_dir(visit_id, video_id, desc_id, splits)
    if cropinfo_dir is None:
        print(f"  No crop info dir for {visit_id}/{video_id}/{desc_id} in splits {splits}")
        return
    crop_jsons = [f for f in os.listdir(cropinfo_dir) if f.endswith('_crop.json')]
    if not crop_jsons:
        return

    print(f"  {len(crop_jsons)} crop json(s)")
    
    crop_json_map = {}
    for cj in crop_jsons:
        base = cj.replace('_crop.json', '')
        crop_json_map[base] = cj
    
    total_masks_processed = 0
    total_images_generated = 0
    
    for npz_file in npz_files:
        base_name = npz_file.replace('_mask_data.npz', '')
        if base_name.endswith('_crop'):
            base_name = base_name[:-5]
        crop_json = crop_json_map.get(base_name)
        if not crop_json:
            for k, v in crop_json_map.items():
                if k in base_name:
                    crop_json = v
                    break
        if not crop_json:
            print(f"No matching crop json for: {npz_file}")
            continue
        with open(os.path.join(cropinfo_dir, crop_json), 'r') as f:
            crop_info = json.load(f)
        bigimg_path = crop_info['original_image']
        bigimg = Image.open(bigimg_path)
        bigimg_shape = (bigimg.height, bigimg.width)
        npz_path = os.path.join(MOLMO_ROOT, visit_id, video_id, desc_id, npz_file)
        data = np.load(npz_path)
        masks = data['masks']
        if masks.ndim != 3:
            # empty failure marker from step5 (no points / SAM failure)
            print(f"    {npz_file}: empty marker, skipping")
            continue

        print(f"    {npz_file}: {masks.shape[0]} mask(s)")

        merged_masks = []
        for i in range(masks.shape[0]):
            merged_mask = merge_mask_to_bigimg(masks[i], crop_info, bigimg_shape)
            merged_masks.append(merged_mask)
        merged_masks = np.stack(merged_masks, axis=0)

        save_dir = os.path.join(MERGE_ROOT, visit_id, video_id, desc_id)
        os.makedirs(save_dir, exist_ok=True)

        bigimg_name = os.path.splitext(os.path.basename(crop_info['original_image']))[0]
        bigimg = Image.open(bigimg_path).convert('RGB')

        for i, mask in enumerate(merged_masks):
            bigimg_jpg_path = os.path.join(save_dir, f"{bigimg_name}_mask_{i:03d}.jpg")
            mask_img = Image.fromarray((mask > 0).astype(np.uint8) * 255).convert('L')
            bigimg_np = np.array(bigimg)
            mask_np = np.array(mask_img)
            bigimg_np[mask_np > 0] = [255, 0, 0]
            Image.fromarray(bigimg_np).save(bigimg_jpg_path)

        save_npz = os.path.join(save_dir, f"{bigimg_name}_mask_data.npz")
        np.savez_compressed(save_npz, masks=merged_masks)

        total_masks_processed += masks.shape[0]
        total_images_generated += masks.shape[0]

    print(f"  Done: {total_masks_processed} mask(s), {total_images_generated} viz image(s)")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge molmo/SAM crop masks back into full images.")
    parser.add_argument('--split', choices=['train', 'val'], required=True)
    parser.add_argument('--molmo_root', default=None,
                        help='Split-specific Step 5 input (default: .../molmo_output/<split>)')
    parser.add_argument('--merge_root', default=None,
                        help='Split-specific output (default: .../molmo_merge_output/<split>)')
    return parser.parse_args()


def main():
    global MOLMO_ROOT, MERGE_ROOT
    args = parse_args()
    MOLMO_ROOT = args.molmo_root or os.path.join(MOLMO_ROOT_BASE, args.split)
    MERGE_ROOT = args.merge_root or os.path.join(MERGE_ROOT_BASE, args.split)
    splits = (args.split,)
    if not os.path.isdir(MOLMO_ROOT):
        raise FileNotFoundError(f"Step 5 split input not found: {MOLMO_ROOT}")
    allowed_visits = visits_in_splits(splits)
    print(f"Split '{args.split}': {len(allowed_visits)} visit(s) in crop-info splits {splits}")

    for visit_id in tqdm(sorted(os.listdir(MOLMO_ROOT)), desc='visit_id'):
        visit_path = os.path.join(MOLMO_ROOT, visit_id)
        if not os.path.isdir(visit_path):
            continue
        if visit_id not in allowed_visits:
            continue
        for video_id in os.listdir(visit_path):
            video_path = os.path.join(visit_path, video_id)
            if not os.path.isdir(video_path):
                continue
            for desc_id in os.listdir(video_path):
                desc_path = os.path.join(video_path, desc_id)
                if not os.path.isdir(desc_path):
                    continue
                files = os.listdir(desc_path)
                process_one_desc(visit_id, video_id, desc_id, files, splits)

if __name__ == '__main__':
    main()
