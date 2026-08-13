import os
import argparse
import json
from PIL import Image, ImageDraw
import torch
import torchvision.transforms.functional as TF
from tqdm import tqdm

def crop_image_gpu(image_path, center_x, center_y, width, height, output_dir, image_name_prefix=None, device='cuda'):
    image = Image.open(image_path).convert('RGB')
    img_w, img_h = image.size
    img_tensor = TF.to_tensor(image).to(device)
    left = int(center_x - width / 2)
    upper = int(center_y - height / 2)
    right = int(center_x + width / 2)
    lower = int(center_y + height / 2)
    left = max(0, left)
    upper = max(0, upper)
    right = min(img_w, right)
    lower = min(img_h, lower)
    cropped = img_tensor[:, upper:lower, left:right]
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.basename(image_path)
    name, ext = os.path.splitext(base_name)
    if image_name_prefix:
        name = image_name_prefix + '_' + name
    output_path = os.path.join(output_dir, f"{name}_crop{ext}")
    cropped_pil = TF.to_pil_image(cropped.cpu())
    cropped_pil.save(output_path)
    meta = {
        "left": left,
        "upper": upper,
        "width": right - left,
        "height": lower - upper,
        "original_image": image_path
    }
    meta_path = os.path.splitext(output_path)[0] + '.json'
    with open(meta_path, 'w') as f:
        import json
        json.dump(meta, f)
    marked_image = image.copy()
    draw = ImageDraw.Draw(marked_image)
    r = max(2, min(width, height) // 20)
    draw.ellipse((center_x - r, center_y - r, center_x + r, center_y + r), fill='red', outline='red')
    draw.rectangle((left, upper, right, lower), outline='green', width=3)
    marked_output_path = os.path.join(output_dir, f"{name}_marked{ext}")
    marked_image.save(marked_output_path)

def process_all_images(input_dir, split, width, height, output_dir, raw_data_root, device):
    json_files = []
    for visit_id in os.listdir(input_dir):
        visit_path = os.path.join(input_dir, visit_id)
        if not os.path.isdir(visit_path):
            continue
        for fname in os.listdir(visit_path):
            if fname.endswith('_point.json'):
                json_files.append((visit_id, fname))
    total_frames = 0
    for visit_id, fname in json_files:
        video_id = fname.replace('_point.json', '')
        json_path = os.path.join(input_dir, visit_id, fname)
        with open(json_path, 'r') as f:
            data = json.load(f)
        for desc_data in data:
            frame_results = desc_data.get('frame_results', [])
            for frame in frame_results:
                if frame.get('object_found', False) and frame.get('coordinates'):
                    total_frames += 1
    pbar = tqdm(total=total_frames, desc='Cropping images')
    for visit_id, fname in json_files:
        video_id = fname.replace('_point.json', '')
        json_path = os.path.join(input_dir, visit_id, fname)
        with open(json_path, 'r') as f:
            data = json.load(f)
        for desc_data in data:
            desc_id = desc_data['desc_id']
            frame_results = desc_data.get('frame_results', [])
            for idx, frame in enumerate(frame_results):
                if not frame.get('object_found', False):
                    continue
                coordinates = frame['coordinates']
                if not coordinates:
                    continue
                image_name = frame['image_name']
                center_x, center_y = coordinates['x'], coordinates['y']
                raw_split_dir = "train_val_set" if split in ('train', 'val') else "test_set"
                raw_img_path = os.path.join(raw_data_root, raw_split_dir, visit_id, video_id, "hires_wide", image_name)
                out_dir = os.path.join(output_dir, visit_id, video_id, desc_id)
                image_prefix = f"frame{idx}"
                try:
                    crop_image_gpu(raw_img_path, center_x, center_y, width, height, out_dir, image_name_prefix=image_prefix, device=device)
                except Exception as e:
                    print(f"Crop failed: {raw_img_path}, error: {e}")
                pbar.update(1)
    pbar.close()


def resolve_io_directories(args):
    if args.input_root:
        input_base = args.input_root
        input_name = os.path.basename(os.path.normpath(args.input_root))
    else:
        input_base = args.data_root
        input_name = os.path.basename(os.path.normpath(args.data_root))
    input_dir = os.path.join(input_base, args.split)

    if args.output_root:
        output_base = args.output_root
    else:
        output_base = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'seg_image_output',
            input_name,
        )
    output_dir = os.path.join(output_base, args.split)
    return input_dir, output_dir


def main():
    parser = argparse.ArgumentParser(description="Crop image regions centered at given points (GPU supported)")
    parser.add_argument(
        '--input_root', '--input-root', type=str, default=None,
        help='Step 3 root containing <split>/<visit_id>/*_point.json',
    )
    parser.add_argument(
        '--output_root', '--output-root', type=str, default=None,
        help='Output root; cropped images and metadata go under <split>/',
    )
    parser.add_argument(
        '--data_root', type=str,
        default='pipeline/step3_point_prediction/point_clipwithaffordance_output',
        help='Legacy Step 3 root; <split> is appended (ignored with --input_root)',
    )
    parser.add_argument('--raw_data_root', type=str, default='scenefun3d', help='Path to raw data (images)')
    parser.add_argument('--split', type=str, required=True, choices=['train', 'val', 'test'], help='Dataset split')
    parser.add_argument('--size', type=int, nargs=2, default=[512, 512], metavar=('W', 'H'), help='Crop width and height (default: 512 512)')
    args = parser.parse_args()
    input_dir, output_dir = resolve_io_directories(args)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(
            f"Step 4 input directory not found: {input_dir}. "
            "Expected --input_root to contain the requested split directory."
        )

    width, height = args.size
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")
    os.makedirs(output_dir, exist_ok=True)
    process_all_images(
        input_dir, args.split, width, height, output_dir,
        args.raw_data_root, device,
    )

if __name__ == '__main__':
    main()
