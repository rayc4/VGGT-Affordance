#!/usr/bin/env python3
"""Convert Step 7 ``lift_results_*.json`` files to ``clip_dir`` layout.

``dataset/preprocess_data_sam2.py`` expects lifted masks as:

    <clip_dir>/<visit_id>/<video_id>/<candidate_id>/mask_result.json

The current Step 7 runner writes one JSON list per visit instead:

    <lift_dir>/lift_results_<visit_id>.json

This script bridges those formats while preserving multiple lifted-mask
candidates for the same ``visit_id/video_id/desc_id``.
"""

import argparse
import csv
import json
import os
import re
from collections import Counter
from pathlib import Path


REQUIRED_KEYS = ("desc_id", "original_indices")


def load_split_visit_ids(data_root, split):
    split_path = Path(data_root) / "benchmark_file_lists" / f"{split}_set.csv"
    if not split_path.is_file():
        raise FileNotFoundError(f"Benchmark split file not found: {split_path}")
    with split_path.open(newline="") as f:
        return {str(row["visit_id"]) for row in csv.DictReader(f)}


def safe_component(value, fallback):
    text = str(value) if value not in (None, "") else fallback
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("._-")
    return text or fallback


def discover_input_files(lift_path, use_merged):
    lift_path = Path(lift_path)
    if lift_path.is_file():
        return [lift_path]

    if not lift_path.is_dir():
        raise FileNotFoundError(f"Input path not found: {lift_path}")

    merged = lift_path / "lift_results_all.json"
    if use_merged:
        if not merged.exists():
            raise FileNotFoundError(f"--use_merged requested, but not found: {merged}")
        return [merged]

    files = sorted(
        p for p in lift_path.glob("lift_results_*.json")
        if re.fullmatch(r"lift_results_\d+\.json", p.name)
    )
    if files:
        return files
    if merged.exists():
        return [merged]
    raise FileNotFoundError(f"No lift_results_<visit>.json files found under {lift_path}")


def load_results(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list in {json_path}")
    return data


def infer_visit_id(json_path, item):
    if item.get("visit_id") not in (None, ""):
        return str(item["visit_id"])

    match = re.fullmatch(r"lift_results_(\d+)\.json", json_path.name)
    if match:
        return match.group(1)
    return ""


def make_candidate_id(item):
    desc_id = safe_component(item.get("desc_id"), "desc")
    frame_id = safe_component(item.get("frame_id"), "noframe")
    mask_idx = safe_component(item.get("mask_idx"), "nomask")
    return f"{desc_id}__frame_{frame_id}__mask_{mask_idx}"


def convert(lift_path, output_dir, use_merged=False, overwrite=False, dry_run=False,
            allowed_visit_ids=None, split=None):
    input_files = discover_input_files(lift_path, use_merged)
    output_dir = Path(output_dir)

    seen_paths = Counter()
    total_items = 0
    written = 0
    skipped = 0
    rejected_wrong_split = 0
    collisions = 0
    manifest = []

    for json_path in input_files:
        results = load_results(json_path)
        for item_index, item in enumerate(results):
            total_items += 1
            missing = [key for key in REQUIRED_KEYS if key not in item]
            visit_id = infer_visit_id(json_path, item)
            video_id = item.get("video_id") or item.get("scan_id")
            if allowed_visit_ids is not None and visit_id not in allowed_visit_ids:
                rejected_wrong_split += 1
                continue
            if missing or not visit_id or not video_id:
                skipped += 1
                print(
                    "Skip item "
                    f"{json_path}:{item_index}; missing "
                    f"{missing or []}, visit_id={visit_id!r}, video_id={video_id!r}"
                )
                continue

            visit_component = safe_component(visit_id, "visit")
            video_component = safe_component(video_id, "video")
            candidate_id = make_candidate_id(item)
            base_rel = Path(visit_component) / video_component / candidate_id

            seen_paths[base_rel] += 1
            rel_path = base_rel
            if seen_paths[base_rel] > 1:
                collisions += 1
                rel_path = Path(str(base_rel) + f"__dup_{seen_paths[base_rel]:03d}")

            out_dir = output_dir / rel_path
            out_json = out_dir / "mask_result.json"
            if out_json.exists() and not overwrite:
                skipped += 1
                print(f"Skip existing {out_json} (use --overwrite to replace)")
                continue

            out_item = dict(item)
            out_item["visit_id"] = visit_id
            out_item["video_id"] = str(video_id)
            out_item["candidate_id"] = rel_path.name
            out_item["source_lift_file"] = str(json_path)
            out_item["source_lift_index"] = item_index

            if not dry_run:
                out_dir.mkdir(parents=True, exist_ok=True)
                with open(out_json, "w") as f:
                    json.dump(out_item, f, indent=2, ensure_ascii=False)

            written += 1
            manifest.append(
                {
                    "mask_result": str(out_json),
                    "source_lift_file": str(json_path),
                    "source_lift_index": item_index,
                    "visit_id": visit_id,
                    "video_id": str(video_id),
                    "desc_id": item.get("desc_id"),
                    "candidate_id": rel_path.name,
                }
            )

    manifest_path = output_dir / "conversion_manifest.json"
    if not dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(manifest_path, "w") as f:
            json.dump(
                {
                    "split": split,
                    "input_files": [str(p) for p in input_files],
                    "total_items": total_items,
                    "written": written,
                    "skipped": skipped,
                    "name_collisions": collisions,
                    "rejected_wrong_split": rejected_wrong_split,
                    "items": manifest,
                },
                f,
                indent=2,
                ensure_ascii=False,
            )

    print(f"Input files: {len(input_files)}")
    print(f"Total items: {total_items}")
    print(f"Written: {written}")
    print(f"Skipped: {skipped}")
    print(f"Rejected outside split: {rejected_wrong_split}")
    print(f"Name collisions disambiguated: {collisions}")
    if dry_run:
        print(f"Dry run only; nothing written to {output_dir}")
    else:
        print(f"Wrote clip_dir layout to {output_dir}")
        print(f"Wrote manifest to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Step 7 lift_results JSON lists to preprocess_data_sam2 clip_dir layout."
    )
    parser.add_argument('--split', required=True, choices=['train', 'val'])
    parser.add_argument('--data_root', default='scenefun3d',
                        help='Dataset root containing benchmark_file_lists')
    parser.add_argument(
        "--lift_dir",
        default=None,
        help="Directory with lift_results_<visit>.json files, or a single lift_results JSON file.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Output clip_dir root to pass as --clip_dir to dataset/preprocess_data_sam2.py.",
    )
    parser.add_argument(
        "--use_merged",
        action="store_true",
        help="When --lift_dir is a directory, read lift_results_all.json instead of per-visit files.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing mask_result.json files.",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Validate inputs and print counts without writing files.",
    )
    args = parser.parse_args()
    if args.lift_dir is None:
        args.lift_dir = os.path.join('pipeline/step7_lift_3d/lift_output', args.split)
    if args.output_dir is None:
        args.output_dir = os.path.join(
            'pipeline/step7_lift_3d/lift_output_organized', args.split
        )
    allowed_visit_ids = load_split_visit_ids(args.data_root, args.split)

    convert(
        lift_path=args.lift_dir,
        output_dir=args.output_dir,
        use_merged=args.use_merged,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        allowed_visit_ids=allowed_visit_ids,
        split=args.split,
    )


if __name__ == "__main__":
    main()
