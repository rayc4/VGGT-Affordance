#!/usr/bin/env python3
"""Combine threshold_sweep.json files into a compact comparison table."""

import argparse
import csv
import glob
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    return parser.parse_args()


def row_at(rows, threshold):
    return min(rows, key=lambda row: abs(row['threshold'] - threshold))


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    paths = sorted(glob.glob(str(root / '*' / 'threshold_sweep.json')))
    if not paths:
        raise SystemExit(f'error: no completed threshold sweeps under {root}')

    summary = []
    for path in paths:
        with open(path) as handle:
            result = json.load(handle)
        baseline = row_at(result['rows'], 0.5)['all_samples']
        best_map = result['best']['all_samples']['mAP']
        best_miou = result['best']['all_samples']['mIoU']
        summary.append({
            'label': result['label'],
            'kind': result['kind'],
            'feature_name': result.get('feature_name') or '',
            'checkpoint': result['checkpoint'],
            'mAP_at_0.5': baseline['mAP'],
            'best_mAP': best_map['value'],
            'best_mAP_threshold': best_map['threshold'],
            'mAP_gain': best_map['value'] - baseline['mAP'],
            'mIoU_at_0.5': baseline['mIoU'],
            'best_mIoU': best_miou['value'],
            'best_mIoU_threshold': best_miou['threshold'],
            'mIoU_gain': best_miou['value'] - baseline['mIoU'],
        })

    fields = list(summary[0])
    csv_path = root / 'summary.csv'
    with csv_path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    with (root / 'summary.json').open('w') as handle:
        json.dump(summary, handle, indent=2)
        handle.write('\n')

    headers = ('label', 'mAP@.5', 'best mAP', 'thr', 'gain',
               'mIoU@.5', 'best mIoU', 'thr', 'gain')
    rendered = []
    for row in summary:
        rendered.append((
            row['label'],
            f"{row['mAP_at_0.5']:.5f}",
            f"{row['best_mAP']:.5f}",
            f"{row['best_mAP_threshold']:.3f}",
            f"{row['mAP_gain']:+.5f}",
            f"{row['mIoU_at_0.5']:.5f}",
            f"{row['best_mIoU']:.5f}",
            f"{row['best_mIoU_threshold']:.3f}",
            f"{row['mIoU_gain']:+.5f}",
        ))
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rendered))
        for index in range(len(headers))
    ]
    print('  '.join(value.ljust(widths[index])
                    for index, value in enumerate(headers)))
    print('  '.join('-' * width for width in widths))
    for row in rendered:
        print('  '.join(value.ljust(widths[index])
                        for index, value in enumerate(row)))
    print(f'\nWrote {csv_path}')


if __name__ == '__main__':
    main()
