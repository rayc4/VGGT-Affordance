#!/usr/bin/env python3
"""Evaluate one mask-refinement checkpoint over many probability thresholds.

Inference is performed once. All thresholds are evaluated in parallel for each
batch, producing compact aggregate metrics for the full validation split and
for the non-empty-ground-truth subset.
"""

import argparse
import csv
from decimal import Decimal
import hashlib
import json
import os
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from hydra import compose, initialize_config_dir
from torch.utils.data import DataLoader

from dataset.AffordanceDataset import AffordanceDataset
from dataset.AffordanceDatasetVGGT import AffordanceDatasetVGGT
from dataset.misc import collate_fn_general
from models.base import create_model, create_model_and_diffusion
import models.cdm_vggt  # noqa: F401 -- register VGGT model classes
from utils.metrics import MAP_METRIC_VERSION, compute_average_precision


METRICS = (
    'Prc', 'mAP', 'AP25', 'AP50',
    'Rec', 'mAR', 'AR25', 'AR50', 'mIoU',
)


def parse_thresholds(spec):
    """Parse comma-separated values or inclusive START:END:STEP notation."""
    if ':' not in spec:
        values = [Decimal(value.strip()) for value in spec.split(',') if value.strip()]
    else:
        parts = [Decimal(value.strip()) for value in spec.split(':')]
        if len(parts) != 3:
            raise ValueError('threshold range must be START:END:STEP')
        start, end, step = parts
        if step <= 0 or end < start:
            raise ValueError('threshold range requires END >= START and STEP > 0')
        values = []
        value = start
        while value <= end:
            values.append(value)
            value += step
    if not values:
        raise ValueError('at least one threshold is required')
    floats = [float(value) for value in values]
    if any(not 0.0 <= value <= 1.0 for value in floats):
        raise ValueError('thresholds must lie in [0, 1]')
    if len(set(floats)) != len(floats):
        raise ValueError('thresholds must be unique')
    return floats


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kind', choices=('base', 'vggt'), required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--label', required=True)
    parser.add_argument('--thresholds', default='0.20:0.80:0.025')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-workers', type=int, default=2)
    parser.add_argument('--expected-frames', type=int, default=5619)
    parser.add_argument('--processed-sam2-dir', type=Path,
                        default=Path('scenefun3d/processed_sam2'))
    parser.add_argument('--vggt-feature-root', type=Path)
    parser.add_argument('--feature-name', default='vggt_feat_uniform.npy')
    parser.add_argument('--confidence-name', default='vggt_conf.npy')
    parser.add_argument('--view-count-name', default='vggt_view_count.npy')
    parser.add_argument('--model-config', default='cdm_vggt_adapter_frozen',
                        help='Hydra model preset used to reconstruct a VGGT checkpoint')
    parser.add_argument('--force', action='store_true')
    return parser.parse_args()


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compose_config(kind, model_config):
    if kind == 'base':
        config_dir = REPO_ROOT / 'pipeline/step8_3d_training/configs'
        overrides = ['model=cdm', 'task=contact_gen']
    else:
        config_dir = REPO_ROOT / 'pipeline/step8_3d_training_vggt/configs'
        overrides = [f'model={model_config}', 'task=mask_refinement_vggt']
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        return compose(config_name='default', overrides=overrides)


def create_dataset(args):
    processed_dir = args.processed_sam2_dir.expanduser().resolve()
    if args.kind == 'base':
        return AffordanceDataset(
            root_dir='',
            processed_sam2_dir=str(processed_dir),
            split='val',
            use_sam2=True,
            require_nonempty_gt=False,
        )

    feature_root = (args.vggt_feature_root.expanduser().resolve()
                    if args.vggt_feature_root
                    else processed_dir / 'vggt_features')
    return AffordanceDatasetVGGT(
        root_dir='',
        processed_sam2_dir=str(processed_dir),
        split='val',
        use_sam2=True,
        require_nonempty_gt=False,
        vggt_feat_root=str(feature_root),
        vggt_feat_name=args.feature_name,
        vggt_conf_name=args.confidence_name,
        vggt_view_count_name=args.view_count_name,
        load_vggt=True,
        load_reliability=True,
    )


def create_checkpoint_model(args, cfg, device):
    if args.kind == 'base':
        model, _ = create_model_and_diffusion(cfg, device=str(device))
    else:
        model = create_model(cfg, device=str(device))
    saved_state = torch.load(args.checkpoint, map_location='cpu')
    if not isinstance(saved_state, dict):
        raise TypeError(f'{args.checkpoint}: checkpoint is not a state dict')
    normalized_state = {}
    for source_key, value in saved_state.items():
        key = source_key[7:] if source_key.startswith('module.') else source_key
        if key in normalized_state:
            raise ValueError(f'{args.checkpoint}: duplicate normalized key {key!r}')
        normalized_state[key] = value
    model_state = model.state_dict()
    missing = sorted(model_state.keys() - normalized_state.keys())
    unexpected = sorted(normalized_state.keys() - model_state.keys())
    mismatched = sorted(
        key for key in model_state.keys() & normalized_state.keys()
        if model_state[key].shape != normalized_state[key].shape
    )
    if missing or unexpected or mismatched:
        raise ValueError(
            f'{args.checkpoint}: incompatible checkpoint for {args.kind}: '
            f'missing={missing[:10]}, unexpected={unexpected[:10]}, '
            f'shape_mismatches={mismatched[:10]}'
        )
    model.load_state_dict(normalized_state, strict=True)
    model.to(device)
    model.eval()
    return model


def batch_metrics(probabilities, ground_truth, thresholds, recall_thresholds):
    predictions = probabilities.unsqueeze(0) > thresholds[:, None, None]
    ground_truth = ground_truth.unsqueeze(0)

    true_positive = torch.logical_and(predictions, ground_truth).sum(-1).float()
    pred_positive = predictions.sum(-1).float()
    gt_positive = ground_truth.sum(-1).float().expand_as(pred_positive)
    union = torch.logical_or(predictions, ground_truth).sum(-1).float()

    precision = torch.where(
        pred_positive > 0,
        true_positive / pred_positive.clamp_min(1),
        torch.zeros_like(true_positive),
    )
    recall = torch.where(
        gt_positive > 0,
        true_positive / gt_positive.clamp_min(1),
        torch.zeros_like(true_positive),
    )
    iou = torch.where(
        union > 0,
        true_positive / union.clamp_min(1),
        torch.zeros_like(true_positive),
    )
    average_precision = compute_average_precision(
        ground_truth.squeeze(0), probabilities
    ).unsqueeze(0).expand_as(precision)
    mean_ar = (recall.unsqueeze(-1) >= recall_thresholds).float().mean(-1)

    values = torch.stack((
        precision,
        average_precision,
        (precision >= 0.25).float(),
        (precision >= 0.50).float(),
        recall,
        mean_ar,
        (recall >= 0.25).float(),
        (recall >= 0.50).float(),
        iou,
    ), dim=-1).double()
    return values, pred_positive


def select_best(rows, scope, metric):
    # Prefer the threshold closest to 0.5 when a discrete metric ties.
    return max(
        rows,
        key=lambda row: (row[scope][metric], -abs(row['threshold'] - 0.5)),
    )


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as handle:
        json.dump(value, handle, indent=2)
        handle.write('\n')
    os.replace(temporary, path)


def write_csv(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    fields = ['threshold']
    for scope in ('all_samples', 'gt_nonempty'):
        fields.extend(f'{scope}.{metric}' for metric in METRICS)
    fields.extend(('pred_empty', 'pred_empty_rate'))
    with temporary.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {'threshold': row['threshold']}
            for scope in ('all_samples', 'gt_nonempty'):
                for metric in METRICS:
                    flat[f'{scope}.{metric}'] = row[scope][metric]
            flat['pred_empty'] = row['pred_empty']
            flat['pred_empty_rate'] = row['pred_empty_rate']
            writer.writerow(flat)
    os.replace(temporary, path)


def main():
    args = parse_args()
    args.checkpoint = args.checkpoint.expanduser().resolve()
    args.output_dir = args.output_dir.expanduser().resolve()
    result_path = args.output_dir / 'threshold_sweep.json'
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f'checkpoint not found: {args.checkpoint}')
    if args.batch_size < 1 or args.num_workers < 0:
        raise ValueError('batch size must be positive and workers non-negative')

    threshold_values = parse_thresholds(args.thresholds)
    checkpoint_sha256 = sha256(args.checkpoint)
    feature_root = (args.vggt_feature_root.expanduser().resolve()
                    if args.vggt_feature_root
                    else args.processed_sam2_dir.expanduser().resolve() / 'vggt_features')
    request = {
        'kind': args.kind,
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': checkpoint_sha256,
        'feature_name': args.feature_name if args.kind == 'vggt' else None,
        'model_config': args.model_config if args.kind == 'vggt' else None,
        'processed_sam2_dir': str(args.processed_sam2_dir.expanduser().resolve()),
        'vggt_feature_root': str(feature_root) if args.kind == 'vggt' else None,
        'thresholds': threshold_values,
        'expected_frames': args.expected_frames,
        'metric_versions': {'mAP': MAP_METRIC_VERSION},
    }
    csv_path = args.output_dir / 'threshold_sweep.csv'
    if result_path.is_file() and not args.force:
        with result_path.open() as handle:
            existing = json.load(handle)
        if existing.get('request') == request and csv_path.is_file():
            print(f'Skip completed sweep: {result_path}')
            return 0
        raise ValueError(
            f'{result_path} exists but does not match this invocation or is '
            'incomplete; pass --force or choose a different output directory'
        )
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    cfg = compose_config(args.kind, args.model_config)
    if args.kind == 'vggt':
        cfg.task.dataset.vggt_feat_name = args.feature_name
        cfg.task.dataset.vggt_conf_name = args.confidence_name
        cfg.task.dataset.vggt_view_count_name = args.view_count_name

    dataset = create_dataset(args)
    if args.expected_frames and len(dataset) != args.expected_frames:
        raise ValueError(
            f'validation dataset has {len(dataset)} frames; '
            f'expected {args.expected_frames}'
        )
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        collate_fn=collate_fn_general,
    )
    model = create_checkpoint_model(args, cfg, device)

    thresholds = torch.tensor(threshold_values, device=device, dtype=torch.float32)
    recall_thresholds = torch.linspace(
        0.5, 0.95, 10, device=device, dtype=torch.float32
    )
    sums_all = torch.zeros(
        (len(threshold_values), len(METRICS)), device=device, dtype=torch.float64
    )
    sums_nonempty = torch.zeros_like(sums_all)
    pred_empty = torch.zeros(len(threshold_values), device=device, dtype=torch.int64)
    sample_count = 0
    nonempty_count = 0

    print(
        f'Evaluate {args.label}: {len(dataset)} frames, '
        f'{len(threshold_values)} thresholds on {device}'
    )
    with torch.inference_mode():
        for batch_index, data in enumerate(dataloader, start=1):
            initial_mask = data['pred_mask_local'].to(device).unsqueeze(-1)
            conditions = {
                key: (value.to(device) if torch.is_tensor(value) else value)
                for key, value in data.items()
                if key.startswith('c_')
            }
            ground_truth = data['gt_mask_local'].to(device).reshape(
                initial_mask.shape[0], -1
            ) > 0.5
            probabilities = torch.sigmoid(
                model(initial_mask, **conditions).squeeze(-1)
            )
            values, pred_positive = batch_metrics(
                probabilities, ground_truth, thresholds, recall_thresholds
            )
            sums_all += values.sum(1)
            nonempty = ground_truth.any(-1)
            if nonempty.any():
                sums_nonempty += values[:, nonempty].sum(1)
            pred_empty += (pred_positive == 0).sum(1)
            sample_count += ground_truth.shape[0]
            nonempty_count += int(nonempty.sum().item())
            if batch_index % 25 == 0 or batch_index == len(dataloader):
                print(f'  batch {batch_index}/{len(dataloader)}')

    sums_all = sums_all.cpu()
    sums_nonempty = sums_nonempty.cpu()
    pred_empty = pred_empty.cpu()
    rows = []
    for index, threshold in enumerate(threshold_values):
        rows.append({
            'threshold': threshold,
            'all_samples': {
                metric: float(sums_all[index, metric_index] / sample_count)
                for metric_index, metric in enumerate(METRICS)
            },
            'gt_nonempty': {
                metric: float(sums_nonempty[index, metric_index] / nonempty_count)
                for metric_index, metric in enumerate(METRICS)
            },
            'pred_empty': int(pred_empty[index]),
            'pred_empty_rate': float(pred_empty[index] / sample_count),
        })

    best = {
        scope: {
            metric: {
                'threshold': select_best(rows, scope, metric)['threshold'],
                'value': select_best(rows, scope, metric)[scope][metric],
            }
            for metric in METRICS
        }
        for scope in ('all_samples', 'gt_nonempty')
    }
    result = {
        'request': request,
        'label': args.label,
        'kind': args.kind,
        'checkpoint': str(args.checkpoint),
        'checkpoint_sha256': checkpoint_sha256,
        'feature_name': args.feature_name if args.kind == 'vggt' else None,
        'model_config': args.model_config if args.kind == 'vggt' else None,
        'processed_sam2_dir': str(args.processed_sam2_dir.expanduser().resolve()),
        'sample_count': sample_count,
        'gt_nonempty_count': nonempty_count,
        'metric_note': (
            'mAP is macro per-frame pointwise average precision computed from '
            'continuous probabilities and is independent of the hard-mask '
            'threshold. mAR and AP25/AP50 retain the repository legacy hard-mask '
            'semantics for compatibility.'
        ),
        'rows': rows,
        'best': best,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    # The JSON is the resumability completion marker, so write it last.
    write_csv(csv_path, rows)
    atomic_json(result_path, result)

    print(f'Wrote {result_path}')
    for metric in ('mAP', 'mIoU', 'Prc', 'Rec'):
        item = best['all_samples'][metric]
        print(f"  best {metric}: {item['value']:.8f} at threshold {item['threshold']:.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
