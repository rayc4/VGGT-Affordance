"""Render an eval results.json (Segment3DEvaluator.save) as a readable report.

Usage:
    python3 scripts/format_results.py                          # newest run under outputs/
    python3 scripts/format_results.py path/to/results.json
    python3 scripts/format_results.py path/to/checkpoint.pt
    python3 scripts/format_results.py <run> --worst 10 --best 10  # best mAP in <run>/eval
    python3 scripts/format_results.py <run> --by-visit
    python3 scripts/format_results.py <run> --csv table.csv
"""

import argparse
import csv
import glob
import json
import os
import re
import statistics
import sys

METRICS = ['mAP', 'AP50', 'AP25', 'mAR', 'AR50', 'AR25', 'mIoU', 'Prc', 'Rec']
LATEX_METRICS = ['mAP', 'AP50', 'AP25', 'mAR', 'AR50', 'AR25', 'mIoU']
ID_KEYS = ('visit_id', 'annot_id')
SELECTION_METRIC = 'mAP'


def find_latest_results():
    """Newest results.json under outputs/, by mtime."""
    hits = glob.glob(
        os.path.join('outputs', '*', 'eval', '**', 'results.json'),
        recursive=True,
    )
    if not hits:
        return None
    return max(hits, key=os.path.getmtime)


def result_score(path, metric=SELECTION_METRIC):
    """Mean checkpoint score, or None when the result cannot be ranked."""
    try:
        with open(path) as f:
            data = json.load(f)
        values = data.get(metric)
        if not isinstance(values, list) or not values:
            return None
        numeric = [
            float(value)
            for value in values
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        ]
        return statistics.fmean(numeric) if numeric else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def find_best_results(root):
    """Highest-mean-mAP results.json recursively below root."""
    hits = glob.glob(os.path.join(root, '**', 'results.json'), recursive=True)
    ranked = [
        (score, os.path.getmtime(path), path)
        for path in hits
        if (score := result_score(path)) is not None
    ]
    return max(ranked)[2] if ranked else None


def checkpoint_root(path):
    """Return the nearest ``ckpt`` ancestor of a checkpoint path."""
    parent = os.path.dirname(os.path.abspath(path))
    while True:
        if os.path.basename(parent) == 'ckpt':
            return parent
        ancestor = os.path.dirname(parent)
        if ancestor == parent:
            return None
        parent = ancestor


def checkpoint_label(path, root):
    """Match the output label used by eval_all_checkpoints.py."""
    relative = os.path.splitext(os.path.relpath(path, root))[0]
    return '__'.join(relative.split(os.sep))


def find_metadata_results(eval_root, checkpoint=None, checkpoint_step=None):
    """Find ranked results whose metadata identifies a checkpoint or step."""
    matches = []
    checkpoint = os.path.realpath(checkpoint) if checkpoint else None
    for metadata_path in glob.glob(
            os.path.join(eval_root, '**', 'metadata.json'), recursive=True):
        try:
            with open(metadata_path) as f:
                metadata = json.load(f)
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            continue

        if checkpoint is not None:
            recorded = metadata.get('checkpoint')
            matched = (isinstance(recorded, str)
                       and os.path.realpath(os.path.expanduser(recorded)) == checkpoint)
        else:
            matched = metadata.get('checkpoint_step') == checkpoint_step
        results_path = os.path.join(os.path.dirname(metadata_path), 'results.json')
        if matched and os.path.isfile(results_path):
            matches.append(results_path)

    ranked = [
        (score, os.path.getmtime(path), path)
        for path in matches
        if (score := result_score(path)) is not None
    ]
    return max(ranked)[2] if ranked else None


def best_checkpoint_step(path):
    """Read the numbered step represented by a ``*_best.pt`` checkpoint."""
    if not os.path.splitext(path)[0].endswith('_best'):
        return None
    info_path = os.path.join(os.path.dirname(path), 'best_checkpoint.json')
    try:
        with open(info_path) as f:
            step = json.load(f).get('step')
        return step if isinstance(step, int) and not isinstance(step, bool) else None
    except (OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def find_checkpoint_results(path):
    """Resolve a checkpoint to the results produced for that checkpoint."""
    root = checkpoint_root(path)
    if root is None:
        return None

    path = os.path.abspath(path)
    eval_root = os.path.join(os.path.dirname(root), 'eval')
    if not os.path.isdir(eval_root):
        return None

    # eval_all_checkpoints.py writes each checkpoint below eval/<label>/.
    result = find_best_results(
        os.path.join(eval_root, checkpoint_label(path, root))
    )
    if result is not None:
        return result

    # Also support evaluations written directly below eval/ or moved within it.
    result = find_metadata_results(eval_root, checkpoint=path)
    if result is not None:
        return result

    # The best checkpoint is a copy of a numbered checkpoint. Batch evaluation
    # commonly skips *_best.pt, so use best_checkpoint.json to find that result.
    step = best_checkpoint_step(path)
    if step is None:
        return None
    stem = os.path.splitext(os.path.basename(path))[0][:-len('_best')]
    numbered = re.compile(rf'{re.escape(stem)}0*{step}$')
    for candidate in glob.glob(os.path.join(root, '**', '*.pt'), recursive=True):
        candidate_stem = os.path.splitext(os.path.basename(candidate))[0]
        if numbered.fullmatch(candidate_stem):
            result = find_best_results(
                os.path.join(eval_root, checkpoint_label(candidate, root))
            )
            if result is not None:
                return result
    return find_metadata_results(eval_root, checkpoint_step=step)


def resolve(path):
    """Accept results, checkpoint, test, run, or eval paths."""
    if path is None:
        return find_latest_results()
    if os.path.isfile(path):
        if os.path.splitext(path)[1].lower() == '.pt':
            return find_checkpoint_results(path)
        return path
    direct_result = os.path.join(path, 'results.json')
    if os.path.isfile(direct_result):
        return direct_result
    eval_dir = os.path.join(path, 'eval')
    search_root = eval_dir if os.path.isdir(eval_dir) else path
    return find_best_results(search_root) if os.path.isdir(search_root) else None


def load(path):
    with open(path) as f:
        data = json.load(f)
    lengths = {len(v) for v in data.values()}
    if len(lengths) != 1:
        sys.exit(f'error: columns have mismatched lengths: '
                 + ', '.join(f'{k}={len(v)}' for k, v in data.items()))
    return data


def rows_of(data):
    """Column-oriented dict -> list of per-sample dicts."""
    keys = list(data)
    return [dict(zip(keys, vals)) for vals in zip(*(data[k] for k in keys))]


def table(headers, rows, aligns=None):
    """Fixed-width text table. aligns: '<' or '>' per column."""
    aligns = aligns or ['<'] * len(headers)
    cols = [[str(h)] + [str(r[i]) for r in rows] for i, h in enumerate(headers)]
    widths = [max(len(c) for c in col) for col in cols]
    out = ['  '.join(f'{h:{a}{w}}' for h, w, a in zip(headers, widths, aligns)).rstrip()]
    out.append('  '.join('-' * w for w in widths))
    for r in rows:
        out.append('  '.join(f'{str(c):{a}{w}}'
                             for c, w, a in zip(r, widths, aligns)).rstrip())
    return '\n'.join(out)


def mean(values):
    return sum(values) / len(values) if values else float('nan')


def pct(value):
    return 'n/a' if value != value else f'{value * 100:.2f}'


def section(title):
    return f'\n{title}\n{"=" * len(title)}'


def summarize(rows, path):
    print(section(f'Eval results  —  {path}'))
    visits = {r['visit_id'] for r in rows}
    scored = [r for r in rows if r['gt_count'] > 0]
    print(f'{len(rows)} samples across {len(visits)} visits '
          f'({len(scored)} with a non-empty ground-truth mask)')

    # Metrics over all samples vs. only those that have a GT mask to match.
    print(section('Metrics (%)'))
    print(table(
        ['Metric', 'All samples', f'GT non-empty (n={len(scored)})'],
        [[m, pct(mean([r[m] for r in rows])), pct(mean([r[m] for r in scored]))]
         for m in METRICS],
        aligns=['<', '>', '>'],
    ))

    print(section('Mask sizes (points per sample)'))
    size_rows = []
    for key in ('pred_count', 'gt_count'):
        vals = [r[key] for r in rows]
        empty = sum(1 for v in vals if v == 0)
        size_rows.append([
            key, f'{mean(vals):.1f}', f'{statistics.median(vals):.0f}',
            f'{min(vals)}', f'{max(vals)}', f'{empty} ({empty / len(vals) * 100:.1f}%)',
        ])
    print(table(['Column', 'Mean', 'Median', 'Min', 'Max', 'Empty'],
                size_rows, aligns=['<', '>', '>', '>', '>', '>']))

    both_empty = sum(1 for r in rows if r['gt_count'] == 0 and r['pred_count'] == 0)
    no_pred = sum(1 for r in scored if r['pred_count'] == 0)
    notes = []
    if both_empty:
        notes.append(f'{both_empty} samples ({both_empty / len(rows) * 100:.1f}%) have '
                     f'both an empty prediction and an empty GT mask; every metric scores '
                     f'0 for them, which pulls the "All samples" column down.')
    if no_pred:
        notes.append(f'{no_pred} of {len(scored)} samples with a real GT mask got an '
                     f'empty prediction ({no_pred / len(scored) * 100:.1f}% misses).')
    if notes:
        print(section('Notes'))
        for note in notes:
            print(f'- {note}')

    print(section('LaTeX row'))
    means = {m: mean([r[m] for r in rows]) for m in LATEX_METRICS}
    print(f'{path} & ' + ' & '.join(pct(means[m]) for m in LATEX_METRICS) + r' \\')


def extremes(rows, n, best):
    if not n:
        return
    scored = [r for r in rows if r['gt_count'] > 0]
    if not scored:
        print(section('Per-sample'))
        print('No samples with a non-empty GT mask to rank.')
        return
    ranked = sorted(scored, key=lambda r: r['mIoU'], reverse=best)[:n]
    print(section(f'{"Best" if best else "Worst"} {len(ranked)} by mIoU '
                  f'(GT non-empty only)'))
    print(table(
        ['visit_id', 'annot_id', 'mIoU', 'Prc', 'Rec', 'pred', 'gt'],
        [[r['visit_id'], r['annot_id'], pct(r['mIoU']), pct(r['Prc']),
          pct(r['Rec']), r['pred_count'], r['gt_count']] for r in ranked],
        aligns=['<', '<', '>', '>', '>', '>', '>'],
    ))


def by_visit(rows, limit):
    groups = {}
    for r in rows:
        groups.setdefault(r['visit_id'], []).append(r)
    ranked = sorted(groups.items(), key=lambda kv: mean([r['mIoU'] for r in kv[1]]),
                    reverse=True)
    shown = ranked[:limit] if limit else ranked
    print(section(f'Per visit ({len(shown)} of {len(ranked)} visits, best mIoU first)'))
    print(table(
        ['visit_id', 'n', 'mIoU', 'AP50', 'AR50', 'GT non-empty'],
        [[visit, len(rs), pct(mean([r['mIoU'] for r in rs])),
          pct(mean([r['AP50'] for r in rs])), pct(mean([r['AR50'] for r in rs])),
          sum(1 for r in rs if r['gt_count'] > 0)] for visit, rs in shown],
        aligns=['<', '>', '>', '>', '>', '>'],
    ))


def write_csv(rows, path):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f'\nWrote {len(rows)} rows to {path}')


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='?', help='results.json, a checkpoint .pt, a test-* '
                                                'dir, or an exp dir (exp dirs use the best '
                                                'mean mAP under DIR/eval; default: newest '
                                                'under outputs/)')
    parser.add_argument('--worst', type=int, default=0, metavar='N',
                        help='list the N lowest-mIoU samples')
    parser.add_argument('--best', type=int, default=0, metavar='N',
                        help='list the N highest-mIoU samples')
    parser.add_argument('--by-visit', action='store_true', help='aggregate per visit_id')
    parser.add_argument('--limit', type=int, default=20, metavar='N',
                        help='max visits shown by --by-visit (0 = all)')
    parser.add_argument('--csv', metavar='PATH', help='dump the per-sample table as CSV')
    args = parser.parse_args()

    path = resolve(args.path)
    if path is None:
        location = (f'for {args.path}' if args.path and os.path.isfile(args.path)
                    else f'under {args.path}' if args.path else 'under outputs/')
        sys.exit(f'error: no results.json found {location}')

    rows = rows_of(load(path))
    if not rows:
        sys.exit(f'error: {path} has no samples')

    summarize(rows, path)
    extremes(rows, args.worst, best=False)
    extremes(rows, args.best, best=True)
    if args.by_visit:
        by_visit(rows, args.limit)
    if args.csv:
        write_csv(rows, args.csv)
    print()


if __name__ == '__main__':
    main()
