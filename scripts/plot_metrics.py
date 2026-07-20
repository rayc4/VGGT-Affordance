#!/usr/bin/env python3
"""Collect training/evaluation metrics and plot checkpoint curves.

Training scalars are read from TensorBoard event files. If the ``tensorboard``
package is unavailable, the script falls back to mask-refinement entries in
``runtime.log``. Evaluation means are read from every ``results.json`` below
each experiment directory.

Examples:
    python3 scripts/plot_metrics.py outputs/my-run
    python3 scripts/plot_metrics.py outputs/run-a outputs/run-b --smooth 10
    python3 scripts/plot_metrics.py outputs/my-run --output-dir reports/my-run
"""

import argparse
import csv
import json
import math
from pathlib import Path
import re
import statistics
import sys


EVAL_METRICS = ("mAP", "AP50", "AP25", "mAR", "AR50", "AR25",
                "mIoU", "Prc", "Rec")
CHECKPOINT_RE = re.compile(r"mask_refinement_model(\d+)(?:\.pt)?")
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LOG_FIELD_RE = re.compile(rf"([A-Za-z][A-Za-z ]*):\s*({NUMBER_RE})")
LOG_FIELD_NAMES = {
    "Epoch": "train/epoch",
    "Loss": "train/total_loss",
    "BCE": "train/mask_loss",
    "Dice": "train/dice_loss",
    "Focal": "train/focal_loss",
    "IoU": "train/iou_loss",
    "PosRatio": "train/positive_ratio",
    "PredMean": "train/pred_mean",
    "GtMean": "train/gt_mean",
}


def warn(message):
    print(f"warning: {message}", file=sys.stderr)


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("experiments", nargs="+", type=Path,
                        help="one or more experiment directories")
    parser.add_argument("--output-dir", type=Path,
                        help="destination (default: EXP/metrics for one input)")
    parser.add_argument("--training-metrics", metavar="TAG,...",
                        help="TensorBoard tags to plot (default: all except epoch)")
    parser.add_argument("--evaluation-metrics", metavar="NAME,...",
                        help="evaluation metrics to plot (default: all)")
    parser.add_argument("--smooth", type=int, default=1, metavar="N",
                        help="training moving-average window (default: %(default)s)")
    parser.add_argument("--no-plots", action="store_true",
                        help="write CSV summaries without importing matplotlib")
    args = parser.parse_args()
    if args.smooth < 1:
        parser.error("--smooth must be at least 1")
    return args


def experiment_label(root, labels_seen):
    label = root.name or str(root)
    if label in labels_seen:
        label = str(root)
    labels_seen.add(label)
    return label


def tensorboard_training_rows(root, experiment):
    event_files = sorted(root.rglob("events.out.tfevents.*"))
    if not event_files:
        return [], None
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        return [], "the tensorboard package is unavailable"

    latest = {}
    for event_file in event_files:
        try:
            accumulator = EventAccumulator(str(event_file),
                                           size_guidance={"scalars": 0})
            accumulator.Reload()
        except Exception as exc:  # a partial event file should not hide the rest
            warn(f"could not read {event_file}: {exc}")
            continue
        for metric in accumulator.Tags().get("scalars", []):
            for event in accumulator.Scalars(metric):
                row = {
                    "experiment": experiment,
                    "step": int(event.step),
                    "metric": metric,
                    "value": float(event.value),
                    "source": str(event_file),
                    "wall_time": float(event.wall_time),
                }
                key = (metric, row["step"])
                if key not in latest or row["wall_time"] > latest[key]["wall_time"]:
                    latest[key] = row
    return list(latest.values()), None


def log_training_rows(root, experiment):
    """Fallback parser for SimpleMaskRefinementTrainLoop log lines."""
    latest = {}
    for log_file in sorted(root.rglob("runtime.log")):
        try:
            lines = log_file.read_text(errors="replace").splitlines()
        except OSError as exc:
            warn(f"could not read {log_file}: {exc}")
            continue
        for line in lines:
            if "[MASK REFINEMENT]" not in line or "Step:" not in line:
                continue
            step_match = re.search(r"\bStep:\s*(\d+)", line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            for display_name, value in LOG_FIELD_RE.findall(line):
                metric = LOG_FIELD_NAMES.get(display_name.strip())
                if metric:
                    latest[(metric, step)] = {
                        "experiment": experiment,
                        "step": step,
                        "metric": metric,
                        "value": float(value),
                        "source": str(log_file),
                        "wall_time": log_file.stat().st_mtime,
                    }
    return list(latest.values())


def gather_training(root, experiment):
    rows, fallback_reason = tensorboard_training_rows(root, experiment)
    if rows:
        return rows
    log_rows = log_training_rows(root, experiment)
    if log_rows and fallback_reason:
        warn(f"{fallback_reason}; parsed training metrics from runtime.log instead")
    elif log_rows:
        warn(f"no TensorBoard scalars found under {root}; parsed runtime.log instead")
    return log_rows


def checkpoint_info(results_path):
    checkpoint = ""
    step = None
    metadata_path = results_path.with_name("metadata.json")
    if metadata_path.is_file():
        try:
            metadata = json.loads(metadata_path.read_text())
            checkpoint = str(metadata.get("checkpoint") or "")
            raw_step = metadata.get("checkpoint_step")
            step = int(raw_step) if raw_step is not None else None
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            warn(f"could not parse {metadata_path}: {exc}")

    if step is None:
        match = CHECKPOINT_RE.search(checkpoint) or CHECKPOINT_RE.search(str(results_path))
        if match:
            step = int(match.group(1))

    log_path = results_path.with_name("test.log")
    if (not checkpoint or step is None) and log_path.is_file():
        try:
            log_text = log_path.read_text(errors="replace")
            matches = list(CHECKPOINT_RE.finditer(log_text))
            if matches:
                match = matches[-1]
                step = int(match.group(1)) if step is None else step
                if not checkpoint:
                    line_start = log_text.rfind("\n", 0, match.start()) + 1
                    line_end = log_text.find("\n", match.end())
                    line_end = len(log_text) if line_end < 0 else line_end
                    line = log_text[line_start:line_end]
                    path_match = re.search(r"Load checkpoint from\s+(.+?\.pt)", line)
                    checkpoint = path_match.group(1).strip() if path_match else match.group(0)
        except OSError as exc:
            warn(f"could not read {log_path}: {exc}")
    return checkpoint, step


def evaluation_row(results_path, experiment):
    try:
        data = json.loads(results_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        warn(f"could not parse {results_path}: {exc}")
        return None
    if not isinstance(data, dict):
        warn(f"expected an object in {results_path}")
        return None

    lengths = [len(values) for values in data.values() if isinstance(values, list)]
    sample_count = max(lengths, default=0)
    checkpoint, step = checkpoint_info(results_path)
    row = {
        "experiment": experiment,
        "checkpoint": checkpoint,
        "step": step,
        "samples": sample_count,
        "gt_nonempty": sum(value > 0 for value in data.get("gt_count", [])),
        "results_path": str(results_path),
        "mtime": results_path.stat().st_mtime,
    }
    for metric in EVAL_METRICS:
        values = data.get(metric, [])
        numeric = [float(value) for value in values
                   if isinstance(value, (int, float)) and not isinstance(value, bool)]
        row[metric] = statistics.fmean(numeric) if numeric else None
    return row


def gather_evaluations(root, experiment):
    rows = []
    for results_path in sorted(root.rglob("results.json")):
        row = evaluation_row(results_path, experiment)
        if row:
            rows.append(row)

    # Keep the newest rerun for each checkpoint so curves do not double back.
    latest = {}
    for row in rows:
        identity = (row["step"] if row["step"] is not None else
                    row["checkpoint"] or row["results_path"])
        if identity not in latest or row["mtime"] > latest[identity]["mtime"]:
            latest[identity] = row
    return list(latest.values())


def write_training_csv(rows, path):
    fields = ("experiment", "step", "metric", "value", "source")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (row["experiment"],
                                                        row["metric"], row["step"])))


def write_evaluation_csv(rows, path):
    fields = ("experiment", "checkpoint", "step", "samples", "gt_nonempty",
              *EVAL_METRICS, "results_path")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: (
            row["experiment"], row["step"] is None,
            row["step"] if row["step"] is not None else row["mtime"])))


def metric_filter(raw, available):
    if not raw:
        return sorted(available)
    requested = [name.strip() for name in raw.split(",") if name.strip()]
    missing = [name for name in requested if name not in available]
    if missing:
        warn("requested metrics were not found: " + ", ".join(missing))
    return [name for name in requested if name in available]


def moving_average(values, window):
    if window <= 1:
        return values
    smoothed = []
    running_sum = 0.0
    for index, value in enumerate(values):
        running_sum += value
        if index >= window:
            running_sum -= values[index - window]
        smoothed.append(running_sum / min(index + 1, window))
    return smoothed


def pyplot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError as exc:
        raise SystemExit(
            "error: plotting requires matplotlib and numpy; install requirements.txt "
            "or rerun with --no-plots (CSV files were still written)"
        ) from exc


def subplot_grid(plt, metric_count, title):
    columns = min(3, metric_count)
    rows = math.ceil(metric_count / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 4 * rows),
                                squeeze=False)
    figure.suptitle(title, fontsize=14)
    return figure, list(axes.flat)


def plot_training(rows, metrics, experiments, output_path, smooth):
    if not rows or not metrics:
        return False
    plt = pyplot()
    figure, axes = subplot_grid(plt, len(metrics), "Training metrics")
    for axis, metric in zip(axes, metrics):
        for experiment in experiments:
            points = sorted((row for row in rows
                             if row["experiment"] == experiment and row["metric"] == metric),
                            key=lambda row: row["step"])
            if points:
                axis.plot([row["step"] for row in points],
                          moving_average([row["value"] for row in points], smooth),
                          label=experiment, linewidth=1.4)
        axis.set_title(metric)
        axis.set_xlabel("Training step")
        axis.grid(alpha=0.25)
        if len(experiments) > 1:
            axis.legend(fontsize="small")
    for axis in axes[len(metrics):]:
        axis.set_visible(False)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def plot_losses(rows, metrics, experiments, output_path, smooth):
    """Plot all loss series together for quick convergence comparison."""
    if not rows or not metrics:
        return False
    plt = pyplot()
    figure, axis = plt.subplots(figsize=(11, 6))
    for experiment in experiments:
        for metric in metrics:
            points = sorted((row for row in rows
                             if row["experiment"] == experiment and row["metric"] == metric),
                            key=lambda row: row["step"])
            if not points:
                continue
            label = metric if len(experiments) == 1 else f"{experiment}: {metric}"
            axis.plot([row["step"] for row in points],
                      moving_average([row["value"] for row in points], smooth),
                      label=label, linewidth=1.4)
    axis.set_title("Training losses")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small", ncol=2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def plot_evaluations(rows, metrics, experiments, output_path):
    if not rows or not metrics:
        return False
    plt = pyplot()
    figure, axes = subplot_grid(plt, len(metrics), "Evaluation metrics")
    for axis, metric in zip(axes, metrics):
        used_ordinal_axis = False
        for experiment in experiments:
            points = [row for row in rows if row["experiment"] == experiment
                      and row.get(metric) is not None]
            points.sort(key=lambda row: (row["step"] is None,
                                         row["step"] if row["step"] is not None else row["mtime"]))
            if not points:
                continue
            if all(row["step"] is not None for row in points):
                x_values = [row["step"] for row in points]
            else:
                x_values = list(range(1, len(points) + 1))
                used_ordinal_axis = True
            axis.plot(x_values, [100.0 * row[metric] for row in points],
                      marker="o", markersize=3, label=experiment, linewidth=1.4)
        axis.set_title(metric)
        axis.set_xlabel("Evaluation index" if used_ordinal_axis else "Checkpoint step")
        axis.set_ylabel("Mean (%)")
        axis.grid(alpha=0.25)
        if len(experiments) > 1:
            axis.legend(fontsize="small")
    for axis in axes[len(metrics):]:
        axis.set_visible(False)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def main():
    args = parse_args()
    roots = [path.expanduser().resolve() for path in args.experiments]
    missing = [path for path in roots if not path.is_dir()]
    if missing:
        print("error: experiment directories do not exist: " +
              ", ".join(str(path) for path in missing), file=sys.stderr)
        return 2

    output_dir = (args.output_dir.expanduser().resolve() if args.output_dir else
                  (roots[0] / "metrics" if len(roots) == 1 else
                   Path.cwd() / "metrics"))
    output_dir.mkdir(parents=True, exist_ok=True)

    labels_seen = set()
    labels = [experiment_label(root, labels_seen) for root in roots]
    training_rows = []
    evaluation_rows = []
    for root, label in zip(roots, labels):
        training_rows.extend(gather_training(root, label))
        evaluation_rows.extend(gather_evaluations(root, label))

    training_csv = output_dir / "training_metrics.csv"
    evaluation_csv = output_dir / "evaluation_metrics.csv"
    write_training_csv(training_rows, training_csv)
    write_evaluation_csv(evaluation_rows, evaluation_csv)
    print(f"Wrote {len(training_rows)} training values to {training_csv}")
    print(f"Wrote {len(evaluation_rows)} evaluation summaries to {evaluation_csv}")

    if not training_rows and not evaluation_rows:
        warn("no training or evaluation metrics were found")
        return 1
    if args.no_plots:
        return 0

    available_training = {row["metric"] for row in training_rows}
    default_training = available_training - {"train/epoch"}
    if "train/total_loss" in default_training:
        default_training.discard("train/loss")  # the training loop logs both identically
    training_metrics = (metric_filter(args.training_metrics, available_training)
                        if args.training_metrics else sorted(default_training))
    available_evaluation = {metric for metric in EVAL_METRICS
                            if any(row.get(metric) is not None for row in evaluation_rows)}
    evaluation_metrics = metric_filter(args.evaluation_metrics, available_evaluation)

    training_plot = output_dir / "training_metrics.png"
    loss_plot = output_dir / "loss_metrics.png"
    evaluation_plot = output_dir / "evaluation_metrics.png"
    if plot_training(training_rows, training_metrics, labels, training_plot, args.smooth):
        print(f"Wrote {training_plot}")
    loss_metrics = sorted(metric for metric in available_training
                          if metric.lower().endswith("loss") and metric != "train/loss")
    if plot_losses(training_rows, loss_metrics, labels, loss_plot, args.smooth):
        print(f"Wrote {loss_plot}")
    if plot_evaluations(evaluation_rows, evaluation_metrics, labels, evaluation_plot):
        print(f"Wrote {evaluation_plot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
