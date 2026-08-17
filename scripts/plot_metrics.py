#!/usr/bin/env python3
"""Plot the main evaluation metrics and training losses in one image.

Training scalars are read from TensorBoard event files. If the ``tensorboard``
package is unavailable, the script falls back to mask-refinement entries in
``runtime.log``. Evaluation means are read from every ``results.json`` below
each experiment directory. The resulting dashboard is written to
``metrics/metrics.png`` by default.

Examples:
    python3 scripts/plot_metrics.py outputs/my-run
    python3 scripts/plot_metrics.py outputs/run-a outputs/run-b --smooth 10
    python3 scripts/plot_metrics.py outputs/my-run --output-dir reports/my-run
"""

import argparse
import json
import math
from pathlib import Path
import re
import statistics
import sys


MAIN_METRICS = ("mAP", "mIoU", "Prc", "Rec")
CHECKPOINT_RE = re.compile(r"mask_refinement_model(\d+)(?:\.pt)?")
NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"
LOG_FIELD_RE = re.compile(rf"([A-Za-z][A-Za-z0-9 ]*):\s*({NUMBER_RE})")
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
VALIDATION_LOG_FIELD_NAMES = {
    "Epoch": "val/epoch",
    "Loss": "val/loss",
    "BCE": "val/mask_loss",
    "Dice": "val/dice_loss",
    "Focal": "val/focal_loss",
    "SoftIoU": "val/soft_iou",
    "Prc": "val/Prc",
    "Rec": "val/Rec",
    "mIoU": "val/mIoU",
    "mAP": "val/mAP",
    "AP25": "val/AP25",
    "AP50": "val/AP50",
    "mAR": "val/mAR",
    "AR25": "val/AR25",
    "AR50": "val/AR50",
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
    parser.add_argument("--smooth", type=int, default=1, metavar="N",
                        help="training moving-average window (default: %(default)s)")
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
            is_validation = "[VALIDATION]" in line
            if not is_validation and "[MASK REFINEMENT]" not in line:
                continue
            if "Step:" not in line:
                continue
            step_match = re.search(r"\bStep:\s*(\d+)", line)
            if not step_match:
                continue
            step = int(step_match.group(1))
            for display_name, value in LOG_FIELD_RE.findall(line):
                field_names = (
                    VALIDATION_LOG_FIELD_NAMES if is_validation else LOG_FIELD_NAMES
                )
                metric = field_names.get(display_name.strip())
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

    checkpoint, step = checkpoint_info(results_path)
    row = {
        "experiment": experiment,
        "checkpoint": checkpoint,
        "step": step,
        "results_path": str(results_path),
        "mtime": results_path.stat().st_mtime,
    }
    for metric in MAIN_METRICS:
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
            "error: plotting requires matplotlib and numpy; install requirements.txt"
        ) from exc


def plot_losses(axis, rows, metrics, experiments, smooth):
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
    axis.set_title("Losses")
    axis.set_xlabel("Training step")
    axis.set_ylabel("Loss")
    axis.grid(alpha=0.25)
    axis.legend(fontsize="small", ncol=2)


def plot_summary(training_rows, evaluation_rows, metrics, losses, experiments,
                 output_path, smooth):
    if not metrics and not losses:
        return False

    plt = pyplot()
    columns = min(2, max(1, len(metrics)))
    metric_rows = math.ceil(len(metrics) / columns)
    total_rows = metric_rows + bool(losses)
    figure = plt.figure(figsize=(7 * columns, 4 * total_rows))
    grid = figure.add_gridspec(total_rows, columns)

    for index, metric in enumerate(metrics):
        axis = figure.add_subplot(grid[index // columns, index % columns])
        plot_evaluation_metric(axis, evaluation_rows, metric, experiments)
    if losses:
        loss_axis = figure.add_subplot(grid[metric_rows, :])
        plot_losses(loss_axis, training_rows, losses, experiments, smooth)

    figure.suptitle("Main metrics and losses", fontsize=14)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    figure.savefig(output_path, dpi=160)
    plt.close(figure)
    return True


def plot_evaluation_metric(axis, rows, metric, experiments):
    used_ordinal_axis = False
    for experiment in experiments:
        points = [row for row in rows if row["experiment"] == experiment
                  and row.get(metric) is not None]
        points.sort(key=lambda row: (
            row["step"] is None,
            row["step"] if row["step"] is not None else row["mtime"],
        ))
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

    if not training_rows and not evaluation_rows:
        warn("no training or evaluation metrics were found")
        return 1

    available_training = {row["metric"] for row in training_rows}
    loss_metrics = sorted(metric for metric in available_training
                          if metric.lower().endswith("loss") and metric != "train/loss")
    evaluation_metrics = [metric for metric in MAIN_METRICS
                          if any(row.get(metric) is not None for row in evaluation_rows)]

    output_path = output_dir / "metrics.png"
    if not plot_summary(training_rows, evaluation_rows, evaluation_metrics,
                        loss_metrics, labels, output_path, args.smooth):
        warn("no main metrics or losses were found")
        return 1
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
