#!/usr/bin/env python3
"""Evaluate every model checkpoint in an experiment directory.

Checkpoints are read from the ``ckpt`` subdirectory of the experiment
directory given as the positional argument.

Examples:
    python3 scripts/eval_all_checkpoints.py outputs/my-run
    python3 scripts/eval_all_checkpoints.py outputs/my-run \\
        --eval-script pipeline/step8_3d_training_vggt/eval_vggt.py \\
        --cuda-visible-devices 1 -- task.evaluator.eval_nbatch=32

Unknown arguments after ``--`` are forwarded to Hydra. Completed checkpoints
are skipped by default, which makes an interrupted batch safe to resume.
"""

import argparse
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_EVAL_SCRIPT = REPO_ROOT / "pipeline/step8_3d_training/eval_no_diff.py"


def natural_key(path: Path):
    """Sort paths so model5000 precedes model10000."""
    return [int(part) if part.isdigit() else part.lower()
            for part in re.split(r"(\d+)", str(path))]


def parse_args():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("experiment_dir", type=Path,
                        help="experiment directory (checkpoints are read from its ckpt subdirectory)")
    parser.add_argument("--pattern", default="mask_refinement_model*.pt",
                        help="checkpoint glob (default: %(default)s)")
    parser.add_argument("--recursive", action="store_true",
                        help="search below the ckpt directory recursively")
    parser.add_argument("--eval-script", type=Path, default=DEFAULT_EVAL_SCRIPT,
                        help="Hydra evaluation entry point (default: %(default)s)")
    parser.add_argument("--output-dir", type=Path,
                        help="batch evaluation root (default: EXP/eval/checkpoints)")
    parser.add_argument("--gpu", default="0",
                        help="logical evaluator GPU, or 'null' for CPU (default: %(default)s)")
    parser.add_argument("--cuda-visible-devices",
                        help="set CUDA_VISIBLE_DEVICES for every evaluation process")
    parser.add_argument("--python", default=sys.executable,
                        help="Python executable used for evaluation (default: current Python)")
    parser.add_argument("--force", action="store_true",
                        help="evaluate checkpoints that already have a results.json")
    parser.add_argument("--keep-going", action="store_true",
                        help="continue after an evaluation fails")
    parser.add_argument("--dry-run", action="store_true",
                        help="print commands without running evaluations")
    args, hydra_args = parser.parse_known_args()
    if hydra_args[:1] == ["--"]:
        hydra_args = hydra_args[1:]
    return args, hydra_args


def checkpoint_label(checkpoint: Path, checkpoint_dir: Path) -> str:
    """Create a stable output name, including subdirectories when recursive."""
    relative = checkpoint.relative_to(checkpoint_dir).with_suffix("")
    return "__".join(relative.parts)


def main() -> int:
    args, hydra_args = parse_args()
    experiment_dir = args.experiment_dir.expanduser().resolve()
    if not experiment_dir.is_dir():
        print(f"error: experiment directory does not exist: {experiment_dir}",
              file=sys.stderr)
        return 2
    checkpoint_dir = experiment_dir / "ckpt"
    if not checkpoint_dir.is_dir():
        print(f"error: checkpoint directory does not exist: {checkpoint_dir}",
              file=sys.stderr)
        return 2

    eval_script = args.eval_script.expanduser()
    if not eval_script.is_absolute():
        eval_script = (Path.cwd() / eval_script).resolve()
    if not eval_script.is_file():
        print(f"error: evaluation script does not exist: {eval_script}",
              file=sys.stderr)
        return 2

    output_dir = (args.output_dir.expanduser().resolve()
                  if args.output_dir else experiment_dir / "eval/checkpoints")

    iterator = (checkpoint_dir.rglob(args.pattern) if args.recursive
                else checkpoint_dir.glob(args.pattern))
    checkpoints = sorted((path.resolve() for path in iterator if path.is_file()),
                         key=natural_key)
    if not checkpoints:
        print(f"error: no checkpoints matching {args.pattern!r} under {checkpoint_dir}",
              file=sys.stderr)
        return 2

    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH")
    env["PYTHONPATH"] = (str(REPO_ROOT) if not existing_pythonpath else
                         os.pathsep.join((str(REPO_ROOT), existing_pythonpath)))
    if args.cuda_visible_devices is not None:
        env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices

    completed = skipped = failed = 0
    print(f"Found {len(checkpoints)} checkpoints in {checkpoint_dir}")
    print(f"Evaluation output: {output_dir}")

    for index, checkpoint in enumerate(checkpoints, start=1):
        label = checkpoint_label(checkpoint, checkpoint_dir)
        checkpoint_output = output_dir / label
        existing_results = list(checkpoint_output.rglob("results.json"))
        if existing_results and not args.force:
            skipped += 1
            print(f"[{index}/{len(checkpoints)}] skip {checkpoint.name} "
                  f"({len(existing_results)} result file(s) found)")
            continue

        if not args.dry_run:
            checkpoint_output.mkdir(parents=True, exist_ok=True)
        command = [
            args.python,
            str(eval_script),
            "hydra/job_logging=none",
            "hydra/hydra_logging=none",
            f"exp_dir={experiment_dir}",
            f"eval_dir={checkpoint_output}",
            f"checkpoint={checkpoint}",
            f"gpu={args.gpu}",
            *hydra_args,
        ]
        print(f"[{index}/{len(checkpoints)}] evaluate {checkpoint.name}")
        print("  " + shlex.join(command))
        if args.dry_run:
            continue

        try:
            subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
            completed += 1
        except subprocess.CalledProcessError as exc:
            failed += 1
            print(f"error: {checkpoint.name} failed with exit code {exc.returncode}",
                  file=sys.stderr)
            if not args.keep_going:
                break
        except KeyboardInterrupt:
            print("\nInterrupted; rerun the same command to resume.", file=sys.stderr)
            return 130

    print(f"Done: {completed} evaluated, {skipped} skipped, {failed} failed.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
