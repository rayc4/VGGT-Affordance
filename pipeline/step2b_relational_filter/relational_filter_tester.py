"""Run the Step 2b Qwen judgment on one image for interactive inspection."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from PIL import Image

from pipeline.step2b_relational_filter.relational_filter import (
    DEFAULT_MODEL,
    QwenRelationalEvaluator,
    build_filter_prompt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Test the Step 2b Qwen filter on one image"
    )
    parser.add_argument("image", help="Path to the image to evaluate")
    parser.add_argument(
        "--description",
        "-d",
        required=True,
        help="Complete task description used by Step 2b",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument(
        "--show-prompt",
        action="store_true",
        help="Print the exact prompt sent with the image",
    )
    parser.add_argument(
        "--output",
        help="Optionally save the result as JSON",
    )
    return parser


def validate_args(parser: argparse.ArgumentParser, args: argparse.Namespace) -> Path:
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        parser.error(f"image does not exist: {image_path}")
    if not args.description.strip():
        parser.error("--description cannot be empty")
    if args.max_new_tokens < 1:
        parser.error("--max-new-tokens must be >= 1")
    for label in ("min_pixels", "max_pixels"):
        value = getattr(args, label)
        if value is not None and value < 1:
            parser.error(f"--{label.replace('_', '-')} must be >= 1")
    if (
        args.min_pixels is not None
        and args.max_pixels is not None
        and args.min_pixels > args.max_pixels
    ):
        parser.error("--min-pixels cannot exceed --max-pixels")
    try:
        with Image.open(image_path) as image:
            image.verify()
    except (OSError, SyntaxError) as exc:
        parser.error(f"cannot read image {image_path}: {exc}")
    return image_path


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    image_path = validate_args(parser, args)
    description = " ".join(args.description.split())
    prompt = build_filter_prompt(description)

    print(f"Image: {image_path}")
    print(f"Description: {description}")
    print(f"Model: {args.model}")
    if args.show_prompt:
        print("\n--- Prompt ---")
        print(prompt)

    evaluator = QwenRelationalEvaluator(
        args.model,
        local_files_only=args.local_files_only,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    decision = evaluator(str(image_path), description)

    print("\n--- Raw Qwen response ---")
    print(decision.raw_response or "<empty>")
    print("\n--- Parsed decision ---")
    print(f"keep: {decision.keep}")
    print(f"reason: {decision.reason}")

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "image": str(image_path),
            "description": description,
            "model": args.model,
            "prompt": prompt,
            "decision": asdict(decision),
        }
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(result, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        print(f"\nSaved: {output_path}")


if __name__ == "__main__":
    main()
