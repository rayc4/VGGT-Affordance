"""Step 2b: directly ask Qwen whether relational candidate frames are usable.

Step 2 emits one ``*_result.json`` file per video. This stage preserves that
schema and only removes frames from relational descriptions when Qwen does not
believe the image visibly satisfies the description. Non-relational
descriptions pass through unchanged.

The model and transformers imports are lazy so result handling and
non-relational inputs can be tested without loading Qwen.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import re
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
DEFAULT_INPUT_ROOT = "pipeline/step2_clipwithaffordance/clipwithaffordance_output"
DEFAULT_OUTPUT_ROOT = "pipeline/step2b_relational_filter/clipwithaffordance_output"
SCHEMA_VERSION = 4


# Detect external visual relations, not part names such as "left drawer" or
# "right closet door". Detection only decides whether Step 2b should run;
# Qwen itself interprets the complete description.
_RELATION_PATTERNS = tuple(
    re.compile(pattern, re.IGNORECASE)
    for pattern in (
        r"\bnext\s+to\b",
        r"\bbeside\b",
        r"\badjacent\s+to\b",
        r"\bbetween\b",
        r"\bamong\b",
        r"\b(?:to|on|at)\s+the\s+(?:direct\s+)?(?:left|right)\s+of\b",
        r"\b(?:directly\s+)?(?:left|right)\s+of\b",
        r"\b(?:directly\s+)?(?:above|below|under)\b",
        r"\bover\b",
        r"\bunderneath\b",
        r"\bbeneath\b",
        r"\bon\s+top\s+of\b",
        r"\batop\b",
        r"(?<!\bturn)(?<!\bswitch)(?<!\bpower)\s+on\s+(?:the|a|an)\b",
        r"\bin\s+front\s+of\b",
        r"\bbehind\b",
        r"\bnear(?:est)?\b",
        r"\bclose(?:st)?\s+to\b",
        r"\bopposite\b",
        r"\bacross\s+from\b",
        r"\balongside\b",
        r"\binside\s+(?:of\s+)?\b",
        r"\bwithin\b",
        r"\b(?:attached|connected|mounted)\s+(?:to|on)\b",
    )
)


class RelationalFilterError(ValueError):
    """An input or model response violates the Step 2b contract."""


@dataclass(frozen=True)
class FilterDecision:
    """Qwen's direct decision for one description/frame pair."""

    keep: bool
    reason: str
    raw_response: str = ""
    schema_version: int = SCHEMA_VERSION


Evaluator = Callable[[str, str], FilterDecision]


def is_relational_description(description: str) -> bool:
    """Return whether an instruction contains an external visual relation."""

    if not isinstance(description, str):
        return False
    normalized = " ".join(description.split())
    return any(pattern.search(normalized) for pattern in _RELATION_PATTERNS)


def build_filter_prompt(description: str) -> str:
    """Build the single direct visual judgment prompt."""

    return f"""Does the scene in the image provide the objects necessary to 
identify objects and perform the following action?
"{description}"

Answer YES if the object or component being physically operated and any
objects used to locate it are recognizable, and their described relationship
looks plausible. A recognizable partial object at the image edge counts.
Do not require a person, the action, its result, a carried device, or an object
controlled by a visible switch or visible remote.

Answer only YES or NO."""


def parse_filter_response(text: str) -> FilterDecision:
    """Parse Qwen's direct YES/NO judgment."""

    if not isinstance(text, str) or not text.strip():
        raise RelationalFilterError("model response is empty")
    match = re.fullmatch(r"\s*(YES|NO)[.!]?\s*", text, re.IGNORECASE)
    if match is None:
        raise RelationalFilterError("model response must be exactly YES or NO")
    answer = match.group(1).upper()
    return FilterDecision(
        keep=answer == "YES",
        reason=f"Qwen answered {answer}",
        raw_response=text,
    )


class QwenRelationalEvaluator:
    """Lazy Qwen2.5-VL direct frame evaluator."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL,
        *,
        local_files_only: bool = False,
        max_new_tokens: int = 8,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
    ) -> None:
        self.model_name = model_name
        self.local_files_only = local_files_only
        self.max_new_tokens = max_new_tokens
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self._model: Any = None
        self._processor: Any = None
        self._process_vision_info: Any = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch
        from qwen_vl_utils import process_vision_info
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        print(f"Loading relational filter model: {self.model_name}")
        self._model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=self.local_files_only,
        )
        self._model.eval()
        processor_kwargs: dict[str, Any] = {
            "use_fast": True,
            "local_files_only": self.local_files_only,
        }
        if self.min_pixels is not None:
            processor_kwargs["min_pixels"] = self.min_pixels
        if self.max_pixels is not None:
            processor_kwargs["max_pixels"] = self.max_pixels
        self._processor = AutoProcessor.from_pretrained(
            self.model_name, **processor_kwargs
        )
        self._process_vision_info = process_vision_info
        print("Relational filter model loaded.")

    def _decode(self, inputs: Any) -> str:
        import torch

        with torch.inference_mode():
            generated = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
            )
        trimmed = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, generated)
        ]
        responses = self._processor.batch_decode(
            trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        return responses[0] if responses else ""

    def __call__(self, image_path: str, description: str) -> FilterDecision:
        self._ensure_loaded()
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": build_filter_prompt(description)},
                ],
            }
        ]
        text = self._processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = self._process_vision_info(messages)
        inputs = self._processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self._model.device)
        response = self._decode(inputs)
        try:
            return parse_filter_response(response)
        except RelationalFilterError as exc:
            return FilterDecision(
                keep=False,
                reason=f"invalid model response: {exc}",
                raw_response=response,
            )


def filter_result_items(
    items: Sequence[Mapping[str, Any]],
    image_dir: str | Path,
    evaluator: Evaluator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter one Step 2 result list and return output plus audit entries."""

    if not isinstance(items, list):
        raise RelationalFilterError("Step 2 result must be a JSON list")
    image_dir = Path(image_dir)
    filtered: list[dict[str, Any]] = []
    audit: list[dict[str, Any]] = []

    for item_index, source_item in enumerate(items):
        if not isinstance(source_item, Mapping):
            raise RelationalFilterError(f"item {item_index} must be an object")
        item = copy.deepcopy(dict(source_item))
        description = item.get("description")
        frame_names = item.get("image_name")
        if not isinstance(description, str) or not description.strip():
            raise RelationalFilterError(
                f"item {item_index} description must be a non-empty string"
            )
        if not isinstance(frame_names, list) or not all(
            isinstance(name, str) and name for name in frame_names
        ):
            raise RelationalFilterError(
                f"item {item_index} image_name must be a list of names"
            )

        relational = is_relational_description(description)
        kept_names: list[str] = []
        decisions: list[dict[str, Any]] = []
        for frame_name in frame_names:
            image_path = image_dir / frame_name
            if not relational:
                decision = FilterDecision(
                    keep=True,
                    reason="description has no external relation",
                )
            elif not image_path.is_file():
                decision = FilterDecision(
                    keep=False,
                    reason="image file is missing",
                )
            else:
                decision = evaluator(str(image_path), description)
                if not isinstance(decision, FilterDecision):
                    raise RelationalFilterError(
                        "evaluator must return a FilterDecision"
                    )
            if decision.keep:
                kept_names.append(frame_name)
            decision_data = asdict(decision)
            decision_data["image_name"] = frame_name
            decisions.append(decision_data)

        item["image_name"] = kept_names
        filtered.append(item)
        audit.append(
            {
                "desc_id": item.get("desc_id"),
                "description": description,
                "relational": relational,
                "input_frame_count": len(frame_names),
                "kept_frame_count": len(kept_names),
                "decisions": decisions,
            }
        )
    return filtered, audit


def _atomic_write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _discover_inputs(input_root: Path, split: str) -> list[Path]:
    split_root = input_root / split
    if not split_root.is_dir():
        raise FileNotFoundError(f"Step 2 split input not found: {split_root}")
    files = sorted(split_root.glob("*/*_result.json"))
    if not files:
        raise FileNotFoundError(f"No Step 2 *_result.json files found under {split_root}")
    return files


def _video_id(result_path: Path) -> str:
    suffix = "_result.json"
    if not result_path.name.endswith(suffix):
        raise RelationalFilterError(f"unexpected Step 2 filename: {result_path.name}")
    return result_path.name[: -len(suffix)]


def _has_current_audit_schema(audit_path: Path) -> bool:
    """Return whether an audit uses the direct-decision schema."""

    if not audit_path.is_file():
        return False
    try:
        with audit_path.open("r", encoding="utf-8") as handle:
            entries = json.load(handle)
        return isinstance(entries, list) and all(
            isinstance(entry, Mapping)
            and isinstance(entry.get("decisions"), list)
            and all(
                isinstance(decision, Mapping)
                and set(decision)
                == {
                    "keep",
                    "reason",
                    "raw_response",
                    "schema_version",
                    "image_name",
                }
                and decision.get("schema_version") == SCHEMA_VERSION
                for decision in entry["decisions"]
            )
            for entry in entries
        )
    except (OSError, json.JSONDecodeError):
        return False


def run_filter(args: argparse.Namespace) -> dict[str, int]:
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    input_files = _discover_inputs(input_root, args.split)
    sharded_files = input_files[args.shard :: args.num_shards]
    evaluator = QwenRelationalEvaluator(
        args.model,
        local_files_only=args.local_files_only,
        max_new_tokens=args.max_new_tokens,
        min_pixels=args.min_pixels,
        max_pixels=args.max_pixels,
    )
    stats = {
        "files": 0,
        "skipped_files": 0,
        "descriptions": 0,
        "relational_descriptions": 0,
        "input_frames": 0,
        "kept_frames": 0,
    }

    print(
        f"Shard {args.shard}/{args.num_shards}: "
        f"{len(sharded_files)} of {len(input_files)} Step 2 file(s)"
    )
    for input_path in sharded_files:
        relative_path = input_path.relative_to(input_root)
        output_path = output_root / relative_path
        video_id = _video_id(input_path)
        audit_path = output_path.with_name(f"{video_id}_relational_filter.json")
        if (
            output_path.exists()
            and _has_current_audit_schema(audit_path)
            and not args.overwrite
        ):
            print(f"Skip completed: {output_path}")
            stats["skipped_files"] += 1
            continue
        if output_path.exists() and not args.overwrite:
            print(f"Reprocess outdated Step 2b output: {output_path}")
        with input_path.open("r", encoding="utf-8") as handle:
            items = json.load(handle)
        visit_id = input_path.parent.name
        split_dir = "train_val_set" if args.split in ("train", "val") else "test_set"
        image_dir = (
            Path(args.data_root)
            / split_dir
            / visit_id
            / video_id
            / "hires_wide"
        )
        filtered, audit = filter_result_items(items, image_dir, evaluator)
        _atomic_write_json(audit_path, audit)
        _atomic_write_json(output_path, filtered)

        relational_count = sum(bool(entry["relational"]) for entry in audit)
        input_frame_count = sum(entry["input_frame_count"] for entry in audit)
        kept_frame_count = sum(entry["kept_frame_count"] for entry in audit)
        stats["files"] += 1
        stats["descriptions"] += len(audit)
        stats["relational_descriptions"] += relational_count
        stats["input_frames"] += input_frame_count
        stats["kept_frames"] += kept_frame_count
        print(
            f"Saved {output_path}: kept {kept_frame_count}/{input_frame_count} "
            f"frame(s), relational descriptions {relational_count}/{len(audit)}"
        )
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Step 2b: directly filter relational frames with Qwen"
    )
    parser.add_argument("--data_root", default="scenefun3d")
    parser.add_argument("--split", required=True, choices=("train", "val", "test"))
    parser.add_argument("--input_root", default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--min-pixels", type=int, default=None)
    parser.add_argument("--max-pixels", type=int, default=None)
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.num_shards < 1:
        parser.error("--num_shards must be >= 1")
    if not 0 <= args.shard < args.num_shards:
        parser.error("--shard must be in [0, num_shards)")
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

    stats = run_filter(args)
    print(
        "Step 2b complete: "
        f"files={stats['files']} skipped={stats['skipped_files']} "
        f"relational_descriptions={stats['relational_descriptions']}/"
        f"{stats['descriptions']} kept_frames={stats['kept_frames']}/"
        f"{stats['input_frames']}"
    )


if __name__ == "__main__":
    main()
