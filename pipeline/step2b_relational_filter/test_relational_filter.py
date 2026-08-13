import json
import tempfile
import unittest
from pathlib import Path

from pipeline.step2b_relational_filter.relational_filter import (
    FilterDecision,
    RelationalFilterError,
    _has_current_audit_schema,
    build_filter_prompt,
    build_parser,
    filter_result_items,
    is_relational_description,
    parse_filter_response,
    run_filter,
)


class RelationDetectionTest(unittest.TestCase):
    def test_detects_external_relations(self):
        for description in (
            "Open the door next to the couch",
            "Use the socket between the radiator and closet",
            "Use the switch to the left of the TV",
            "Pick up the remote on the coffee table",
            "Open the door behind the bin",
        ):
            with self.subTest(description=description):
                self.assertTrue(is_relational_description(description))

    def test_part_modifiers_alone_are_not_external_relations(self):
        for description in (
            "Open the right closet door",
            "Open the left drawer",
            "Turn on the light",
            "Open the second drawer",
        ):
            with self.subTest(description=description):
                self.assertFalse(is_relational_description(description))


class DirectPromptAndParsingTest(unittest.TestCase):
    def test_prompt_is_direct_and_handles_known_ambiguities(self):
        prompt = build_filter_prompt(
            "Plug the device in the left socket between the radiator and closet"
        )
        normalized_prompt = " ".join(prompt.split())
        self.assertIn("physically operated", prompt)
        self.assertIn("partial object at the image edge counts", prompt)
        self.assertIn("carried device", prompt)
        self.assertIn("object controlled by a visible switch", normalized_prompt)
        self.assertIn("Answer only YES or NO", prompt)
        for removed_concept in (
            "target_point",
            "reference_points",
            "relation_checks",
            "coordinates",
            '"keep"',
        ):
            self.assertNotIn(removed_concept, prompt)

    def test_parses_direct_yes_or_no(self):
        decision = parse_filter_response(" YES. \n")
        self.assertTrue(decision.keep)
        self.assertEqual(decision.reason, "Qwen answered YES")
        self.assertEqual(decision.schema_version, 4)
        self.assertFalse(parse_filter_response("NO").keep)

    def test_rejects_invalid_response_shapes(self):
        for response in (
            "",
            "maybe",
            "YES because it is visible",
            '{"keep":true}',
        ):
            with self.subTest(response=response):
                with self.assertRaises(RelationalFilterError):
                    parse_filter_response(response)


class ResultFilteringTest(unittest.TestCase):
    def test_directly_filters_relational_frames_and_passes_plain_items(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            image_dir = Path(temporary_dir)
            for name in ("good.jpg", "bad.jpg", "plain.jpg"):
                (image_dir / name).touch()

            calls = []

            def evaluator(image_path, description):
                calls.append((Path(image_path).name, description))
                if Path(image_path).name == "good.jpg":
                    return FilterDecision(True, "door and couch are visible")
                return FilterDecision(False, "couch is absent")

            items = [
                {
                    "desc_id": "rel",
                    "description": "Open the door next to the couch",
                    "image_name": ["good.jpg", "bad.jpg", "missing.jpg"],
                },
                {
                    "desc_id": "plain",
                    "description": "Flush the toilet",
                    "image_name": ["plain.jpg"],
                },
            ]
            filtered, audit = filter_result_items(items, image_dir, evaluator)

            self.assertEqual(filtered[0]["image_name"], ["good.jpg"])
            self.assertEqual(filtered[1], items[1])
            self.assertEqual([call[0] for call in calls], ["good.jpg", "bad.jpg"])
            self.assertEqual(audit[0]["kept_frame_count"], 1)
            self.assertEqual(
                audit[0]["decisions"][2]["reason"], "image file is missing"
            )
            self.assertFalse(audit[1]["relational"])
            self.assertEqual(
                set(audit[0]["decisions"][0]),
                {"keep", "reason", "raw_response", "schema_version", "image_name"},
            )

    def test_current_schema_requires_only_direct_fields(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "audit.json"
            direct = [
                {
                    "decisions": [
                        {
                            "keep": True,
                            "reason": "visible",
                            "raw_response": "{}",
                            "schema_version": 4,
                            "image_name": "frame.jpg",
                        }
                    ]
                }
            ]
            path.write_text(json.dumps(direct))
            self.assertTrue(_has_current_audit_schema(path))
            direct[0]["decisions"][0]["target_point"] = [0.5, 0.5]
            path.write_text(json.dumps(direct))
            self.assertFalse(_has_current_audit_schema(path))

    def test_non_relational_cli_path_does_not_load_qwen(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            input_root = root / "step2"
            result_dir = input_root / "val" / "visit"
            result_dir.mkdir(parents=True)
            source = [
                {
                    "visit_id": "visit",
                    "video_id": "video",
                    "desc_id": "plain",
                    "description": "Flush the toilet",
                    "image_name": ["frame.jpg"],
                }
            ]
            (result_dir / "video_result.json").write_text(json.dumps(source))
            output_root = root / "step2b"
            args = build_parser().parse_args(
                [
                    "--data_root",
                    str(root / "data"),
                    "--split",
                    "val",
                    "--input_root",
                    str(input_root),
                    "--output_root",
                    str(output_root),
                ]
            )

            stats = run_filter(args)
            result_path = output_root / "val" / "visit" / "video_result.json"
            self.assertEqual(json.loads(result_path.read_text()), source)
            self.assertEqual(stats["kept_frames"], 1)

            second_stats = run_filter(args)
            self.assertEqual(second_stats["files"], 0)
            self.assertEqual(second_stats["skipped_files"], 1)


if __name__ == "__main__":
    unittest.main()
