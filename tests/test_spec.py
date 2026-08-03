import json
import tempfile
import unittest
from pathlib import Path

from survey_llm_eval.spec import BenchmarkSpec


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class BenchmarkSpecTest(unittest.TestCase):
    def test_repository_spec_is_valid(self) -> None:
        spec = BenchmarkSpec.from_path(
            REPOSITORY_ROOT / "benchmarks" / "cgss_gender_attitudes.json"
        )
        self.assertEqual(spec.items, ("A421", "A422", "A423", "A424", "A425"))
        self.assertEqual(spec.scale_values, (1, 2, 3, 4, 5))
        self.assertEqual(len(spec.sha256), 64)

    def test_duplicate_items_are_rejected(self) -> None:
        payload = {
            "name": "invalid",
            "profile_fields": ["profile_id"],
            "items": ["A1", "A1"],
            "scale": {"minimum": 1, "maximum": 5},
            "subgroups": [],
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "spec.json"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "non-empty and unique"):
                BenchmarkSpec.from_path(path)

    def test_current_and_legacy_model_configs_are_explicit(self) -> None:
        legacy_config = json.loads(
            (REPOSITORY_ROOT / "config.json").read_text(encoding="utf-8")
        )
        current_config = json.loads(
            (REPOSITORY_ROOT / "config_qwen35_lmstudio.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(legacy_config["provider"], "ollama")
        self.assertEqual(legacy_config["model"], "qwen3:8b")
        self.assertEqual(current_config["provider"], "lmstudio")
        self.assertEqual(current_config["model"], "qwen/qwen3.5-9b")
        self.assertNotEqual(legacy_config["model"], current_config["model"])


if __name__ == "__main__":
    unittest.main()
