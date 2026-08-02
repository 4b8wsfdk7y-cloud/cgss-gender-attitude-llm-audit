import json
import tempfile
import unittest
from pathlib import Path

from survey_llm_eval.runs import assert_output_compatible, load_completed


class RunLedgerTest(unittest.TestCase):
    def _ledger(self, records: list[dict]) -> Path:
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "responses.jsonl"
        path.write_text(
            "\n".join(json.dumps(record) for record in records) + "\n",
            encoding="utf-8",
        )
        return path

    def test_completed_key_retains_prompt_hash(self) -> None:
        path = self._ledger(
            [
                {
                    "profile_id": "p1",
                    "condition": "neutral",
                    "repeat": 1,
                    "prompt_sha256": "abc",
                    "success": True,
                }
            ]
        )
        self.assertEqual(load_completed(path), {("p1", "neutral", 1): "abc"})

    def test_different_model_is_rejected(self) -> None:
        path = self._ledger(
            [{"model": "qwen3:8b", "model_digest": "digest-1"}]
        )
        with self.assertRaisesRegex(RuntimeError, "separate --output-dir"):
            assert_output_compatible(path, "qwen/qwen3.5-9b", "digest-2")

    def test_matching_model_and_digest_are_allowed(self) -> None:
        path = self._ledger(
            [{"model": "qwen3:8b", "model_digest": "digest-1"}]
        )
        assert_output_compatible(path, "qwen3:8b", "digest-1")


if __name__ == "__main__":
    unittest.main()
