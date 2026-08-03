import tempfile
import unittest
from pathlib import Path

from survey_llm_eval.demo import (
    DEFAULT_HUMAN,
    DEFAULT_SPEC,
    evaluate_csv_files,
    run_demo,
)
from survey_llm_eval.model import DeterministicMockModel


class DemoTest(unittest.TestCase):
    def test_mock_model_is_reproducible(self) -> None:
        model = DeterministicMockModel()
        profile = {"profile_id": "p1", "wave": "2021", "sex": "woman"}
        first = model.generate(profile, ["A1"], repeat=1, minimum=1, maximum=5)
        second = model.generate(profile, ["A1"], repeat=1, minimum=1, maximum=5)
        self.assertEqual(first, second)

    def test_end_to_end_demo(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report = run_demo(
                spec_path=DEFAULT_SPEC,
                human_path=DEFAULT_HUMAN,
                output_dir=directory,
                repeats=3,
            )
            self.assertTrue(report["demo_only"])
            self.assertEqual(report["human_records"], 12)
            self.assertEqual(report["model_records"], 36)
            self.assertEqual(len(report["marginal_diagnostics"]), 5)
            self.assertEqual(report["relational_diagnostics"]["total_pairs"], 10)
            self.assertTrue((Path(directory) / "demo_report.json").exists())
            self.assertTrue((Path(directory) / "mock_responses.csv").exists())

    def test_user_supplied_csv_evaluation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            demo_dir = Path(directory) / "demo"
            run_demo(output_dir=demo_dir, repeats=3)
            output_path = Path(directory) / "evaluation.json"
            report = evaluate_csv_files(
                spec_path=DEFAULT_SPEC,
                human_path=DEFAULT_HUMAN,
                model_path=demo_dir / "mock_responses.csv",
                output_path=output_path,
            )
            self.assertEqual(report["report_type"], "user_supplied_evaluation")
            self.assertEqual(report["relational_diagnostics"]["total_pairs"], 10)
            self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
