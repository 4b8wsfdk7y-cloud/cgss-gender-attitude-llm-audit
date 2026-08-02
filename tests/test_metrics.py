import unittest

from survey_llm_eval.metrics import (
    evaluate_marginals,
    total_variation,
    within_profile_agreement,
)


class MetricsTest(unittest.TestCase):
    def test_total_variation(self) -> None:
        left = {1: 0.5, 2: 0.5}
        right = {1: 0.0, 2: 1.0}
        self.assertAlmostEqual(total_variation(left, right), 0.5)

    def test_marginal_diagnostics(self) -> None:
        human = [{"A1": "1"}, {"A1": "2"}]
        model = [{"A1": 2}, {"A1": 2}]
        result = evaluate_marginals(human, model, ["A1"], [1, 2])[0]
        self.assertEqual(result["absolute_mean_error"], 0.5)
        self.assertEqual(result["total_variation"], 0.5)
        self.assertEqual(result["variance_ratio"], 0.0)

    def test_repeat_agreement_is_a_separate_diagnostic(self) -> None:
        rows = [
            {"profile_id": "p1", "A1": 1},
            {"profile_id": "p1", "A1": 1},
            {"profile_id": "p1", "A1": 2},
        ]
        result = within_profile_agreement(rows, ["A1"])
        self.assertEqual(result["profile_item_cells"], 1)
        self.assertAlmostEqual(result["mean_modal_agreement"], 2 / 3, places=6)


if __name__ == "__main__":
    unittest.main()
