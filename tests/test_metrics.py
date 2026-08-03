import unittest

from survey_llm_eval.metrics import (
    evaluate_correlations,
    evaluate_marginals,
    pearson_correlation,
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

    def test_correlation_structure_error(self) -> None:
        human = [
            {"A1": 1, "A2": 1},
            {"A1": 2, "A2": 2},
            {"A1": 3, "A2": 3},
        ]
        model = [
            {"A1": 1, "A2": 3},
            {"A1": 2, "A2": 2},
            {"A1": 3, "A2": 1},
        ]
        result = evaluate_correlations(human, model, ["A1", "A2"])
        self.assertEqual(result["eligible_pairs"], 1)
        self.assertAlmostEqual(result["correlation_rmse"], 2.0)

    def test_constant_item_has_no_defined_correlation(self) -> None:
        rows = [{"A1": 1, "A2": 2}, {"A1": 1, "A2": 3}]
        self.assertIsNone(pearson_correlation(rows, "A1", "A2"))


if __name__ == "__main__":
    unittest.main()
