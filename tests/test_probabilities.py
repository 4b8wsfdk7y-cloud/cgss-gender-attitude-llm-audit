import unittest
from math import log

from survey_llm_eval.probabilities import (
    entropy,
    expected_score,
    log_score,
    mixture,
    modal_option,
    normalize_option_logprobs,
    ranked_probability_score,
    variance_decomposition,
)

SCALE = [1, 2, 3, 4, 5]


class ProbabilitiesTest(unittest.TestCase):
    def test_normalization_reports_valid_option_mass(self) -> None:
        logprobs = {1: log(0.1), 2: log(0.2), 3: log(0.1)}
        distribution, mass = normalize_option_logprobs(logprobs)
        self.assertAlmostEqual(mass, 0.4)
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertAlmostEqual(distribution[2], 0.5)

    def test_expected_score_entropy_and_mode(self) -> None:
        certain = {1: 0.0, 2: 0.0, 3: 0.0, 4: 1.0, 5: 0.0}
        self.assertEqual(expected_score(certain), 4.0)
        self.assertEqual(entropy(certain), 0.0)
        self.assertEqual(modal_option({1: 0.4, 2: 0.4, 3: 0.2}), 1)

    def test_ranked_probability_score_bounds(self) -> None:
        perfect = {1: 0.0, 2: 0.0, 3: 1.0, 4: 0.0, 5: 0.0}
        worst = {1: 1.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        self.assertAlmostEqual(ranked_probability_score(perfect, 3, SCALE), 0.0)
        self.assertAlmostEqual(ranked_probability_score(worst, 5, SCALE), 1.0)
        with self.assertRaises(ValueError):
            ranked_probability_score(perfect, 7, SCALE)

    def test_log_score_uses_floor(self) -> None:
        self.assertAlmostEqual(log_score({1: 0.5, 2: 0.5}, 2), -log(0.5))
        self.assertGreater(log_score({1: 1.0}, 2), 20)

    def test_mixture_and_variance_decomposition(self) -> None:
        low = {1: 1.0, 5: 0.0}
        high = {1: 0.0, 5: 1.0}
        self.assertEqual(mixture([low, high]), {1: 0.5, 5: 0.5})
        split = variance_decomposition([low, high])
        self.assertAlmostEqual(split["between_profile_variance"], 4.0)
        self.assertAlmostEqual(split["within_profile_variance"], 0.0)
        uncertain = {1: 0.5, 5: 0.5}
        noisy = variance_decomposition([uncertain, uncertain])
        self.assertAlmostEqual(noisy["between_profile_variance"], 0.0)
        self.assertAlmostEqual(noisy["within_share"], 1.0)


if __name__ == "__main__":
    unittest.main()
