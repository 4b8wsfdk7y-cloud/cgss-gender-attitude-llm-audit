"""Diagnostics for option-level predictive distributions over an ordinal scale."""

from __future__ import annotations

from math import exp, fsum, log
from typing import Mapping, Sequence

Distribution = Mapping[int, float]


def normalize_option_logprobs(
    logprobs: Mapping[int, float],
) -> tuple[dict[int, float], float]:
    """Renormalize option log-probabilities over the response scale.

    Returns the normalized distribution and the probability mass the model
    placed on valid options before renormalization. A low mass means the model
    preferred tokens outside the response scale.
    """
    if not logprobs:
        raise ValueError("At least one option log-probability is required")
    peak = max(logprobs.values())
    weights = {option: exp(value - peak) for option, value in logprobs.items()}
    total = fsum(weights.values())
    mass = exp(peak) * total
    distribution = {option: weight / total for option, weight in weights.items()}
    return distribution, mass


def expected_score(distribution: Distribution) -> float:
    return fsum(option * probability for option, probability in distribution.items())


def predictive_variance(distribution: Distribution) -> float:
    mean = expected_score(distribution)
    return fsum(
        probability * (option - mean) ** 2
        for option, probability in distribution.items()
    )


def entropy(distribution: Distribution) -> float:
    """Shannon entropy in nats."""
    return -fsum(
        probability * log(probability)
        for probability in distribution.values()
        if probability > 0
    )


def modal_option(distribution: Distribution) -> int:
    """Most probable option; ties resolve to the lower scale value."""
    return min(distribution, key=lambda option: (-distribution[option], option))


def ranked_probability_score(
    distribution: Distribution,
    observed: int,
    scale_values: Sequence[int],
) -> float:
    """Ranked probability score for an ordinal outcome, scaled to [0, 1]."""
    if observed not in scale_values:
        raise ValueError(f"Observed value {observed!r} is outside the scale")
    ordered = sorted(scale_values)
    cumulative_forecast = 0.0
    total = 0.0
    for option in ordered[:-1]:
        cumulative_forecast += distribution.get(option, 0.0)
        cumulative_observed = 1.0 if observed <= option else 0.0
        total += (cumulative_forecast - cumulative_observed) ** 2
    return total / (len(ordered) - 1)


def log_score(distribution: Distribution, observed: int, floor: float = 1e-12) -> float:
    """Negative log-likelihood of the observed option."""
    return -log(max(distribution.get(observed, 0.0), floor))


def mixture(distributions: Sequence[Distribution]) -> dict[int, float]:
    """Population distribution implied by averaging profile-level forecasts."""
    if not distributions:
        raise ValueError("Cannot mix an empty set of distributions")
    support = sorted({option for row in distributions for option in row})
    count = len(distributions)
    return {
        option: fsum(row.get(option, 0.0) for row in distributions) / count
        for option in support
    }


def variance_decomposition(distributions: Sequence[Distribution]) -> dict[str, float]:
    """Split population predictive variance into between- and within-profile parts.

    Uses the law of total variance over equally weighted profiles. The
    between-profile component reflects stable differences in expected scores;
    the within-profile component reflects uncertainty inside each forecast.
    """
    if not distributions:
        raise ValueError("Cannot decompose an empty set of distributions")
    means = [expected_score(row) for row in distributions]
    grand_mean = fsum(means) / len(means)
    between = fsum((mean - grand_mean) ** 2 for mean in means) / len(means)
    within = fsum(predictive_variance(row) for row in distributions) / len(distributions)
    total = between + within
    return {
        "between_profile_variance": between,
        "within_profile_variance": within,
        "total_predictive_variance": total,
        "within_share": within / total if total > 0 else 0.0,
    }
