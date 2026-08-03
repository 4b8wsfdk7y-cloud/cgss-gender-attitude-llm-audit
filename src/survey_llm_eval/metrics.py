"""Deterministic fidelity diagnostics for ordinal survey responses."""

from __future__ import annotations

from collections import Counter, defaultdict
from itertools import combinations
from math import fsum, sqrt
from typing import Any, Iterable, Mapping, Sequence


def _scores(rows: Iterable[Mapping[str, Any]], item: str) -> list[int]:
    values: list[int] = []
    for row in rows:
        value = row.get(item)
        if value not in (None, ""):
            values.append(int(value))
    return values


def _mean(values: Sequence[int]) -> float:
    if not values:
        raise ValueError("Cannot calculate a mean from no observations")
    return fsum(values) / len(values)


def _variance(values: Sequence[int]) -> float:
    mean = _mean(values)
    return fsum((value - mean) ** 2 for value in values) / len(values)


def empirical_distribution(
    values: Sequence[int], scale_values: Sequence[int]
) -> dict[int, float]:
    if not values:
        raise ValueError("Cannot calculate a distribution from no observations")
    counts = Counter(values)
    return {value: counts[value] / len(values) for value in scale_values}


def total_variation(
    left: Mapping[int, float], right: Mapping[int, float]
) -> float:
    support = set(left) | set(right)
    return 0.5 * fsum(abs(left.get(key, 0.0) - right.get(key, 0.0)) for key in support)


def evaluate_marginals(
    human_rows: Sequence[Mapping[str, Any]],
    model_rows: Sequence[Mapping[str, Any]],
    items: Sequence[str],
    scale_values: Sequence[int],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for item in items:
        human = _scores(human_rows, item)
        model = _scores(model_rows, item)
        human_variance = _variance(human)
        model_variance = _variance(model)
        diagnostics.append(
            {
                "item": item,
                "human_n": len(human),
                "model_n": len(model),
                "human_mean": round(_mean(human), 6),
                "model_mean": round(_mean(model), 6),
                "absolute_mean_error": round(abs(_mean(model) - _mean(human)), 6),
                "human_variance": round(human_variance, 6),
                "model_variance": round(model_variance, 6),
                "variance_ratio": (
                    round(model_variance / human_variance, 6)
                    if human_variance > 0
                    else None
                ),
                "total_variation": round(
                    total_variation(
                        empirical_distribution(human, scale_values),
                        empirical_distribution(model, scale_values),
                    ),
                    6,
                ),
            }
        )
    return diagnostics


def evaluate_subgroups(
    human_rows: Sequence[Mapping[str, Any]],
    model_rows: Sequence[Mapping[str, Any]],
    items: Sequence[str],
    scale_values: Sequence[int],
    subgroups: Sequence[str],
) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for subgroup in subgroups:
        human_levels = {str(row[subgroup]) for row in human_rows}
        model_levels = {str(row[subgroup]) for row in model_rows}
        for level in sorted(human_levels & model_levels):
            human_subset = [row for row in human_rows if str(row[subgroup]) == level]
            model_subset = [row for row in model_rows if str(row[subgroup]) == level]
            for result in evaluate_marginals(
                human_subset, model_subset, items, scale_values
            ):
                diagnostics.append(
                    {
                        "subgroup": subgroup,
                        "level": level,
                        **result,
                    }
                )
    return diagnostics


def pearson_correlation(
    rows: Sequence[Mapping[str, Any]], left_item: str, right_item: str
) -> float | None:
    """Return a pairwise-complete Pearson correlation for two survey items."""

    pairs = [
        (int(row[left_item]), int(row[right_item]))
        for row in rows
        if row.get(left_item) not in (None, "")
        and row.get(right_item) not in (None, "")
    ]
    if len(pairs) < 2:
        return None

    left_values = [pair[0] for pair in pairs]
    right_values = [pair[1] for pair in pairs]
    left_mean = _mean(left_values)
    right_mean = _mean(right_values)
    numerator = fsum(
        (left - left_mean) * (right - right_mean)
        for left, right in pairs
    )
    left_sum_squares = fsum((value - left_mean) ** 2 for value in left_values)
    right_sum_squares = fsum((value - right_mean) ** 2 for value in right_values)
    denominator = sqrt(left_sum_squares * right_sum_squares)
    if denominator == 0:
        return None
    return numerator / denominator


def evaluate_correlations(
    human_rows: Sequence[Mapping[str, Any]],
    model_rows: Sequence[Mapping[str, Any]],
    items: Sequence[str],
) -> dict[str, Any]:
    """Compare human and model pairwise correlations without extra dependencies."""

    diagnostics: list[dict[str, Any]] = []
    squared_errors: list[float] = []
    for left_item, right_item in combinations(items, 2):
        human_correlation = pearson_correlation(
            human_rows, left_item, right_item
        )
        model_correlation = pearson_correlation(
            model_rows, left_item, right_item
        )
        absolute_error = None
        if human_correlation is not None and model_correlation is not None:
            error = model_correlation - human_correlation
            absolute_error = abs(error)
            squared_errors.append(error**2)
        diagnostics.append(
            {
                "left_item": left_item,
                "right_item": right_item,
                "human_correlation": (
                    round(human_correlation, 6)
                    if human_correlation is not None
                    else None
                ),
                "model_correlation": (
                    round(model_correlation, 6)
                    if model_correlation is not None
                    else None
                ),
                "absolute_error": (
                    round(absolute_error, 6)
                    if absolute_error is not None
                    else None
                ),
            }
        )

    return {
        "eligible_pairs": len(squared_errors),
        "total_pairs": len(diagnostics),
        "correlation_rmse": (
            round(sqrt(fsum(squared_errors) / len(squared_errors)), 6)
            if squared_errors
            else None
        ),
        "pairs": diagnostics,
    }


def within_profile_agreement(
    model_rows: Sequence[Mapping[str, Any]], items: Sequence[str]
) -> dict[str, float | int | None]:
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for row in model_rows:
        profile_id = str(row["profile_id"])
        for item in items:
            if row.get(item) not in (None, ""):
                grouped[(profile_id, item)].append(int(row[item]))

    agreements: list[float] = []
    for values in grouped.values():
        if len(values) > 1:
            agreements.append(max(Counter(values).values()) / len(values))
    return {
        "profile_item_cells": len(agreements),
        "mean_modal_agreement": (
            round(fsum(agreements) / len(agreements), 6) if agreements else None
        ),
    }
