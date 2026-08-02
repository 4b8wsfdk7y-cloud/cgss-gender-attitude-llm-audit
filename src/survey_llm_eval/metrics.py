"""Deterministic fidelity diagnostics for ordinal survey responses."""

from __future__ import annotations

from collections import Counter, defaultdict
from math import fsum
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
