"""Reusable evaluation components for LLM-generated survey responses."""

from .metrics import (
    evaluate_correlations,
    evaluate_marginals,
    evaluate_subgroups,
    pearson_correlation,
    total_variation,
)
from .spec import BenchmarkSpec

__all__ = [
    "BenchmarkSpec",
    "evaluate_correlations",
    "evaluate_marginals",
    "evaluate_subgroups",
    "pearson_correlation",
    "total_variation",
]

__version__ = "0.1.0"
