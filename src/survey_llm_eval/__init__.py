"""Reusable evaluation components for LLM-generated survey responses."""

from .metrics import evaluate_marginals, evaluate_subgroups, total_variation
from .spec import BenchmarkSpec

__all__ = [
    "BenchmarkSpec",
    "evaluate_marginals",
    "evaluate_subgroups",
    "total_variation",
]

__version__ = "0.1.0"
