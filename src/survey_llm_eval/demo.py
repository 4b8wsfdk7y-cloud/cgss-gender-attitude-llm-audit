"""End-to-end public demo using synthetic records and a deterministic mock model."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .io import read_csv, write_csv, write_json
from .metrics import evaluate_marginals, evaluate_subgroups, within_profile_agreement
from .model import DeterministicMockModel
from .spec import BenchmarkSpec


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC = REPOSITORY_ROOT / "benchmarks" / "cgss_gender_attitudes.json"
DEFAULT_HUMAN = REPOSITORY_ROOT / "fixtures" / "demo_human_synthetic.csv"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "output" / "public_demo"


def run_demo(
    *,
    spec_path: str | Path = DEFAULT_SPEC,
    human_path: str | Path = DEFAULT_HUMAN,
    output_dir: str | Path = DEFAULT_OUTPUT,
    repeats: int = 3,
) -> dict[str, Any]:
    if repeats < 2:
        raise ValueError("The stability diagnostic requires at least two repeats")

    spec = BenchmarkSpec.from_path(spec_path)
    human_rows = read_csv(human_path)
    for row in human_rows:
        spec.validate_row(row, require_items=True)

    model = DeterministicMockModel()
    model_rows: list[dict[str, Any]] = []
    for human_row in human_rows:
        profile = {field: human_row[field] for field in spec.profile_fields}
        for repeat in range(1, repeats + 1):
            model_rows.append(
                {
                    **profile,
                    "repeat": repeat,
                    **model.generate(
                        profile,
                        spec.items,
                        repeat=repeat,
                        minimum=spec.scale_minimum,
                        maximum=spec.scale_maximum,
                    ),
                }
            )

    report = {
        "benchmark": spec.name,
        "benchmark_spec_sha256": spec.sha256,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "demo_only": True,
        "data_notice": (
            "All inputs are synthetic. The mock adapter exercises the evaluation "
            "pipeline and does not represent an LLM or a CGSS result."
        ),
        "human_records": len(human_rows),
        "model_records": len(model_rows),
        "repeats": repeats,
        "marginal_diagnostics": evaluate_marginals(
            human_rows, model_rows, spec.items, spec.scale_values
        ),
        "subgroup_diagnostics": evaluate_subgroups(
            human_rows,
            model_rows,
            spec.items,
            spec.scale_values,
            spec.subgroups,
        ),
        "repeat_stability": within_profile_agreement(model_rows, spec.items),
    }

    output_path = Path(output_dir)
    write_csv(output_path / "mock_responses.csv", model_rows)
    write_json(output_path / "demo_report.json", report)
    return report
