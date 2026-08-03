"""Command-line interface for the public evaluation package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .demo import (
    DEFAULT_HUMAN,
    DEFAULT_OUTPUT,
    DEFAULT_SPEC,
    evaluate_csv_files,
    run_demo,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="survey-llm-eval")
    subparsers = parser.add_subparsers(dest="command", required=True)
    demo_parser = subparsers.add_parser(
        "demo", help="run the dependency-free synthetic evaluation demo"
    )
    demo_parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    demo_parser.add_argument("--human", default=str(DEFAULT_HUMAN))
    demo_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    demo_parser.add_argument("--repeats", type=int, default=3)

    evaluate_parser = subparsers.add_parser(
        "evaluate", help="evaluate human-reference and model-response CSV files"
    )
    evaluate_parser.add_argument("--spec", default=str(DEFAULT_SPEC))
    evaluate_parser.add_argument("--human", required=True)
    evaluate_parser.add_argument("--model", required=True)
    evaluate_parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "demo":
        report = run_demo(
            spec_path=Path(args.spec),
            human_path=Path(args.human),
            output_dir=Path(args.output_dir),
            repeats=args.repeats,
        )
        summary = {
            "benchmark": report["benchmark"],
            "demo_only": report["demo_only"],
            "human_records": report["human_records"],
            "model_records": report["model_records"],
            "correlation_rmse": report["relational_diagnostics"][
                "correlation_rmse"
            ],
            "output_dir": args.output_dir,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "evaluate":
        report = evaluate_csv_files(
            spec_path=Path(args.spec),
            human_path=Path(args.human),
            model_path=Path(args.model),
            output_path=Path(args.output),
        )
        summary = {
            "benchmark": report["benchmark"],
            "report_type": report["report_type"],
            "human_records": report["human_records"],
            "model_records": report["model_records"],
            "correlation_pairs": report["relational_diagnostics"][
                "eligible_pairs"
            ],
            "correlation_rmse": report["relational_diagnostics"][
                "correlation_rmse"
            ],
            "output": args.output,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    raise RuntimeError(f"Unknown command: {args.command}")
