"""Command-line interface for the public evaluation package."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .demo import DEFAULT_HUMAN, DEFAULT_OUTPUT, DEFAULT_SPEC, run_demo


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
            "output_dir": args.output_dir,
        }
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    raise RuntimeError(f"Unknown command: {args.command}")
