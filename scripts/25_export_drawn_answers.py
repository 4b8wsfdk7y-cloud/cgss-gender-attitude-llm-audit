#!/usr/bin/env python3
# requires: Python 3.10+, numpy, the survey_llm_eval package
"""Export synthetic answer datasets for the downstream statistics
(PREREGISTRATION_V2.md, 2026-11-15 entry).

Rebuilds the drawn answers of the sampled-history robustness check
(scripts/14_analyze_sampled_history.py: five seeded paths per profile; history
cells condition on earlier drawn answers, the other cells draw from the primary
distributions) and the most probable answer per item and cell, and writes them
with the matched CGSS respondents' answers and the profile attributes used as
controls. The output is profile-level and must be written outside the
repository; scripts/26_downstream_statistics.R reads it.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "analyze_sampled_history", REPO_ROOT / "scripts" / "14_analyze_sampled_history.py")
run14 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run14)

from survey_llm_eval.factorial import CELLS, item_ids  # noqa: E402

COVARIATES = ("female", "educ_years", "age", "urban_residence")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="option_probabilities_greedy.jsonl")
    parser.add_argument("--sampled-ledger", required=True, help="option_probabilities_sampled.jsonl")
    parser.add_argument("--matched", required=True, help="profiles_pilot.csv (matched answers)")
    parser.add_argument("--output", required=True, help="profile-level CSV, outside the repository")
    parser.add_argument("--paths", type=int, default=5)
    parser.add_argument("--draw-seed", type=int, default=20261013,
                        help="seed passed to 07_run_logprob_audit.py --seed")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = Path(args.output).resolve()
    if REPO_ROOT in output.parents:
        raise SystemExit("--output must be outside the repository: the file is profile-level")
    if list(run14.SCALE) != [1, 2, 3, 4, 5]:
        raise SystemExit("unexpected response scale")

    cells = [c.name for c in CELLS]
    history_cells = [c.name for c in CELLS if c.history]
    plain_cells = [c.name for c in CELLS if not c.history]
    _, profiles, probs, _ = run14.run09.load_ledger(Path(args.ledger), cells)
    _, _, _, drawn_history = run14.load_sampled(
        Path(args.sampled_ledger), history_cells, profiles, args.paths, args.draw_seed)
    drawn = np.zeros((len(cells), len(profiles), args.paths, len(item_ids())), dtype=int)
    drawn[[cells.index(c) for c in history_cells]] = drawn_history
    drawn[[cells.index(c) for c in plain_cells]] = run14.draws_without_history(
        Path(args.ledger), plain_cells, profiles, args.paths, args.draw_seed)
    modal = probs.argmax(axis=-1) + 1

    attributes = {r["profile_id"]: r for r in csv.DictReader(open(args.matched, encoding="utf-8"))}
    items = [i.lower() for i in item_ids()]
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["source", "cell", "path", "profile_id", "wave", *items, *COVARIATES])
        for p, (wave, pid) in enumerate(profiles):
            a = attributes[pid]
            covariates = [a[c] for c in COVARIATES]
            writer.writerow(["H-matched", "", "", pid, wave, *[a[i] for i in items], *covariates])
            for c, cell in enumerate(cells):
                for k in range(args.paths):
                    writer.writerow(["S-drawn", cell, k, pid, wave, *drawn[c, p, k], *covariates])
                writer.writerow(["S-modal", cell, "", pid, wave, *modal[c, p], *covariates])
    print(f"wrote {len(profiles)} profiles x {len(cells)} cells to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
