#!/usr/bin/env python3
# requires: Python 3.10+, numpy, the survey_llm_eval package
"""Exploratory cells V1-T1 and V1-T0 (PREREGISTRATION_V2.md, 2026-10-27).

Compares the v1 joint prompt scored with thinking open (V1-T1, the v1
context) and closed (V1-T0) with the v2 reference cell A1-B1-C1, using the
Part 1 metrics, the Part 2 primary outcome, and the same profile-cluster
bootstrap as 09_analyze_logprob_audit.py. Also summarizes the A421 diagnostic
that preceded the cells: how well each context accounts for the answers v1
actually sampled. Only aggregate tables are written.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "analyze_logprob_audit", REPO_ROOT / "scripts" / "09_analyze_logprob_audit.py")
run09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run09)

from survey_llm_eval.factorial import item_ids  # noqa: E402

SCALE = run09.SCALE
REFERENCE = run09.REFERENCE_CELL
V1_CELLS = ["V1-T1", "V1-T0"]
CONTRASTS = (("V1-T1", "V1-T0"), ("V1-T0", REFERENCE), ("V1-T1", REFERENCE))
V1_TEMPERATURE, V1_TOP_P = 0.7, 0.9


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="option_probabilities_greedy.jsonl")
    parser.add_argument("--v1-context-ledger", required=True,
                        help="option_probabilities_v1_context.jsonl")
    parser.add_argument("--matched", required=True, help="profiles_pilot.csv (matched answers)")
    parser.add_argument("--human-dir", required=True, help="output of 08_export_human_benchmark.R")
    parser.add_argument("--v1-responses", required=True, help="v1 Qwen3.5 responses.jsonl")
    parser.add_argument("--diagnostic", help="A421 open/closed diagnostic jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def v1_decoded(dist: dict[int, float]) -> dict[int, float]:
    """The v1 sampling distribution over 1-5: temperature 0.7, then top-p 0.9."""
    weights = {k: v ** (1 / V1_TEMPERATURE) for k, v in dist.items()}
    total = sum(weights.values())
    ranked = sorted(((v / total, k) for k, v in weights.items()), reverse=True)
    kept, mass = {}, 0.0
    for v, k in ranked:
        kept[k] = v
        mass += v
        if mass >= V1_TOP_P:
            break
    total = sum(kept.values())
    return {k: kept.get(k, 0.0) / total for k in dist}


def expected(dist: dict[int, float]) -> float:
    return sum(k * v for k, v in dist.items())


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = [i.lower() for i in item_ids()]

    n_ref, profiles, ref_probs, ref_mass = run09.load_ledger(Path(args.ledger), [REFERENCE])
    n_v1, _, v1_probs, v1_mass = run09.load_ledger(Path(args.v1_context_ledger), V1_CELLS, profiles)
    cells = [REFERENCE] + V1_CELLS
    probs = np.concatenate([ref_probs, v1_probs])
    mass = np.concatenate([ref_mass, v1_mass])
    waves = np.array([wave for wave, _ in profiles])
    wave_names = sorted(set(waves))
    wave_index = [np.flatnonzero(waves == w) for w in wave_names]

    human_dir = Path(args.human_dir)
    moments = run09.read_csv(human_dir / "human_moments.csv")
    human_mean = {(r["wave"], r["item"]): float(r["mean"]) for r in moments}
    human_var = {(r["wave"], r["item"]): float(r["variance"]) for r in moments}
    human_marginal: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in run09.read_csv(human_dir / "human_marginals.csv"):
        human_marginal[(r["wave"], r["item"])][int(r["option"])] = float(r["probability"])
    matched = {r["profile_id"]: r for r in run09.read_csv(Path(args.matched))}

    item_rows = run09.item_metrics(cells, probs, mass, profiles, wave_names, wave_index, matched,
                                   (human_mean, human_var, human_marginal))
    run09.write_csv(out / "v1_context_item_metrics.csv", [r for r in item_rows if r["cell"] != REFERENCE])

    # Primary outcome and bootstrap contrasts.
    scale = np.array(SCALE, dtype=float)
    exp_scores = probs @ scale
    outcome, by_wave = run09.primary_outcome(exp_scores, wave_index)
    rng = np.random.default_rng(args.seed)
    boot = np.empty((args.replicates, len(cells)))
    for b in range(args.replicates):
        draw = [rng.choice(idx, size=len(idx), replace=True) for idx in wave_index]
        boot[b], _ = run09.primary_outcome(exp_scores, draw)

    cell_rows = []
    for c, cell in enumerate(cells):
        low, high = np.percentile(boot[:, c], [2.5, 97.5])
        cell_rows.append({
            "cell": cell, "a425_mean_abs_correlation": outcome[c], "ci95_low": low, "ci95_high": high,
            **{f"wave_{w}": by_wave[c, k] for k, w in enumerate(wave_names)},
            **{k: v for k, v in run09.cell_summary(item_rows, cell).items() if k != "cell"},
        })
    run09.write_csv(out / "v1_context_cells.csv", cell_rows)

    contrast_rows = []
    for a, b in CONTRASTS:
        ia, ib = cells.index(a), cells.index(b)
        diff = boot[:, ia] - boot[:, ib]
        low, high = np.percentile(diff, [2.5, 97.5])
        row = {"contrast": f"{a} minus {b}", "metric": "a425_mean_abs_correlation",
               "estimate": outcome[ia] - outcome[ib], "ci95_low": low, "ci95_high": high,
               "interval_excludes_zero": bool(low > 0 or high < 0)}
        contrast_rows.append(row)
        sa, sb = run09.cell_summary(item_rows, a), run09.cell_summary(item_rows, b)
        for key in ("mean_abs_mean_error", "mean_total_variation", "mean_between_variance_ratio",
                    "mean_within_share", "mean_matched_rps", "mean_matched_log_score",
                    "mean_valid_option_mass"):
            contrast_rows.append({"contrast": f"{a} minus {b}", "metric": key,
                                  "estimate": sa[key] - sb[key], "ci95_low": "", "ci95_high": "",
                                  "interval_excludes_zero": ""})
    run09.write_csv(out / "v1_context_contrasts.csv", contrast_rows)

    # Item means: cells (untempered and with v1 decoding) versus v1 sampled answers.
    v1_records = [json.loads(line) for line in Path(args.v1_responses).open(encoding="utf-8")]
    mean_rows = []
    for j, item in enumerate(items):
        row = {"item": item,
               "v1_sampled_mean": float(np.mean([r["scores"][item.upper()] for r in v1_records])),
               "cgss_weighted_mean": float(np.mean([human_mean[(w, item)] for w in wave_names]))}
        for c, cell in enumerate(cells):
            dists = [dict(zip(SCALE, probs[c, p, j])) for p in range(len(profiles))]
            row[f"{cell}_mean"] = float(np.mean([expected(d) for d in dists]))
            row[f"{cell}_mean_v1_decoding"] = float(np.mean([expected(v1_decoded(d)) for d in dists]))
        mean_rows.append(row)
    run09.write_csv(out / "v1_context_item_means.csv", mean_rows)

    metadata = {
        "ledger_sha256": run09.sha256(Path(args.ledger)), "reference_records": n_ref,
        "v1_context_ledger_sha256": run09.sha256(Path(args.v1_context_ledger)), "v1_context_records": n_v1,
        "profiles": len(profiles), "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
        "bootstrap": "profiles resampled with replacement within wave; all cells and items move together",
        "v1_decoding": {"temperature": V1_TEMPERATURE, "top_p": V1_TOP_P},
    }

    # A421 diagnostic: which context accounts for the answers v1 sampled?
    if args.diagnostic:
        diag: dict[str, dict[str, dict[int, float]]] = defaultdict(dict)
        for line in Path(args.diagnostic).open(encoding="utf-8"):
            r = json.loads(line)
            diag[r["profile_id"]][r["context"]] = {int(k): v for k, v in r["distribution"].items()}
        draws: dict[str, list[int]] = defaultdict(list)
        for r in v1_records:
            draws[r["profile_id"]].append(int(r["scores"]["A421"]))
        ids = [p for p in diag if len(diag[p]) == 2]
        observed_means = [np.mean(draws[p]) for p in ids]
        diag_rows = []
        for context in ("open", "closed"):
            decoded = {p: v1_decoded(diag[p][context]) for p in ids}
            loglik = [math.log(max(decoded[p][x], 1e-12)) for p in ids for x in draws[p]]
            diag_rows.append({
                "context": f"thinking {context}", "profiles": len(ids),
                "v1_draws": sum(len(draws[p]) for p in ids),
                "v1_sampled_mean": float(np.mean([x for p in ids for x in draws[p]])),
                "expected_untempered": float(np.mean([expected(diag[p][context]) for p in ids])),
                "expected_v1_decoding": float(np.mean([expected(decoded[p]) for p in ids])),
                "mean_loglik_v1_draws": float(np.mean(loglik)),
                "v1_draws_with_zero_probability": int(sum(decoded[p][x] == 0 for p in ids for x in draws[p])),
                "corr_with_v1_profile_means": float(np.corrcoef(
                    [expected(decoded[p]) for p in ids], observed_means)[0, 1]),
            })
        run09.write_csv(out / "v1_context_a421_diagnostic.csv", diag_rows)
        metadata["diagnostic_sha256"] = run09.sha256(Path(args.diagnostic))
    (out / "v1_context_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("A425 mean |r| by cell:")
    for row in cell_rows:
        print(f"  {row['cell']:10s} {row['a425_mean_abs_correlation']:.3f} "
              f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]  |mean error| {row['mean_abs_mean_error']:.3f}")
    for row in (r for r in contrast_rows if r["metric"] == "a425_mean_abs_correlation"):
        print(f"  {row['contrast']:24s} {row['estimate']:+.3f} [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
