#!/usr/bin/env python3
# requires: Python 3.10+, numpy, the survey_llm_eval package
"""Exploratory subgroup gradients (PREREGISTRATION_V2.md, 2026-10-29, item 2).

For each factorial cell and the V1-T1/V1-T0 cells, regresses the
egalitarian-coded expected score of each profile on female, years of
education, age in decades, and urban residence within wave (the first audit's
specification) and compares the coefficients with the first audit's weighted
full-CGSS estimates. Only aggregate tables are written.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "analyze_logprob_audit", REPO_ROOT / "scripts" / "09_analyze_logprob_audit.py")
run09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run09)

from survey_llm_eval.factorial import CELLS  # noqa: E402

PREDICTORS = ("female", "educ_years", "age_decade", "urban_residence")
OUTCOMES = ("public", "household", "a421", "a422", "a423", "a424", "a425")
V1_CELLS = ["V1-T1", "V1-T0"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="option_probabilities_greedy.jsonl")
    parser.add_argument("--v1-context-ledger", required=True)
    parser.add_argument("--profiles", required=True, help="profiles_llm_input.csv")
    parser.add_argument("--human-gradients", default=str(REPO_ROOT / "output" / "metrics_subgroup_gradients.csv"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def egalitarian_outcomes(probs: np.ndarray) -> np.ndarray:
    """(cell, profile, item, option) -> (cell, profile, outcome) in OUTCOMES order."""
    expected = probs @ np.arange(1, 6, dtype=float)
    eq = expected.copy()
    eq[..., :4] = 6 - eq[..., :4]
    public = eq[..., :4].mean(axis=-1, keepdims=True)
    return np.concatenate([public, eq[..., 4:5], eq], axis=-1)


def design(attributes: np.ndarray) -> np.ndarray:
    female, educ, age, urban = attributes.T
    age_decade = (age - age.mean()) / 10
    return np.column_stack([np.ones(len(age)), female, educ, age_decade, urban])


def coefficients(y: np.ndarray, attributes: np.ndarray, wave_index) -> np.ndarray:
    """(cell, profile, outcome) -> (cell, wave, outcome, predictor)."""
    per_wave = []
    for idx in wave_index:
        pinv = np.linalg.pinv(design(attributes[idx]))           # 5 x n
        beta = np.einsum("kn,cno->cok", pinv, y[:, idx, :])      # cell, outcome, 5
        per_wave.append(beta[..., 1:])
    return np.stack(per_wave, axis=1)


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    factorial = [c.name for c in CELLS]
    records, profiles, probs, _ = run09.load_ledger(Path(args.ledger), factorial)
    _, _, v1_probs, _ = run09.load_ledger(Path(args.v1_context_ledger), V1_CELLS, profiles)
    cells = factorial + V1_CELLS
    y = egalitarian_outcomes(np.concatenate([probs, v1_probs], axis=0))

    with Path(args.profiles).open(newline="", encoding="utf-8") as handle:
        attrs = {r["profile_id"]: r for r in csv.DictReader(handle)}
    attributes = np.array([[float(attrs[pid]["female"]), float(attrs[pid]["educ_years"]),
                            float(attrs[pid]["age"]), float(attrs[pid]["urban_residence"])]
                           for _, pid in profiles])
    waves = np.array([w for w, _ in profiles])
    wave_names = sorted(set(waves))
    wave_index = [np.flatnonzero(waves == w) for w in wave_names]

    human = {}
    v1 = {}
    for r in run09.read_csv(Path(args.human_gradients)):
        key = (r["wave"], r["outcome"], r["predictor"])
        human[key] = (float(r["human_estimate"]), float(r["human_std_error"]))
        if r["condition"] == "neutral_verbal":
            v1[key] = float(r["llm_estimate"])
    h_est = np.array([[[human[(w, o, p)][0] for p in PREDICTORS] for o in OUTCOMES] for w in wave_names])
    h_se = np.array([[[human[(w, o, p)][1] for p in PREDICTORS] for o in OUTCOMES] for w in wave_names])

    beta = coefficients(y, attributes, wave_index)            # cell, wave, outcome, predictor
    rng = np.random.default_rng(args.seed)
    boot = np.empty((args.replicates,) + beta.shape)
    boot_human = np.empty((args.replicates,) + h_est.shape)
    for b in range(args.replicates):
        idx = [rng.choice(i, size=len(i), replace=True) for i in wave_index]
        boot[b] = coefficients(y, attributes, idx)
        boot_human[b] = rng.normal(h_est, h_se)

    def ci(values: np.ndarray) -> tuple[float, float]:
        low, high = np.percentile(values, [2.5, 97.5])
        return float(low), float(high)

    rows = []
    for c, cell in enumerate(cells):
        for w, wave in enumerate(wave_names):
            for o, outcome in enumerate(OUTCOMES):
                for p, predictor in enumerate(PREDICTORS):
                    low, high = ci(boot[:, c, w, o, p])
                    rows.append({"cell": cell, "wave": wave, "outcome": outcome, "predictor": predictor,
                                 "model_estimate": float(beta[c, w, o, p]),
                                 "model_ci95_low": low, "model_ci95_high": high,
                                 "human_estimate": float(h_est[w, o, p]),
                                 "human_std_error": float(h_se[w, o, p])})
    run09.write_csv(out / "exploratory_subgroup_gradients_by_wave.csv", rows)

    # Averaged over waves, with intervals for the model-minus-CGSS difference.
    mean_beta, mean_boot = beta.mean(axis=1), boot.mean(axis=2)
    mean_h, mean_boot_h = h_est.mean(axis=0), boot_human.mean(axis=1)
    summary = []
    for o, outcome in enumerate(OUTCOMES):
        for p, predictor in enumerate(PREDICTORS):
            v1_mean = float(np.mean([v1[(w, outcome, predictor)] for w in wave_names]))
            for c, cell in enumerate(cells):
                diff = mean_boot[:, c, o, p] - mean_boot_h[:, o, p]
                low, high = ci(diff)
                m_low, m_high = ci(mean_boot[:, c, o, p])
                summary.append({
                    "cell": cell, "outcome": outcome, "predictor": predictor,
                    "model_estimate": float(mean_beta[c, o, p]),
                    "model_ci95_low": m_low, "model_ci95_high": m_high,
                    "human_estimate": float(mean_h[o, p]),
                    "difference": float(mean_beta[c, o, p] - mean_h[o, p]),
                    "difference_ci95_low": low, "difference_ci95_high": high,
                    "difference_excludes_zero": bool(low > 0 or high < 0),
                    "ratio_to_human": float(mean_beta[c, o, p] / mean_h[o, p]),
                    "v1_sampled_estimate": v1_mean,
                })
    run09.write_csv(out / "exploratory_subgroup_gradients.csv", summary)

    # Public-minus-household gap in the female and education coefficients.
    pub, hh = OUTCOMES.index("public"), OUTCOMES.index("household")
    gaps = []
    for predictor in ("female", "educ_years"):
        p = PREDICTORS.index(predictor)
        human_gap = float(mean_h[pub, p] - mean_h[hh, p])
        v1_gap = float(np.mean([v1[(w, "public", predictor)] - v1[(w, "household", predictor)]
                                for w in wave_names]))
        for c, cell in enumerate(cells):
            model_gap = mean_beta[c, pub, p] - mean_beta[c, hh, p]
            boot_gap = mean_boot[:, c, pub, p] - mean_boot[:, c, hh, p]
            boot_human_gap = mean_boot_h[:, pub, p] - mean_boot_h[:, hh, p]
            low, high = ci(boot_gap)
            d_low, d_high = ci(boot_gap - boot_human_gap)
            gaps.append({"cell": cell, "predictor": predictor,
                         "model_public_minus_household": float(model_gap),
                         "model_ci95_low": low, "model_ci95_high": high,
                         "human_public_minus_household": human_gap,
                         "difference": float(model_gap - human_gap),
                         "difference_ci95_low": d_low, "difference_ci95_high": d_high,
                         "v1_sampled_public_minus_household": v1_gap})
    run09.write_csv(out / "exploratory_subgroup_public_household.csv", gaps)

    metadata = {"ledger_records": records, "profiles": len(profiles), "cells": cells,
                "specification": "OLS within wave on female, educ_years, age_decade (centred within wave), "
                                 "urban_residence; egalitarian-coded expected scores; unweighted profiles",
                "human_benchmark": "first audit, weighted full CGSS (output/metrics_subgroup_gradients.csv)",
                "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
                "bootstrap": "profiles resampled within wave; CGSS coefficients drawn from normal(estimate, se)"}
    (out / "exploratory_subgroup_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Wave-averaged coefficients, model vs CGSS (public | household):")
    for predictor in ("female", "educ_years", "urban_residence"):
        print(f"  {predictor}: CGSS {mean_h[pub, PREDICTORS.index(predictor)]:+.3f} | "
              f"{mean_h[hh, PREDICTORS.index(predictor)]:+.3f}")
        for c, cell in enumerate(cells):
            p = PREDICTORS.index(predictor)
            print(f"    {cell:9s} {mean_beta[c, pub, p]:+.3f} | {mean_beta[c, hh, p]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
