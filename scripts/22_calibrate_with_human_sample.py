#!/usr/bin/env python3
# requires: Python 3.10+, numpy, the survey_llm_eval package
"""Recalibration with a small human sample
(PREREGISTRATION_V2.md, 2026-10-29 evening, item 1).

For each cell and item, the option distributions are recalibrated on the real
answers of n matched respondents, by category offsets (M1) or offsets and an
inverse temperature (M2), and the errors reported earlier are recomputed:
marginal error, between-profile variance, subgroup gradients, and, in the
human-history cells, the history dependence of A425. Only aggregate tables are
written.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run09 = _load("analyze_logprob_audit", "09_analyze_logprob_audit.py")
run17 = _load("subgroup_gradients", "17_subgroup_gradients.py")
run18 = _load("analyze_history_comparisons", "18_analyze_history_comparisons.py")

from survey_llm_eval.factorial import CELLS, item_ids  # noqa: E402

SCALE = np.arange(1, 6, dtype=float)
VALID = {"1", "2", "3", "4", "5"}
HUMAN_CELLS = ("H-A1-B1-C1", "H-A0-B1-C0")
SIZES = (50, 100, 200, None)          # None: all matched respondents
METHODS = ("M1", "M2")
OFFSET_SD, INV_TEMP_SD = 5.0, 1.0
REPORTED = [("public", p) for p in run17.PREDICTORS] + [("household", p) for p in run17.PREDICTORS]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="option_probabilities_greedy.jsonl")
    parser.add_argument("--human-history-ledger", required=True)
    parser.add_argument("--matched", required=True, help="profiles_pilot.csv (matched answers)")
    parser.add_argument("--profiles", required=True, help="profiles_llm_input.csv")
    parser.add_argument("--ml-data", required=True, help="human_ml_input.csv, for weighted CGSS slopes")
    parser.add_argument("--human-dir", default=str(REPO_ROOT / "output" / "v2" / "human_benchmark"))
    parser.add_argument("--human-gradients", default=str(REPO_ROOT / "output" / "metrics_subgroup_gradients.csv"))
    parser.add_argument("--history-dependence", default=str(REPO_ROOT / "output" / "v2" / "exploratory_history_dependence.csv"))
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20261029)
    return parser.parse_args()


def softmax(u: np.ndarray) -> np.ndarray:
    u = u - u.max(axis=-1, keepdims=True)
    e = np.exp(u)
    return e / e.sum(axis=-1, keepdims=True)


def objective(z, y1, b, s):
    """Penalised log-likelihood for a batch of problems: z (B, n, 5), y1 one-hot."""
    u = s[:, None, None] * z + b[:, None, :]
    u = u - u.max(axis=-1, keepdims=True)
    ll = (y1 * (u - np.log(np.exp(u).sum(axis=-1, keepdims=True)))).sum(axis=(1, 2))
    return ll - (b ** 2).sum(axis=1) / (2 * OFFSET_SD ** 2) - (s - 1) ** 2 / (2 * INV_TEMP_SD ** 2)


def fit(z: np.ndarray, y: np.ndarray, method: str, iterations: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Damped Newton for q ∝ exp(s·z + b), b_1 = 0; s fixed at 1 under M1.

    The objective is concave in (b, s), so Newton steps with step halving converge.
    """
    batch, _, k = z.shape
    y1 = np.eye(k)[y]
    b, s = np.zeros((batch, k)), np.ones(batch)
    free_s = method == "M2"
    d = (k - 1) + int(free_s)
    current = objective(z, y1, b, s)
    for _ in range(iterations):
        q = softmax(s[:, None, None] * z + b[:, None, :])
        grad = np.zeros((batch, d))
        hess = np.zeros((batch, d, d))
        grad[:, : k - 1] = (y1 - q).sum(axis=1)[:, 1:] - b[:, 1:] / OFFSET_SD ** 2
        cov = np.einsum("bnk,kl->bnkl", q, np.eye(k)) - np.einsum("bnk,bnl->bnkl", q, q)
        hess[:, : k - 1, : k - 1] = -cov.sum(axis=1)[:, 1:, 1:] - np.eye(k - 1) / OFFSET_SD ** 2
        if free_s:
            zbar = (q * z).sum(axis=-1, keepdims=True)
            dev = z - zbar
            grad[:, -1] = ((y1 * z).sum(axis=-1) - zbar[..., 0]).sum(axis=1) - (s - 1) / INV_TEMP_SD ** 2
            cross = -(q * dev).sum(axis=1)[:, 1:]
            hess[:, : k - 1, -1] = cross
            hess[:, -1, : k - 1] = cross
            hess[:, -1, -1] = -(q * dev ** 2).sum(axis=(1, 2)) - 1 / INV_TEMP_SD ** 2
        step = -np.linalg.solve(hess, grad[..., None])[..., 0]
        scale = np.ones(batch)
        for _ in range(30):
            nb = b.copy()
            nb[:, 1:] = b[:, 1:] + scale[:, None] * step[:, : k - 1]
            ns = s + scale * step[:, -1] if free_s else s
            new = objective(z, y1, nb, ns)
            worse = new < current - 1e-12
            if not worse.any():
                break
            scale = np.where(worse, scale / 2, scale)
        b, s, change = nb, ns, np.abs(new - current).max()
        current = new
        if change < 1e-10:
            break
    return b, s


def apply(z: np.ndarray, b: np.ndarray, s: np.ndarray) -> np.ndarray:
    """z (..., profile, 5) with parameters (..., 5) and (...)."""
    return softmax(s[..., None, None] * z + b[..., None, :])


def outcomes(probs, h425, idx, waves, human, pinvs, history_mean):
    """probs (cell, profile, item, option) and h425 (cell, profile, option), on resampled idx."""
    human_mean, human_var, human_marginal = human
    expected = (probs * SCALE).sum(axis=-1)                    # cell, profile, item
    abs_err, tv, between = [], [], []
    for w, rows in zip(waves, idx):
        mix = probs[:, rows].mean(axis=1)                      # cell, item, option
        mean = expected[:, rows].mean(axis=1)                  # cell, item
        abs_err.append(np.abs(mean - human_mean[w]))
        tv.append(0.5 * np.abs(mix - human_marginal[w]).sum(axis=-1))
        between.append(expected[:, rows].var(axis=1) / human_var[w])
    summary = np.stack([np.mean(abs_err, axis=(0, 2)), np.mean(tv, axis=(0, 2)),
                        np.mean(between, axis=(0, 2))], axis=1)  # cell, 3
    eq = expected.copy()
    eq[..., :4] = 6 - eq[..., :4]
    y = np.stack([eq[..., :4].mean(axis=-1), eq[..., 4]], axis=-1)     # cell, profile, 2
    beta = np.mean([np.einsum("kn,cno->cok", pinv, y[:, rows])[..., 1:]
                    for pinv, rows in zip(pinvs, idx)], axis=0)        # cell, outcome, predictor
    a425 = (h425 * SCALE).sum(axis=-1)
    _, slope = run18.dependence(a425, history_mean, idx)
    return summary, beta, slope


def read_human(human_dir: Path, waves: list[str], items: list[str]):
    moments = run09.read_csv(human_dir / "human_moments.csv")
    marginals = run09.read_csv(human_dir / "human_marginals.csv")
    mean = {w: np.array([next(float(r["mean"]) for r in moments if r["wave"] == w and r["item"] == i)
                         for i in items]) for w in waves}
    var = {w: np.array([next(float(r["variance"]) for r in moments if r["wave"] == w and r["item"] == i)
                        for i in items]) for w in waves}
    marg = {w: np.zeros((len(items), 5)) for w in waves}
    for r in marginals:
        if r["wave"] in marg and r["item"] in items:
            marg[r["wave"]][items.index(r["item"]), int(r["option"]) - 1] = float(r["probability"])
    return mean, var, marg


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    items = [i.lower() for i in item_ids()]
    factorial = [c.name for c in CELLS]

    _, profiles_all, probs_all, _ = run09.load_ledger(Path(args.ledger), factorial)
    matched = {r["profile_id"]: r for r in run09.read_csv(Path(args.matched))}
    keep = [p for p, (_, pid) in enumerate(profiles_all)
            if all(matched[pid][i] in VALID for i in items)]
    profiles = [profiles_all[p] for p in keep]
    probs = probs_all[:, keep]                                          # cell, profile, item, option
    position = {pid: i for i, (_, pid) in enumerate(profiles)}

    h425 = np.full((len(HUMAN_CELLS), len(profiles), 5), np.nan)
    for line in Path(args.human_history_ledger).open(encoding="utf-8"):
        r = json.loads(line)
        if r["item"] == "A425" and r["cell"] in HUMAN_CELLS and r["profile_id"] in position:
            h425[HUMAN_CELLS.index(r["cell"]), position[r["profile_id"]]] = \
                [r["probabilities"][str(k)] for k in range(1, 6)]
    if np.isnan(h425).any():
        raise SystemExit("Human-history ledger is incomplete for A425")

    answers = np.array([[int(matched[pid][i]) - 1 for i in items] for _, pid in profiles])
    history_mean = (6 - (answers[:, :4] + 1)).mean(axis=1).astype(float)
    attrs = {r["profile_id"]: r for r in run09.read_csv(Path(args.profiles))}
    attributes = np.array([[float(attrs[pid][k]) for k in ("female", "educ_years", "age", "urban_residence")]
                           for _, pid in profiles])
    waves_of = np.array([w for w, _ in profiles])
    waves = sorted(set(waves_of))
    wave_index = [np.flatnonzero(waves_of == w) for w in waves]
    human = read_human(Path(args.human_dir), waves, items)

    gradients = {}
    for r in run09.read_csv(Path(args.human_gradients)):
        if r["condition"] == "neutral_verbal":
            gradients[(r["wave"], r["outcome"], r["predictor"])] = (float(r["human_estimate"]),
                                                                    float(r["human_std_error"]))
    preds = list(run17.PREDICTORS)
    h_est = np.array([[[gradients[(w, o, p)][0] for p in preds] for o in ("public", "household")] for w in waves])
    h_se = np.array([[[gradients[(w, o, p)][1] for p in preds] for o in ("public", "household")] for w in waves])
    cgss_slope = float(np.mean([r["slope"] for r in run18.weighted_cgss(Path(args.ml_data))]))
    hgb_slope = next(float(r["slope"]) for r in run09.read_csv(Path(args.history_dependence))
                     if r["source"] == "HGB with respondent history")

    # Problems: (cell, item) for the factorial cells, then A425 in the human-history cells.
    logp = np.log(np.clip(probs, 1e-12, None))                         # cell, profile, item, option
    h_logp = np.log(np.clip(h425, 1e-12, None))
    z_all = np.concatenate([logp.transpose(0, 2, 1, 3).reshape(-1, len(profiles), 5), h_logp])
    y_all = np.concatenate([np.tile(answers.T, (len(factorial), 1)),
                            np.tile(answers[:, 4], (len(HUMAN_CELLS), 1))])
    n_fact = len(factorial) * len(items)

    configs = [("uncalibrated", 0)] + [(m, n if n else len(profiles)) for m in METHODS for n in SIZES]
    store = {c: {"summary": [], "beta": [], "slope": [], "s": []} for c in configs}
    for rep in range(args.replicates):
        idx = [rng.choice(i, size=len(i), replace=True) for i in wave_index]
        pool = np.concatenate(idx)
        pinvs = [np.linalg.pinv(run17.design(attributes[rows])) for rows in idx]
        for method, n in configs:
            if method == "uncalibrated":
                cal, hcal, s = probs, h425, np.ones(len(z_all))
            else:
                sample = pool if n == len(profiles) else pool[rng.choice(len(pool), size=n, replace=False)]
                b, s = fit(z_all[:, sample], y_all[:, sample], method)
                q = apply(z_all, b, s)                                   # problem, profile, option
                cal = q[:n_fact].reshape(len(factorial), len(items), len(profiles), 5).transpose(0, 2, 1, 3)
                hcal = q[n_fact:]
            summary, beta, slope = outcomes(cal, hcal, idx, waves, human, pinvs, history_mean)
            store[(method, n)]["summary"].append(summary)
            store[(method, n)]["beta"].append(beta)
            store[(method, n)]["slope"].append(slope)
            store[(method, n)]["s"].append(s)
        if (rep + 1) % 100 == 0:
            print(f"replicate {rep + 1}/{args.replicates}", flush=True)

    boot_h = rng.normal(h_est, h_se, size=(args.replicates,) + h_est.shape).mean(axis=1)  # rep, outcome, predictor
    mean_h = h_est.mean(axis=0)

    def stats(values) -> dict:
        lo, med, hi = np.percentile(values, [2.5, 50, 97.5])
        return {"median": float(med), "ci95_low": float(lo), "ci95_high": float(hi)}

    def labelled(prefix: str, values) -> dict:
        return {f"{prefix}_{k}": v for k, v in stats(values).items()}

    summary_rows, gradient_rows, slope_rows, param_rows = [], [], [], []
    for (method, n), data in store.items():
        summary = np.array(data["summary"])                              # rep, cell, 3
        beta = np.array(data["beta"])                                    # rep, cell, outcome, predictor
        slope = np.array(data["slope"])                                  # rep, cell
        s = np.array(data["s"])                                          # rep, problem
        for c, cell in enumerate(factorial):
            summary_rows.append({"cell": cell, "method": method, "n": n,
                                 **labelled("mean_abs_mean_error", summary[:, c, 0]),
                                 **labelled("mean_total_variation", summary[:, c, 1]),
                                 **labelled("between_variance_ratio", summary[:, c, 2])})
            for o, outcome in enumerate(("public", "household")):
                for p, predictor in enumerate(preds):
                    gradient_rows.append({"cell": cell, "method": method, "n": n, "outcome": outcome,
                                          "predictor": predictor, **labelled("model", beta[:, c, o, p]),
                                          "cgss_estimate": float(mean_h[o, p]),
                                          **labelled("difference", beta[:, c, o, p] - boot_h[:, o, p])})
            e = preds.index("educ_years")
            gap = beta[:, c, 0, e] - beta[:, c, 1, e]
            gap_h = boot_h[:, 0, e] - boot_h[:, 1, e]
            gradient_rows.append({"cell": cell, "method": method, "n": n, "outcome": "public_minus_household",
                                  "predictor": "educ_years", **labelled("model", gap),
                                  "cgss_estimate": float(mean_h[0, e] - mean_h[1, e]),
                                  **labelled("difference", gap - gap_h)})
            if method != "uncalibrated":
                for j, item in enumerate(items):
                    param_rows.append({"cell": cell, "item": item, "method": method, "n": n,
                                       **labelled("inverse_temperature", s[:, c * len(items) + j])})
        for h, cell in enumerate(HUMAN_CELLS):
            slope_rows.append({"cell": cell, "method": method, "n": n, **labelled("slope", slope[:, h]),
                               "cgss_weighted_slope": cgss_slope, "hgb_with_history_slope": hgb_slope,
                               **labelled("minus_cgss", slope[:, h] - cgss_slope),
                               **labelled("minus_hgb", slope[:, h] - hgb_slope)})
            if method != "uncalibrated":
                param_rows.append({"cell": cell, "item": "a425", "method": method, "n": n,
                                   **labelled("inverse_temperature", s[:, n_fact + h])})

    run09.write_csv(out / "exploratory_calibration_summary.csv", summary_rows)
    run09.write_csv(out / "exploratory_calibration_gradients.csv", gradient_rows)
    run09.write_csv(out / "exploratory_calibration_history_slope.csv", slope_rows)
    run09.write_csv(out / "exploratory_calibration_parameters.csv", param_rows)
    meta = {"profiles": len(profiles), "excluded_missing_answers": len(profiles_all) - len(profiles),
            "replicates": args.replicates, "seed": args.seed, "sizes": [n for _, n in configs[1:5]],
            "methods": {"M1": "q ∝ p·exp(b_k), b_1 = 0", "M2": "q ∝ p^s·exp(b_k), b_1 = 0"},
            "penalty": f"sum b_k^2/(2*{OFFSET_SD}^2) + (s-1)^2/(2*{INV_TEMP_SD}^2)",
            "cgss_weighted_slope": cgss_slope, "hgb_with_history_slope": hgb_slope,
            "ledger_sha256": run09.sha256(Path(args.ledger)),
            "human_history_ledger_sha256": run09.sha256(Path(args.human_history_ledger)),
            "note": "inverse-temperature table is descriptive and was not pre-specified"}
    (out / "exploratory_calibration_metadata.json").write_text(json.dumps(meta, indent=2) + "\n",
                                                               encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
