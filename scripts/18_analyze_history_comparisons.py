#!/usr/bin/env python3
# requires: Python 3.10+, numpy, the survey_llm_eval package
"""Human-history cells and survey-trained comparators
(PREREGISTRATION_V2.md, 2026-10-29, items 1 and 3).

- 1(a) matched calibration on A422-A425 with the respondent's own history,
  the model's modal history, and no history, for the model and for HGB;
- 1(b) history dependence of A425: correlation and slope of A425 on the mean
  of the respondent's A421-A424 answers (egalitarian coding);
- 3(a) response-level coherence of five draws from the HGB distributions.

All quantities are computed for the matched profiles with valid answers on all
five items. Only aggregate tables are written.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, file: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


run09 = _load("analyze_logprob_audit", "09_analyze_logprob_audit.py")
run14 = _load("analyze_sampled_history", "14_analyze_sampled_history.py")

from survey_llm_eval.factorial import item_ids  # noqa: E402
from survey_llm_eval.probabilities import log_score, ranked_probability_score  # noqa: E402

SCALE = run09.SCALE
VALID = {"1", "2", "3", "4", "5"}
# human-history cell -> (same prompt with modal history, matching cell without history)
PAIRS = {"H-A1-B1-C1": ("A1-B1-C1", "A1-B0-C1"), "H-A0-B1-C0": ("A0-B1-C0", "A0-B0-C0")}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="option_probabilities_greedy.jsonl")
    parser.add_argument("--human-history-ledger", required=True)
    parser.add_argument("--matched", required=True, help="profiles_pilot.csv (matched answers)")
    parser.add_argument("--hgb-predictions", required=True, help="first-audit ml/output/predictions.csv")
    parser.add_argument("--hgb-history", required=True, help="hgb_with_history_matched.csv")
    parser.add_argument("--ml-data", required=True, help="human_ml_input.csv, for weighted CGSS slopes")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--human-cells", nargs="+", default=list(PAIRS),
                        help="human-history cells to analyse (default: both)")
    parser.add_argument("--paths", type=int, default=5)
    parser.add_argument("--draw-seed", type=int, default=20261013)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def raw_from_eq(eq: list[float], item_index: int) -> np.ndarray:
    """HGB distributions are in egalitarian coding; A421-A424 are reversed."""
    eq = np.asarray(eq, dtype=float)
    return eq[::-1].copy() if item_index < 4 else eq


def load_hgb(path: Path, model: str | None, pids: set[str], items: list[str]) -> dict:
    out: dict[tuple[str, int], np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for r in csv.DictReader(handle):
            if model is not None and (r.get("split") != "oof" or r.get("model") != model):
                continue
            if r["profile_id"] not in pids:
                continue
            j = items.index(r["item"].removeprefix("eq_").upper())
            out[(r["profile_id"], j)] = raw_from_eq([float(r[f"p{k}"]) for k in SCALE], j)
    return out


def calibration(dist: np.ndarray, observed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(profile, item, option) and (profile, item) -> per-profile, per-item log score and RPS."""
    n, m, _ = dist.shape
    ls = np.empty((n, m))
    rps = np.empty((n, m))
    for p in range(n):
        for j in range(m):
            d = {k: float(v) for k, v in zip(SCALE, dist[p, j])}
            ls[p, j] = log_score(d, int(observed[p, j]))
            rps[p, j] = ranked_probability_score(d, int(observed[p, j]), SCALE)
    return ls, rps


def dependence(a425: np.ndarray, history_mean: np.ndarray, wave_index) -> tuple[np.ndarray, np.ndarray]:
    """(source, profile) -> wave-averaged correlation and slope of A425 on the history mean."""
    corr, slope = [], []
    for idx in wave_index:
        y = a425[:, idx] - a425[:, idx].mean(axis=1, keepdims=True)
        x = history_mean[idx] - history_mean[idx].mean()
        cov = (y * x).sum(axis=1)
        corr.append(cov / np.sqrt((y ** 2).sum(axis=1) * (x ** 2).sum()))
        slope.append(cov / (x ** 2).sum())
    return np.mean(corr, axis=0), np.mean(slope, axis=0)


def weighted_cgss(path: Path) -> list[dict]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    out = []
    for wave in sorted({r["wave"] for r in rows}):
        keep = [r for r in rows if r["wave"] == wave
                and all(r[f"eq_a42{k}"] not in ("", "NA") for k in range(1, 6))]
        w = np.array([float(r["analysis_weight"]) for r in keep])
        x = np.array([np.mean([float(r[f"eq_a42{k}"]) for k in range(1, 5)]) for r in keep])
        y = np.array([float(r["eq_a425"]) for r in keep])
        xm, ym = np.average(x, weights=w), np.average(y, weights=w)
        cov = np.average((x - xm) * (y - ym), weights=w)
        vx, vy = np.average((x - xm) ** 2, weights=w), np.average((y - ym) ** 2, weights=w)
        out.append({"wave": wave, "n": len(keep), "correlation": cov / np.sqrt(vx * vy), "slope": cov / vx})
    return out


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    items = list(item_ids())
    unknown = sorted(set(args.human_cells) - set(PAIRS))
    if unknown:
        raise SystemExit(f"Unknown human-history cells: {unknown}")
    pairs = {name: PAIRS[name] for name in PAIRS if name in args.human_cells}

    cells = sorted({c for pair in pairs.values() for c in pair})
    _, all_profiles, probs, _ = run09.load_ledger(Path(args.ledger), cells)
    matched = {r["profile_id"]: r for r in run09.read_csv(Path(args.matched))}
    keep = [i for i, (_, pid) in enumerate(all_profiles)
            if all(matched[pid][i_.lower()] in VALID for i_ in items)]
    profiles = [all_profiles[i] for i in keep]
    probs = probs[:, keep]
    pids = [pid for _, pid in profiles]
    position = {pid: i for i, pid in enumerate(pids)}
    waves = np.array([w for w, _ in profiles])
    wave_names = sorted(set(waves))
    wave_index = [np.flatnonzero(waves == w) for w in wave_names]
    observed = np.array([[int(matched[pid][i.lower()]) for i in items] for pid in pids])
    history_mean = (6 - observed[:, :4]).mean(axis=1)       # egalitarian coding

    # Human-history distributions; A421 comes from the matching modal-history cell.
    h_names = list(pairs)
    h_probs = np.full((len(h_names), len(pids), len(items), len(SCALE)), np.nan)
    for name in h_names:
        h_probs[h_names.index(name), :, 0] = probs[cells.index(pairs[name][0]), :, 0]
    h_records = 0
    for line in Path(args.human_history_ledger).open(encoding="utf-8"):
        r = json.loads(line)
        if r["profile_id"] in position and r["cell"] in h_names:
            p = position[r["profile_id"]]
            if r["history"] != [int(a) for a in observed[p, : r["target_index"]]]:
                raise SystemExit(f"history is not the respondent's answers: {r['profile_id']}")
            h_probs[h_names.index(r["cell"]), p, items.index(r["item"])] = [
                r["probabilities"][str(k)] for k in SCALE]
            h_records += 1
    if np.isnan(h_probs).any():
        raise SystemExit("human-history ledger is incomplete")

    hgb_none = load_hgb(Path(args.hgb_predictions), "hist_gradient_boosting", set(pids), items)
    hgb_hist = load_hgb(Path(args.hgb_history), None, set(pids), items)
    hgb = np.array([[hgb_none[(pid, j)] for j in range(len(items))] for pid in pids])
    hgb_h = hgb.copy()
    for p, pid in enumerate(pids):
        for j in range(1, len(items)):
            hgb_h[p, j] = hgb_hist[(pid, j)]

    # Sources for 1(a) and 1(b): (label, distributions over profiles x items x options).
    sources = []
    for name, (modal_cell, plain_cell) in pairs.items():
        sources += [(f"{name} (respondent history)", h_probs[h_names.index(name)]),
                    (f"{modal_cell} (model's modal history)", probs[cells.index(modal_cell)]),
                    (f"{plain_cell} (no history)", probs[cells.index(plain_cell)])]
    sources += [("HGB with respondent history", hgb_h), ("HGB without history", hgb)]
    labels = [s[0] for s in sources]
    later = slice(1, len(items))                            # A422-A425
    scores = [calibration(d[:, later], observed[:, later]) for _, d in sources]
    ls = np.stack([s[0] for s in scores])                   # source, profile, item
    rps = np.stack([s[1] for s in scores])
    a425 = np.stack([(d[:, 4] * np.array(SCALE, dtype=float)).sum(axis=1) for _, d in sources])
    a425 = np.vstack([a425, observed[:, 4][None, :].astype(float)])
    dep_labels = labels + ["Matched respondents (observed answers)"]

    def point(idx_by_wave):
        idx = np.concatenate(idx_by_wave)
        corr, slope = dependence(a425, history_mean, idx_by_wave)
        return ls[:, idx].mean(axis=1), rps[:, idx].mean(axis=1), corr, slope

    ls_hat, rps_hat, corr_hat, slope_hat = point(wave_index)
    rng = np.random.default_rng(args.seed)
    boot = {"ls": [], "rps": [], "corr": [], "slope": []}
    for _ in range(args.replicates):
        idx = [rng.choice(i, size=len(i), replace=True) for i in wave_index]
        for key, value in zip(boot, point(idx)):
            boot[key].append(value)
    boot = {k: np.array(v) for k, v in boot.items()}

    def ci(values):
        low, high = np.percentile(values, [2.5, 97.5], axis=0)
        return low, high

    item_names = [i.lower() for i in items[1:]]
    rows = []
    for s, label in enumerate(labels):
        row = {"source": label}
        for j, item in enumerate(item_names):
            row[f"log_score_{item}"] = float(ls_hat[s, j])
            row[f"rps_{item}"] = float(rps_hat[s, j])
        row["log_score_a422_a425"] = float(ls_hat[s].mean())
        row["rps_a422_a425"] = float(rps_hat[s].mean())
        rows.append(row)
    run09.write_csv(out / "exploratory_human_history_calibration.csv", rows)

    contrasts = []
    for name, (modal_cell, plain_cell) in pairs.items():
        h = labels.index(f"{name} (respondent history)")
        contrasts += [(h, labels.index(f"{modal_cell} (model's modal history)")),
                      (h, labels.index(f"{plain_cell} (no history)"))]
    contrasts.append((labels.index("HGB with respondent history"), labels.index("HGB without history")))
    contrast_rows = []
    for a, b in contrasts:
        for metric, hat, draws in (("log_score", ls_hat, boot["ls"]), ("rps", rps_hat, boot["rps"])):
            for j, item in enumerate(item_names + ["a422_a425"]):
                if item == "a422_a425":
                    est, diff = hat[a].mean() - hat[b].mean(), draws[:, a].mean(1) - draws[:, b].mean(1)
                else:
                    est, diff = hat[a, j] - hat[b, j], draws[:, a, j] - draws[:, b, j]
                low, high = ci(diff)
                contrast_rows.append({"contrast": f"{labels[a]} minus {labels[b]}", "metric": metric,
                                      "item": item, "estimate": float(est), "ci95_low": float(low),
                                      "ci95_high": float(high),
                                      "interval_excludes_zero": bool(low > 0 or high < 0)})
    run09.write_csv(out / "exploratory_human_history_contrasts.csv", contrast_rows)

    cgss = weighted_cgss(Path(args.ml_data))
    dep_rows = []
    c_low, c_high = ci(boot["corr"])
    s_low, s_high = ci(boot["slope"])
    for s, label in enumerate(dep_labels):
        dep_rows.append({"source": label, "correlation": float(corr_hat[s]),
                         "correlation_ci95_low": float(c_low[s]), "correlation_ci95_high": float(c_high[s]),
                         "slope": float(slope_hat[s]), "slope_ci95_low": float(s_low[s]),
                         "slope_ci95_high": float(s_high[s])})
    dep_rows.append({"source": "CGSS weighted, full sample (observed answers)",
                     "correlation": float(np.mean([r["correlation"] for r in cgss])),
                     "correlation_ci95_low": "", "correlation_ci95_high": "",
                     "slope": float(np.mean([r["slope"] for r in cgss])),
                     "slope_ci95_low": "", "slope_ci95_high": ""})
    hgb_h_index = labels.index("HGB with respondent history")
    for name in h_names:
        s = labels.index(f"{name} (respondent history)")
        for metric, draws, hat in (("slope", boot["slope"], slope_hat), ("correlation", boot["corr"], corr_hat)):
            low, high = ci(draws[:, s] - draws[:, hgb_h_index])
            dep_rows.append({"source": f"{name} minus HGB with respondent history ({metric})",
                             metric: float(hat[s] - hat[hgb_h_index]),
                             f"{metric}_ci95_low": float(low), f"{metric}_ci95_high": float(high)})
    fields = list(dep_rows[0])
    run09.write_csv(out / "exploratory_history_dependence.csv",
                    [{k: r.get(k, "") for k in fields} for r in dep_rows])

    # 3(a): response-level coherence of HGB draws (outcome R1 of the 2026-10-28 entry).
    _, full_profiles, _, _ = run09.load_ledger(Path(args.ledger), ["A0-B0-C0"])
    full_pids = [pid for _, pid in full_profiles]
    hgb_full = load_hgb(Path(args.hgb_predictions), "hist_gradient_boosting", set(full_pids), items)
    full_waves = np.array([w for w, _ in full_profiles])
    full_index = [np.flatnonzero(full_waves == w) for w in wave_names]
    draws = np.zeros((1, len(full_pids), args.paths, len(items)), dtype=int)
    missing = 0
    for p, pid in enumerate(full_pids):
        for k in range(args.paths):
            rng_draw = random.Random(f"{args.draw_seed}|{pid}|HGB|{k}")
            for j in range(len(items)):
                dist = hgb_full.get((pid, j))
                if dist is None:                            # respondent missing this item
                    missing += 1
                    dist = np.full(len(SCALE), 0.2)
                draws[0, p, k, j] = run14.draw(rng_draw, {o: float(v) for o, v in zip(SCALE, dist)})
    r1, r1_wave, undefined = run14.response_level(draws, full_index)
    boot_r1 = np.empty(args.replicates)
    rng = np.random.default_rng(args.seed)
    for b in range(args.replicates):
        idx = [rng.choice(i, size=len(i), replace=True) for i in full_index]
        boot_r1[b] = run14.response_level(draws, idx)[0][0]
    low, high = np.percentile(boot_r1, [2.5, 97.5])
    run09.write_csv(out / "exploratory_hgb_response_level.csv", [{
        "source": "HGB out-of-fold distributions, five draws per profile",
        "response_level": float(r1[0]), "ci95_low": float(low), "ci95_high": float(high),
        **{f"wave_{w}": float(r1_wave[0, k]) for k, w in enumerate(wave_names)},
        "undefined_correlations": undefined, "profiles_without_hgb_prediction_items": missing}])

    metadata = {"profiles": len(pids), "profiles_excluded_missing_answers": len(all_profiles) - len(pids),
                "human_history_cells": pairs, "human_history_records": h_records, "cgss_weighted_by_wave": cgss,
                "coding": "calibration on the original scale; dependence in egalitarian coding",
                "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
                "hgb_draw_seed": args.draw_seed}
    (out / "exploratory_human_history_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("Mean log score / RPS on A422-A425 (lower is better):")
    for s, label in enumerate(labels):
        print(f"  {label:42s} {ls_hat[s].mean():.3f} / {rps_hat[s].mean():.3f}   A425 {ls_hat[s, 3]:.3f}")
    print("A425 on the respondent's A421-A424 mean: correlation / slope")
    for s, label in enumerate(dep_labels):
        print(f"  {label:42s} {corr_hat[s]:+.3f} / {slope_hat[s]:+.3f}")
    print(f"  CGSS weighted: {np.mean([r['correlation'] for r in cgss]):+.3f} / "
          f"{np.mean([r['slope'] for r in cgss]):+.3f}")
    print(f"HGB response-level coherence: {r1[0]:.3f} [{low:.3f}, {high:.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
