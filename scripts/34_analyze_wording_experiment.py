#!/usr/bin/env python3
# requires: Python 3.10+, numpy
"""Wording experiment (PREREGISTRATION_V2.md, 2026-11-22).

Five items are reworded (`WORDING_VARIANTS` in src/survey_llm_eval/factorial.py)
and scored by Qwen3.5-9B in the two cells without battery listing or history,
A0-B0-C0 and A0-B0-C1, where the prompt holds the target item alone. Each
reworded item is correlated, across profiles, with the original scores of the
other items of its battery from the main runs; the same quantity for the
original wording is the comparison. Indicators are computed per cell and
averaged over the two cells; A421-A425 correlations are computed within wave
and averaged over waves. Profile-cluster bootstrap; only aggregate tables are
written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CELLS = ("A0-B0-C0", "A0-B0-C1")
MASS_FLAG = 0.90
BOUND = 0.10  # equivalence bound for the controls
BATTERIES = {
    "gender": {"items": ("A421", "A422", "A423", "A424", "A425"), "options": 5},
    "trust": {"items": ("N1601", "N1603", "N1605", "N1606", "N1611", "N1612", "N1613", "N1614"),
              "options": 4},
    "issp": {"items": ("M101", "M102", "M103", "M104", "M105", "M201", "M202"), "options": 5},
}
# name, battery, item, partner items, signed or absolute, role, prediction
INDICATORS = (
    ("M105", "issp", "M105", ("M102", "M103", "M104", "M202"), "signed", "cue",
     "reworded level > 0"),
    ("M201", "issp", "M201", ("M101", "M102", "M103", "M104", "M105", "M202"), "absolute", "cue",
     "difference < 0"),
    ("A423", "gender", "A423", ("A421", "A422", "A424"), "signed", "cue", "difference > 0"),
    ("A422", "gender", "A422", ("A421", "A423", "A424", "A425"), "absolute", "control",
     f"difference within +/-{BOUND}"),
    ("N1605", "trust", "N1605", ("N1601", "N1603", "N1606", "N1611", "N1612", "N1613", "N1614"),
     "absolute", "control", f"difference within +/-{BOUND}"),
)


def parse_args() -> argparse.Namespace:
    home = Path.home() / "Projects/restricted-data"
    p = argparse.ArgumentParser(description=__doc__)
    for b, d in (("gender", "surveyllm-v2"), ("trust", "surveyllm-v2-trust"), ("issp", "surveyllm-v2-issp")):
        p.add_argument(f"--{b}-ledger", default=home / d / "logprobs/option_probabilities_greedy.jsonl")
        p.add_argument(f"--{b}-wording-ledger",
                       default=home / "surveyllm-v2-wording" / b / "option_probabilities_greedy.jsonl")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261013)
    return p.parse_args()


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path, battery: str, items: tuple[str, ...], profiles=None):
    """Expected scores (cell, profile, item) for CELLS and the given items; valid-option mass."""
    k = BATTERIES[battery]["options"]
    scale = np.arange(1, k + 1, dtype=float)
    records = [r for r in map(json.loads, Path(path).open(encoding="utf-8"))
               if r["cell"] in CELLS and r["item"] in items]
    if profiles is None:
        profiles = sorted({(r["wave"], r["profile_id"]) for r in records})
    pos = {pid: i for i, (_, pid) in enumerate(profiles)}
    x = np.full((len(CELLS), len(profiles), len(items)), np.nan)
    mass = np.full_like(x, np.nan)
    keys = set()
    for r in records:
        prob = np.array([r["probabilities"][str(v)] for v in range(1, k + 1)], dtype=float)
        c, p, j = CELLS.index(r["cell"]), pos[r["profile_id"]], items.index(r["item"])
        x[c, p, j] = prob @ scale / prob.sum()
        mass[c, p, j] = r["valid_option_mass"]
        keys.add((r["cell"], r["profile_id"], r["item"]))
    expected = len(CELLS) * len(profiles) * len(items)
    if len(profiles) != 300 or len(keys) != expected or np.isnan(x).any():
        raise SystemExit(f"{path}: {len(keys)} unique records, expected {expected}")
    return profiles, x, mass


def indicator(target: np.ndarray, partners: np.ndarray, kind: str, groups) -> float:
    """Mean (signed or absolute) correlation of target with partners, per group, averaged."""
    vals = []
    for g in groups:
        t = target[g] - target[g].mean()
        o = partners[g] - partners[g].mean(axis=0)
        r = (o * t[:, None]).sum(0) / np.sqrt((o ** 2).sum(0) * (t ** 2).sum())
        vals.append(r.mean() if kind == "signed" else np.abs(r).mean())
    return float(np.mean(vals))


def evaluate(data, groups_by_battery, draws=None):
    out = {}
    for name, battery, item, partners, kind, _, _ in INDICATORS:
        orig, var = data[battery]["orig"], data[battery]["var"]
        items = BATTERIES[battery]["items"]
        groups = draws[battery] if draws else groups_by_battery[battery]
        j, js = items.index(item), [items.index(p) for p in partners]
        o = np.mean([indicator(orig[c][:, j], orig[c][:, js], kind, groups) for c in range(len(CELLS))])
        v = np.mean([indicator(var[c][:, item_pos(data, battery, item)], orig[c][:, js], kind, groups)
                     for c in range(len(CELLS))])
        out[name] = (o, v, v - o)
    return out


def item_pos(data, battery, item) -> int:
    return data[battery]["var_items"].index(item)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data, groups, inputs, masses = {}, {}, {}, []
    for battery in ("gender", "trust", "issp"):
        items = BATTERIES[battery]["items"]
        var_items = tuple(i for _, b, i, *_ in INDICATORS if b == battery)
        orig_path = getattr(args, f"{battery}_ledger")
        var_path = getattr(args, f"{battery}_wording_ledger")
        profiles, orig, _ = load(orig_path, battery, items)
        _, var, mass = load(var_path, battery, var_items, profiles)
        data[battery] = {"orig": orig, "var": var, "var_items": var_items}
        waves = np.array([w for w, _ in profiles])
        groups[battery] = [np.flatnonzero(waves == w) for w in sorted(set(waves))]
        inputs[f"{battery}_ledger_sha256"] = sha256(orig_path)
        inputs[f"{battery}_wording_ledger_sha256"] = sha256(var_path)
        for j, item in enumerate(var_items):
            for c, cell in enumerate(CELLS):
                masses.append({"item": item, "cell": cell,
                               "mean_valid_option_mass": float(mass[c, :, j].mean()),
                               "min_valid_option_mass": float(mass[c, :, j].min()),
                               "records_below_mass_flag": int((mass[c, :, j] < MASS_FLAG).sum())})

    point = evaluate(data, groups)
    rng = np.random.default_rng(args.seed)
    boot = {name: np.empty((args.replicates, 3)) for name in point}
    for rep in range(args.replicates):
        draws = {b: [rng.choice(g, len(g), replace=True) for g in groups[b]] for b in groups}
        for name, vals in evaluate(data, groups, draws).items():
            boot[name][rep] = vals

    rows, verdicts = [], {}
    for name, battery, item, partners, kind, role, prediction in INDICATORS:
        lo, hi = np.percentile(boot[name], [2.5, 97.5], axis=0)
        o, v, d = point[name]
        if name == "M105":
            holds = lo[1] > 0
        elif name == "M201":
            holds = hi[2] < 0
        elif name == "A423":
            holds = lo[2] > 0
        else:
            holds = lo[2] > -BOUND and hi[2] < BOUND
        verdicts[name] = bool(holds)
        rows.append({"item": item, "battery": battery, "role": role, "statistic": f"{kind} mean r",
                     "partners": " ".join(partners), "original": o, "original_ci95_low": lo[0],
                     "original_ci95_high": hi[0], "reworded": v, "reworded_ci95_low": lo[1],
                     "reworded_ci95_high": hi[1], "difference": d, "difference_ci95_low": lo[2],
                     "difference_ci95_high": hi[2], "prediction": prediction, "prediction_holds": holds})
    cue = all(verdicts[n] for n, *_ , r, _ in INDICATORS if r == "cue")
    control = all(verdicts[n] for n, *_ , r, _ in INDICATORS if r == "control")
    overall = ("supported" if cue and control else
               "not supported" if not any(verdicts.values()) else "partly supported")

    for fname, table in (("wording_experiment_results.csv", rows), ("wording_experiment_mass.csv", masses)):
        with (out_dir / fname).open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(table[0]))
            w.writeheader()
            w.writerows(table)
    meta = {"preregistration": "PREREGISTRATION_V2.md, 2026-11-22 entry", **inputs,
            "cells": list(CELLS), "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
            "bootstrap": "profiles resampled within battery (within wave for A421-A425); "
                         "original and reworded scores move together",
            "control_equivalence_bound": BOUND, "verdicts": verdicts,
            "cue_predictions_all_hold": cue, "control_predictions_all_hold": control,
            "overall": overall}
    (out_dir / "wording_experiment_metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    for r in rows:
        print(f"{r['item']:6} {r['role']:8} orig {r['original']:+.3f}  reworded {r['reworded']:+.3f} "
              f"[{r['reworded_ci95_low']:+.3f}, {r['reworded_ci95_high']:+.3f}]  diff {r['difference']:+.3f} "
              f"[{r['difference_ci95_low']:+.3f}, {r['difference_ci95_high']:+.3f}]  "
              f"{r['prediction']}: {'holds' if r['prediction_holds'] else 'does not hold'}")
    print(f"overall: {overall}; min mass {min(m['min_valid_option_mass'] for m in masses):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
