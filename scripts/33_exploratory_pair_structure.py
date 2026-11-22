#!/usr/bin/env python3
# requires: Python 3.10+, numpy
"""Exploratory item-pair structure: model correlations without history vs survey-trained HGB.

Post hoc, written 2026-11-22 after the battery-level results of A421-A425, the
trust battery, and the ISSP battery were known; the plan was fixed before any
pair-level value was computed (its text and hash are stored in the metadata).

For every item pair within a battery (10 + 28 + 21 = 59 pairs) the script
computes the Pearson correlation of HGB expected scores across the matched
profiles (r_H) and the mean over the four no-history cells of the model's
per-cell correlation (r_M0), and the same over the four history cells (r_M1).
A421-A425 correlations are computed within wave and averaged over waves.
The main fit is |r_M0| = a + b |r_H| across pairs, with a profile-cluster
bootstrap. Inputs are profile-linked and stay outside the repository; only
aggregate tables are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

CELLS = ["A0-B0-C0", "A0-B0-C1", "A0-B1-C0", "A0-B1-C1",
         "A1-B0-C0", "A1-B0-C1", "A1-B1-C0", "A1-B1-C1"]
B0 = [c for c, name in enumerate(CELLS) if "-B0-" in name]
B1 = [c for c, name in enumerate(CELLS) if "-B1-" in name]
BATTERIES = {
    "gender": {"items": ("A421", "A422", "A423", "A424", "A425"), "options": 5},
    "trust": {"items": ("N1601", "N1603", "N1605", "N1606", "N1611", "N1612", "N1613", "N1614"),
              "options": 4},
    "issp": {"items": ("M101", "M102", "M103", "M104", "M105", "M201", "M202"), "options": 5},
}


def parse_args() -> argparse.Namespace:
    home = Path.home() / "Projects/restricted-data"
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--gender-ledger", default=home / "surveyllm-v2/logprobs/option_probabilities_greedy.jsonl")
    p.add_argument("--gemma-ledger", default=home / "surveyllm-v2/gemma4/logprobs/option_probabilities_greedy.jsonl")
    p.add_argument("--trust-ledger", default=home / "surveyllm-v2-trust/logprobs/option_probabilities_greedy.jsonl")
    p.add_argument("--issp-ledger", default=home / "surveyllm-v2-issp/logprobs/option_probabilities_greedy.jsonl")
    p.add_argument("--gender-hgb", default=home / "04_学年论文/CSS_writing sample/ml/output/predictions.csv")
    p.add_argument("--trust-hgb", default=home / "surveyllm-v2-trust/hgb/hgb_trust_matched.csv")
    p.add_argument("--issp-hgb", default=home / "surveyllm-v2-issp/hgb/hgb_issp_matched.csv")
    p.add_argument("--plan", required=True, help="plan written before the run")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--replicates", type=int, default=2000)
    p.add_argument("--seed", type=int, default=20261013)
    return p.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_ledger(path: Path, battery: str):
    """Expected scores (cell, profile, item), renormalised over the valid options; profile waves."""
    items, k = BATTERIES[battery]["items"], BATTERIES[battery]["options"]
    records = [json.loads(line) for line in Path(path).open(encoding="utf-8")]
    profiles = sorted({(r["wave"], r["profile_id"]) for r in records})
    if len(profiles) != 300 or len(records) != 300 * len(CELLS) * len(items):
        raise SystemExit(f"{path}: {len(records)} records, {len(profiles)} profiles")
    pos = {pid: i for i, (_, pid) in enumerate(profiles)}
    scale = np.arange(1, k + 1, dtype=float)
    x = np.full((len(CELLS), 300, len(items)), np.nan)
    for r in records:
        prob = np.array([r["probabilities"][str(v)] for v in range(1, k + 1)], dtype=float)
        x[CELLS.index(r["cell"]), pos[r["profile_id"]], items.index(r["item"])] = prob @ scale / prob.sum()
    if np.isnan(x).any():
        raise SystemExit(f"{path} is incomplete")
    return profiles, x


def load_hgb_gender(path: Path, profiles) -> np.ndarray:
    """HGB out-of-fold expected scores, equality coding converted back to agreement (6 - E for A421-A424)."""
    pos = {pid: i for i, (_, pid) in enumerate(profiles)}
    h = np.full((300, 5), np.nan)
    with Path(path).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["split"] == "oof" and r["model"] == "hist_gradient_boosting" and r["profile_id"] in pos:
                j = int(r["item"][-1]) - 1
                e = sum(v * float(r[f"p{v}"]) for v in range(1, 6))
                h[pos[r["profile_id"]], j] = e if j == 4 else 6 - e
    return h


def load_hgb_wide(path: Path, profiles, battery: str) -> np.ndarray:
    items, k = BATTERIES[battery]["items"], BATTERIES[battery]["options"]
    pos = {pid: i for i, (_, pid) in enumerate(profiles)}
    h = np.full((300, len(items)), np.nan)
    with Path(path).open(newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            for j, item in enumerate(items):
                prob = np.array([float(r[f"{item}_p{v}"]) for v in range(1, k + 1)])
                h[pos[r["profile_id"]], j] = prob @ np.arange(1, k + 1) / prob.sum()
    if np.isnan(h).any():
        raise SystemExit(f"{path} is incomplete")
    return h


def corr(x: np.ndarray) -> np.ndarray:
    """Pairwise-complete Pearson correlation matrix of the columns of x (rows are profiles)."""
    n = x.shape[1]
    r = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1, n):
            ok = ~np.isnan(x[:, i]) & ~np.isnan(x[:, j])
            r[i, j] = r[j, i] = np.corrcoef(x[ok, i], x[ok, j])[0, 1]
    return r


def pair_values(model: np.ndarray, hgb: np.ndarray, groups: list[np.ndarray]):
    """Signed per-pair r_H, r_M0, r_M1 and |r| versions, averaged over groups (waves)."""
    n = model.shape[2]
    iu = np.triu_indices(n, 1)
    rh, rm = [], []
    complete = [g[~np.isnan(hgb[g]).any(axis=1)] for g in groups]   # listwise, as in script 09
    for g, gc in zip(groups, complete):
        rh.append(corr(hgb[gc])[iu])
        rm.append(np.stack([corr(model[c][g])[iu] for c in range(len(CELLS))]))
    rh, rm = np.mean(rh, axis=0), np.mean(rm, axis=0)       # (pairs,), (cells, pairs)
    abs_rm_by_wave = np.mean([np.abs(np.stack([corr(model[c][g])[iu] for c in range(len(CELLS))]))
                              for g in groups], axis=0)
    abs_rh = np.mean([np.abs(corr(hgb[gc])[iu]) for gc in complete], axis=0)
    return {"r_h": rh, "r_m0": rm[B0].mean(0), "r_m1": rm[B1].mean(0),
            "abs_r_h": abs_rh, "abs_r_m0": abs_rm_by_wave[B0].mean(0),
            "abs_r_m1": abs_rm_by_wave[B1].mean(0)}


def fit(y: np.ndarray, x: np.ndarray) -> dict[str, float]:
    b, a = np.polyfit(x, y, 1)
    resid = y - (a + b * x)
    ss = ((y - y.mean()) ** 2).sum()
    return {"intercept": a, "slope": b, "r2": 1 - (resid ** 2).sum() / ss if ss > 0 else np.nan,
            "residual_sd": float(np.sqrt((resid ** 2).sum() / (len(y) - 2)))}


def summaries(v: dict[str, dict[str, np.ndarray]], qwen: list[str]) -> dict[str, float]:
    """All scalar quantities of the plan for one set of pair values."""
    cat = {k: np.concatenate([v[b][k] for b in qwen]) for k in v[qwen[0]]}
    out = {}
    for name, f in (("m0", fit(cat["abs_r_m0"], cat["abs_r_h"])),
                    ("m1", fit(cat["abs_r_m1"], cat["abs_r_h"])),
                    ("increment_on_m0", fit(cat["abs_r_m1"] - cat["abs_r_m0"], cat["abs_r_m0"]))):
        out.update({f"pooled_{name}_{k}": val for k, val in f.items()})
    for b in qwen + ["gemma_gender"]:
        if b not in v:
            continue
        f = fit(v[b]["abs_r_m0"], v[b]["abs_r_h"])
        out.update({f"{b}_m0_{k}": val for k, val in f.items()})
        out[f"{b}_mean_abs_r_m0"] = float(v[b]["abs_r_m0"].mean())
        out[f"{b}_mean_abs_r_m1"] = float(v[b]["abs_r_m1"].mean())
        out[f"{b}_mean_abs_r_h"] = float(v[b]["abs_r_h"].mean())
    same = np.sign(cat["r_m0"]) == np.sign(cat["r_h"])
    strong = np.abs(cat["r_h"]) >= 0.10
    out["pooled_sign_agreement"] = float(same.mean())
    out["pooled_sign_agreement_abs_r_h_ge_0_10"] = float(same[strong].mean())
    out["pooled_pairs_abs_r_h_ge_0_10"] = int(strong.sum())
    return out


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = {}
    for name, ledger, battery in (("gender", args.gender_ledger, "gender"),
                                  ("gemma_gender", args.gemma_ledger, "gender"),
                                  ("trust", args.trust_ledger, "trust"),
                                  ("issp", args.issp_ledger, "issp")):
        profiles, x = load_ledger(ledger, battery)
        if battery == "gender":
            h = load_hgb_gender(args.gender_hgb, profiles)
        else:
            h = load_hgb_wide(getattr(args, f"{battery}_hgb"), profiles, battery)
        waves = np.array([w for w, _ in profiles])
        groups = [np.flatnonzero(waves == w) for w in sorted(set(waves))]
        data[name] = (x, h, groups, battery)
    if [p for p in load_ledger(args.gender_ledger, "gender")[0]] != \
       [p for p in load_ledger(args.gemma_ledger, "gender")[0]]:
        raise SystemExit("Qwen and Gemma gender ledgers have different profiles")

    qwen = ["gender", "trust", "issp"]
    point_pairs = {n: pair_values(x, h, g) for n, (x, h, g, _) in data.items()}
    point = summaries(point_pairs, qwen)

    rng = np.random.default_rng(args.seed)
    boot = {k: np.empty(args.replicates) for k in point}
    for rep in range(args.replicates):
        drawn = {}
        gender_draw = None
        for n, (x, h, g, _) in data.items():
            if n in ("gender", "gemma_gender"):
                # Qwen and Gemma share profiles: the same draw for both.
                gender_draw = gender_draw or [rng.choice(idx, len(idx), replace=True) for idx in g]
                draw = gender_draw
            else:
                draw = [rng.choice(idx, len(idx), replace=True) for idx in g]
            drawn[n] = pair_values(x, h, draw)
        s = summaries(drawn, qwen)
        for k in point:
            boot[k][rep] = s[k]

    rows = []
    for k, val in point.items():
        low, high = np.nanpercentile(boot[k], [2.5, 97.5])
        rows.append({"quantity": k, "estimate": val, "ci95_low": low, "ci95_high": high})
    with (out / "exploratory_pairs_summary.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    pair_rows = []
    for n, (x, h, g, battery) in data.items():
        items = BATTERIES[battery]["items"]
        iu = np.triu_indices(len(items), 1)
        v = point_pairs[n]
        for p, (i, j) in enumerate(zip(*iu)):
            pair_rows.append({"model": "gemma4_12b" if n == "gemma_gender" else "qwen3_5_9b",
                              "battery": battery, "item_x": items[i], "item_y": items[j],
                              **{k: float(v[k][p]) for k in v}})
    with (out / "exploratory_pairs_by_pair.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(pair_rows[0]))
        w.writeheader()
        w.writerows(pair_rows)

    meta = {
        "status": "exploratory, post hoc; plan fixed before any pair-level value was computed",
        "plan_sha256": sha256(args.plan), "plan": Path(args.plan).read_text(encoding="utf-8"),
        "inputs_sha256": {k: sha256(getattr(args, k)) for k in
                          ("gender_ledger", "gemma_ledger", "trust_ledger", "issp_ledger",
                           "gender_hgb", "trust_hgb", "issp_hgb")},
        "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
        "bootstrap": "profiles resampled within battery (within wave for A421-A425); "
                     "Qwen and Gemma gender draws shared; model and HGB move together",
        "pairs": {n: len(point_pairs[n]["r_h"]) for n in point_pairs},
    }
    (out / "exploratory_pairs_metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                                        encoding="utf-8")
    for r in rows:
        print(f"{r['quantity']:<45} {r['estimate']:+.3f} [{r['ci95_low']:+.3f}, {r['ci95_high']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
