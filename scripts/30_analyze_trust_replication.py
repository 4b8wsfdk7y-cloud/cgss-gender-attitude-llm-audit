#!/usr/bin/env python3
# requires: Python 3.10+, numpy, pandas
"""Trust replication (PREREGISTRATION_V2.md, 2026-11-17, points 6-8).

Inputs are the profile-linked ledger written by
`07_run_logprob_audit.py --battery trust --history greedy`, the eligible CGSS
2012 respondents written by 28_export_trust_ml_input.R, and the HGB
out-of-fold predictions for the matched profiles written by 29_fit_trust_hgb.py.
Only aggregate tables are written and printed.

Primary outcome (point 6): for each cell, the mean absolute Pearson correlation
across the 300 profiles between the expected score on N1605 and the expected
scores on the other seven items; factorial effects as in Part 2, with a
profile-cluster bootstrap. Secondary outcomes (point 7): the N1612-N1613
correlation and mean difference, within- and between-block correlations, item
mean error against weighted CGSS 2012, and valid-option mass. References
(point 8): weighted CGSS 2012 at the individual level and the HGB expected
scores for the matched profiles. The matched respondents' own answers are
reported as an additional, unregistered descriptive row.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ITEMS = ("N1601", "N1603", "N1605", "N1606", "N1611", "N1612", "N1613", "N1614")
LABELS = ("亲戚", "邻居", "陌生人", "医生", "老师", "地方政府官员", "中央政府官员", "警察")
TARGET = ITEMS.index("N1605")
LOCAL, CENTRAL = ITEMS.index("N1612"), ITEMS.index("N1613")
BLOCKS = ((ITEMS.index("N1601"), ITEMS.index("N1603")),
          (ITEMS.index("N1606"), ITEMS.index("N1611")),
          (LOCAL, CENTRAL, ITEMS.index("N1614")))
CELLS = tuple(f"A{a}-B{b}-C{c}" for a in (0, 1) for b in (0, 1) for c in (0, 1))
LEVELS = tuple((a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1))
EFFECTS = ("A", "B", "C", "AB", "AC", "BC", "ABC")
SCALE = np.arange(1, 5, dtype=float)
MASS_FLAG = 0.90
N_PROFILES, N_RECORDS = 300, 300 * 8 * 8


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="option_probabilities_greedy.jsonl")
    parser.add_argument("--human", required=True, help="trust_ml_input.csv")
    parser.add_argument("--hgb", required=True, help="hgb_trust_matched.csv")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ledger(path: Path) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Expected scores (cell, profile, item) and valid-option mass, with completeness checks."""
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    keys = [(r["cell"], r["profile_id"], r["item"]) for r in records]
    if len(records) != N_RECORDS or len(set(keys)) != N_RECORDS:
        raise SystemExit(f"Ledger has {len(records)} records, {len(set(keys))} unique; "
                         f"expected {N_RECORDS}")
    profiles = sorted({r["profile_id"] for r in records})
    if len(profiles) != N_PROFILES:
        raise SystemExit(f"Ledger has {len(profiles)} profiles, expected {N_PROFILES}")
    pos = {p: i for i, p in enumerate(profiles)}
    expected = np.full((len(CELLS), N_PROFILES, len(ITEMS)), np.nan)
    mass = np.full_like(expected, np.nan)
    for r in records:
        c, p, j = CELLS.index(r["cell"]), pos[r["profile_id"]], ITEMS.index(r["item"])
        prob = np.array([r["probabilities"][str(k)] for k in range(1, 5)], dtype=float)
        expected[c, p, j] = prob @ SCALE / prob.sum()
        mass[c, p, j] = r["valid_option_mass"]
    if np.isnan(expected).any():
        raise SystemExit("Ledger is incomplete")
    return profiles, expected, mass


def corr_matrix(x: np.ndarray, w: np.ndarray | None = None) -> np.ndarray:
    """Pearson correlations of the columns of x (rows are units), optionally weighted."""
    return np.corrcoef(x, rowvar=False) if w is None else _weighted_corr(x, w)


def _weighted_corr(x: np.ndarray, w: np.ndarray) -> np.ndarray:
    cov = np.cov(x, rowvar=False, aweights=w)
    sd = np.sqrt(np.diag(cov))
    return cov / np.outer(sd, sd)


def structure(x: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    """All correlation-based outcomes for one set of scores (units x items)."""
    r = corr_matrix(x, w)
    others = [j for j in range(len(ITEMS)) if j != TARGET]
    within = [r[i, j] for block in BLOCKS for k, i in enumerate(block) for j in block[k + 1:]]
    block_of = {j: b for b, block in enumerate(BLOCKS) for j in block}
    between = [r[i, j] for i in block_of for j in block_of if i < j and block_of[i] != block_of[j]]
    out = {"stranger_mean_abs_correlation": float(np.mean(np.abs(r[TARGET, others]))),
           "local_central_correlation": float(r[LOCAL, CENTRAL])}
    for b, name in enumerate(("interpersonal", "occupations", "authorities")):
        block = BLOCKS[b]
        out[f"within_{name}"] = float(np.mean([r[i, j] for k, i in enumerate(block) for j in block[k + 1:]]))
    out["within_blocks_mean"] = float(np.mean(within))
    out["between_blocks_mean"] = float(np.mean(between))
    return out


def stranger_outcome(expected: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Primary outcome for every cell on the profile rows given (vectorised for the bootstrap)."""
    x = expected[:, rows, :]
    x = x - x.mean(axis=1, keepdims=True)
    target = x[:, :, TARGET:TARGET + 1]
    others = np.delete(x, TARGET, axis=2)
    r = (others * target).sum(1) / np.sqrt((others ** 2).sum(1) * (target ** 2).sum(1))
    return np.abs(r).mean(axis=1)


def factorial_effects(outcome: np.ndarray) -> dict[str, float]:
    """As in 09_analyze_logprob_audit.py: main effects average over the other
    factors; interactions are differences of simple effects, not halved."""
    y = np.zeros((2, 2, 2))
    for value, (a, b, c) in zip(outcome, LEVELS):
        y[a, b, c] = value

    def did(m: np.ndarray) -> float:
        return (m[1, 1] - m[0, 1]) - (m[1, 0] - m[0, 0])

    return {
        "A": y[1].mean() - y[0].mean(),
        "B": y[:, 1].mean() - y[:, 0].mean(),
        "C": y[:, :, 1].mean() - y[:, :, 0].mean(),
        "AB": np.mean([did(y[:, :, c]) for c in (0, 1)]),
        "AC": np.mean([did(y[:, b, :]) for b in (0, 1)]),
        "BC": np.mean([did(y[a, :, :]) for a in (0, 1)]),
        "ABC": did(y[:, :, 1]) - did(y[:, :, 0]),
    }


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Path(args.ledger).expanduser()
    profiles, expected, mass = load_ledger(ledger)

    human = pd.read_csv(Path(args.human).expanduser(), dtype={"profile_id": str})
    answers = human[[f"t_{i.lower()}" for i in ITEMS]].to_numpy(dtype=float)
    weights = human["analysis_weight"].to_numpy(dtype=float)
    cgss_mean = np.average(answers, axis=0, weights=weights)
    matched = human[human["is_llm_profile"] == 1].set_index("profile_id").loc[profiles]
    matched_answers = matched[[f"t_{i.lower()}" for i in ITEMS]].to_numpy(dtype=float)

    hgb = pd.read_csv(Path(args.hgb).expanduser(), dtype={"profile_id": str}).set_index("profile_id").loc[profiles]
    hgb_expected = np.stack([hgb[[f"{i}_p{k}" for k in range(1, 5)]].to_numpy() @ SCALE for i in ITEMS], axis=1)

    # References (point 8) and the unregistered matched-respondent row.
    references = []
    for name, x, w in (("CGSS 2012, weighted, eligible respondents", answers, weights),
                       ("HGB expected scores, matched profiles", hgb_expected, None),
                       ("matched respondents' answers, unweighted (descriptive)", matched_answers, None)):
        row = {"reference": name, "n": len(x), **structure(x, w)}
        means = np.average(x, axis=0, weights=w)
        row["central_minus_local_mean"] = float(means[CENTRAL] - means[LOCAL])
        row["item_mean_abs_error_vs_cgss"] = float(np.mean(np.abs(means - cgss_mean)))
        references.append(row)
    pd.DataFrame(references).to_csv(out / "trust_references.csv", index=False)

    # Primary outcome with the profile-cluster bootstrap (one wave, no strata).
    all_rows = np.arange(N_PROFILES)
    primary = stranger_outcome(expected, all_rows)
    estimates = factorial_effects(primary)
    rng = np.random.default_rng(args.seed)
    boot_primary = np.empty((args.replicates, len(CELLS)))
    boot_effects = np.empty((args.replicates, len(EFFECTS)))
    boot_secondary_b = np.empty((args.replicates, 2))
    for b in range(args.replicates):
        draw = rng.choice(all_rows, size=N_PROFILES, replace=True)
        boot_primary[b] = stranger_outcome(expected, draw)
        effects = factorial_effects(boot_primary[b])
        boot_effects[b] = [effects[e] for e in EFFECTS]
        sec = [structure(expected[c, draw]) for c in range(len(CELLS))]
        boot_secondary_b[b] = [
            factorial_effects(np.array([s[k] for s in sec]))["B"]
            for k in ("local_central_correlation", "between_blocks_mean")]

    by_cell = []
    for c, cell in enumerate(CELLS):
        low, high = np.percentile(boot_primary[:, c], [2.5, 97.5])
        s = structure(expected[c])
        means = expected[c].mean(axis=0)
        by_cell.append({
            "cell": cell, "battery_visible": LEVELS[c][0], "answer_history": LEVELS[c][1],
            "json_output": LEVELS[c][2],
            "stranger_mean_abs_correlation": primary[c], "ci95_low": low, "ci95_high": high,
            **{k: v for k, v in s.items() if k != "stranger_mean_abs_correlation"},
            "central_minus_local_mean": float(means[CENTRAL] - means[LOCAL]),
            "item_mean_abs_error_vs_cgss": float(np.mean(np.abs(means - cgss_mean))),
            "mean_valid_option_mass": float(mass[c].mean()),
            "min_valid_option_mass": float(mass[c].min()),
            "records_below_mass_flag": int((mass[c] < MASS_FLAG).sum()),
            "mass_flag": bool(mass[c].mean() < MASS_FLAG),
        })
    pd.DataFrame(by_cell).to_csv(out / "trust_by_cell.csv", index=False)

    effect_rows = []
    for k, name in enumerate(EFFECTS):
        low, high = np.percentile(boot_effects[:, k], [2.5, 97.5])
        effect_rows.append({"outcome": "stranger_mean_abs_correlation", "effect": name,
                            "estimate": estimates[name], "ci95_low": low, "ci95_high": high})
    for k, name in enumerate(("local_central_correlation", "between_blocks_mean")):
        est = factorial_effects(np.array([row[name] for row in by_cell]))["B"]
        low, high = np.percentile(boot_secondary_b[:, k], [2.5, 97.5])
        effect_rows.append({"outcome": name, "effect": "B", "estimate": est,
                            "ci95_low": low, "ci95_high": high})
    pd.DataFrame(effect_rows).to_csv(out / "trust_effects.csv", index=False)

    item_rows = []
    for c, cell in enumerate(CELLS):
        means = expected[c].mean(axis=0)
        for j, (item, label) in enumerate(zip(ITEMS, LABELS)):
            item_rows.append({"cell": cell, "item": item, "label": label, "model_mean": means[j],
                              "cgss_weighted_mean": cgss_mean[j], "error": means[j] - cgss_mean[j]})
    pd.DataFrame(item_rows).to_csv(out / "trust_item_means_by_cell.csv", index=False)

    run_meta_path = ledger.parent / "run_metadata_greedy.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.exists() else {}
    metadata = {
        "preregistration": "PREREGISTRATION_V2.md, 2026-11-17 entry",
        "ledger_sha256": sha256(ledger), "records": N_RECORDS, "profiles": N_PROFILES,
        "human_sha256": sha256(Path(args.human).expanduser()),
        "hgb_sha256": sha256(Path(args.hgb).expanduser()),
        "run_code_commit": run_meta.get("code_commit"),
        "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
        "bootstrap": "profiles resampled with replacement; all cells and items move together",
        "interaction_definition": "differences of simple effects, not halved",
        "expected_score": "option probabilities renormalised over 1-4",
        "mass_flag_threshold": MASS_FLAG,
    }
    (out / "trust_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"records {N_RECORDS}, profiles {N_PROFILES}, run commit {run_meta.get('code_commit')}")
    print("Primary outcome by cell (stranger mean |r|):")
    for row in by_cell:
        print(f"  {row['cell']}  {row['stranger_mean_abs_correlation']:.3f} "
              f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]  mass {row['mean_valid_option_mass']:.4f}")
    print("Effects:")
    for row in effect_rows:
        print(f"  {row['outcome']:32s} {row['effect']:4s} {row['estimate']:+.3f} "
              f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]")
    print("References:")
    for row in references:
        print(f"  {row['reference']}: stranger {row['stranger_mean_abs_correlation']:.3f}, "
              f"local-central r {row['local_central_correlation']:.3f}, "
              f"central-local mean {row['central_minus_local_mean']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
