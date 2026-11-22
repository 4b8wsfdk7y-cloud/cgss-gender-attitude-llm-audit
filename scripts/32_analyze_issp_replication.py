#!/usr/bin/env python3
# requires: Python 3.10+, numpy, pandas
"""Second gender battery (PREREGISTRATION_V2.md, 2026-11-18, points 6-9).

Inputs are the profile-linked ledger written by
`07_run_logprob_audit.py --battery issp --history greedy`, the eligible CGSS
2012 respondents written by `28_export_trust_ml_input.R DIR issp`, and the HGB
out-of-fold predictions for the matched profiles written by
`29_fit_trust_hgb.py --battery issp`. Only aggregate tables are written and
printed.

Primary outcome (point 6): for each cell, the mean absolute Pearson correlation
across the 300 profiles between the expected score on M201 and the expected
scores on the other six items; factorial effects as in Part 2, with a
profile-cluster bootstrap. Source comparison (point 7): the excess of the cells
without history over the HGB value, E0 = mean(B0 cells) - HGB, and
D = B - E0, with the HGB value recomputed on every bootstrap draw. Secondary
outcomes (point 8): the mean absolute correlation over all 21 item pairs, the
M101 outcome, item mean error against weighted CGSS 2012, and valid-option
mass. References (point 9): weighted CGSS 2012 and the HGB expected scores for
the matched profiles; the matched respondents' own answers are an additional,
unregistered descriptive row.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location(
    "trust_analysis", Path(__file__).resolve().parent / "30_analyze_trust_replication.py")
trust = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(trust)
factorial_effects, corr_matrix = trust.factorial_effects, trust.corr_matrix
CELLS, LEVELS, EFFECTS = trust.CELLS, trust.LEVELS, trust.EFFECTS

ITEMS = ("M101", "M102", "M103", "M104", "M105", "M201", "M202")
TARGET = ITEMS.index("M201")
SECOND = ITEMS.index("M101")
N_OPTIONS = 5
SCALE = np.arange(1, N_OPTIONS + 1, dtype=float)
MASS_FLAG = 0.90
N_PROFILES, N_RECORDS = 300, 300 * 8 * len(ITEMS)
B0 = [c for c, (_, b, _) in enumerate(LEVELS) if b == 0]
PAIRS = np.triu_indices(len(ITEMS), 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="option_probabilities_greedy.jsonl")
    parser.add_argument("--human", required=True, help="issp_ml_input.csv")
    parser.add_argument("--hgb", required=True, help="hgb_issp_matched.csv")
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
        prob = np.array([r["probabilities"][str(k)] for k in range(1, N_OPTIONS + 1)], dtype=float)
        expected[c, p, j] = (prob * SCALE).sum() / prob.sum()
        mass[c, p, j] = r["valid_option_mass"]
    if np.isnan(expected).any():
        raise SystemExit("Ledger is incomplete")
    return profiles, expected, mass


def structure(x: np.ndarray, w: np.ndarray | None = None) -> dict[str, float]:
    """Correlation-based outcomes for one set of scores (units x items)."""
    r = corr_matrix(x, w)

    def item_outcome(j: int) -> float:
        return float(np.mean(np.abs(np.delete(r[j], j))))

    return {"m201_mean_abs_correlation": item_outcome(TARGET),
            "m101_mean_abs_correlation": item_outcome(SECOND),
            "all_pairs_mean_abs_correlation": float(np.mean(np.abs(r[PAIRS])))}


def item_outcome(x: np.ndarray, j: int) -> np.ndarray:
    """Mean |r| of item j with the others, for scores shaped (..., units, items)."""
    x = x - x.mean(axis=-2, keepdims=True)
    target = x[..., j:j + 1]
    others = np.delete(x, j, axis=-1)
    r = (others * target).sum(-2) / np.sqrt((others ** 2).sum(-2) * (target ** 2).sum(-2))
    return np.abs(r).mean(axis=-1)


def all_pairs(x: np.ndarray) -> np.ndarray:
    """Mean |r| over all item pairs, for scores shaped (cells, units, items)."""
    return np.array([np.mean(np.abs(np.corrcoef(c, rowvar=False)[PAIRS])) for c in x])


def source_comparison(primary: np.ndarray, hgb_value: float) -> dict[str, float]:
    b = factorial_effects(primary)["B"]
    e0 = float(primary[B0].mean() - hgb_value)
    return {"B": b, "E0": e0, "D": b - e0}


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Path(args.ledger).expanduser()
    profiles, expected, mass = load_ledger(ledger)

    columns = [f"t_{i.lower()}" for i in ITEMS]
    human = pd.read_csv(Path(args.human).expanduser(), dtype={"profile_id": str})
    answers = human[columns].to_numpy(dtype=float)
    weights = human["analysis_weight"].to_numpy(dtype=float)
    cgss_mean = np.average(answers, axis=0, weights=weights)
    matched = human[human["is_llm_profile"] == 1].set_index("profile_id").loc[profiles]
    matched_answers = matched[columns].to_numpy(dtype=float)

    hgb = pd.read_csv(Path(args.hgb).expanduser(), dtype={"profile_id": str}).set_index("profile_id").loc[profiles]
    hgb_prob = np.stack([hgb[[f"{i}_p{k}" for k in range(1, N_OPTIONS + 1)]].to_numpy() for i in ITEMS], axis=1)
    hgb_expected = (hgb_prob * SCALE).sum(axis=2)

    references = []
    for name, x, w in (("CGSS 2012, weighted, eligible respondents", answers, weights),
                       ("HGB expected scores, matched profiles", hgb_expected, None),
                       ("matched respondents' answers, unweighted (descriptive)", matched_answers, None)):
        means = np.average(x, axis=0, weights=w)
        references.append({"reference": name, "n": len(x), **structure(x, w),
                           "item_mean_abs_error_vs_cgss": float(np.mean(np.abs(means - cgss_mean)))})
    pd.DataFrame(references).to_csv(out / "issp_references.csv", index=False)

    all_rows = np.arange(N_PROFILES)
    primary = item_outcome(expected, TARGET)
    second = item_outcome(expected, SECOND)
    pairs = all_pairs(expected)
    hgb_primary = float(item_outcome(hgb_expected, TARGET))
    estimates = factorial_effects(primary)
    source = source_comparison(primary, hgb_primary)

    rng = np.random.default_rng(args.seed)
    boot_primary = np.empty((args.replicates, len(CELLS)))
    boot_effects = np.empty((args.replicates, len(EFFECTS)))
    boot_source = np.empty((args.replicates, 3))
    boot_secondary_b = np.empty((args.replicates, 2))
    for b in range(args.replicates):
        draw = rng.choice(all_rows, size=N_PROFILES, replace=True)
        x = expected[:, draw]
        boot_primary[b] = item_outcome(x, TARGET)
        effects = factorial_effects(boot_primary[b])
        boot_effects[b] = [effects[e] for e in EFFECTS]
        s = source_comparison(boot_primary[b], float(item_outcome(hgb_expected[draw], TARGET)))
        boot_source[b] = [s["B"], s["E0"], s["D"]]
        boot_secondary_b[b] = [factorial_effects(all_pairs(x))["B"],
                               factorial_effects(item_outcome(x, SECOND))["B"]]

    by_cell = []
    for c, cell in enumerate(CELLS):
        low, high = np.percentile(boot_primary[:, c], [2.5, 97.5])
        means = expected[c].mean(axis=0)
        by_cell.append({
            "cell": cell, "battery_visible": LEVELS[c][0], "answer_history": LEVELS[c][1],
            "json_output": LEVELS[c][2],
            "m201_mean_abs_correlation": primary[c], "ci95_low": low, "ci95_high": high,
            "m101_mean_abs_correlation": second[c],
            "all_pairs_mean_abs_correlation": pairs[c],
            "item_mean_abs_error_vs_cgss": float(np.mean(np.abs(means - cgss_mean))),
            "mean_valid_option_mass": float(mass[c].mean()),
            "min_valid_option_mass": float(mass[c].min()),
            "records_below_mass_flag": int((mass[c] < MASS_FLAG).sum()),
            "mass_flag": bool(mass[c].mean() < MASS_FLAG),
        })
    pd.DataFrame(by_cell).to_csv(out / "issp_by_cell.csv", index=False)

    effect_rows = []
    for k, name in enumerate(EFFECTS):
        low, high = np.percentile(boot_effects[:, k], [2.5, 97.5])
        effect_rows.append({"outcome": "m201_mean_abs_correlation", "effect": name,
                            "estimate": estimates[name], "ci95_low": low, "ci95_high": high})
    for k, name in enumerate(("B", "E0", "D")):
        low, high = np.percentile(boot_source[:, k], [2.5, 97.5])
        effect_rows.append({"outcome": "m201_source_comparison", "effect": name,
                            "estimate": source[name], "ci95_low": low, "ci95_high": high})
    for k, (name, values) in enumerate((("all_pairs_mean_abs_correlation", pairs),
                                        ("m101_mean_abs_correlation", second))):
        low, high = np.percentile(boot_secondary_b[:, k], [2.5, 97.5])
        effect_rows.append({"outcome": name, "effect": "B",
                            "estimate": factorial_effects(values)["B"],
                            "ci95_low": low, "ci95_high": high})
    pd.DataFrame(effect_rows).to_csv(out / "issp_effects.csv", index=False)

    d_low, d_high = np.percentile(boot_source[:, 2], [2.5, 97.5])
    verdict = ("answer history is the larger source" if d_low > 0 else
               "the excess without history is the larger source" if d_high < 0 else
               "undetermined")

    item_rows = []
    for c, cell in enumerate(CELLS):
        means = expected[c].mean(axis=0)
        for j, item in enumerate(ITEMS):
            item_rows.append({"cell": cell, "item": item, "model_mean": means[j],
                              "cgss_weighted_mean": cgss_mean[j], "error": means[j] - cgss_mean[j]})
    pd.DataFrame(item_rows).to_csv(out / "issp_item_means_by_cell.csv", index=False)

    run_meta_path = ledger.parent / "run_metadata_greedy.json"
    run_meta = json.loads(run_meta_path.read_text(encoding="utf-8")) if run_meta_path.exists() else {}
    metadata = {
        "preregistration": "PREREGISTRATION_V2.md, 2026-11-18 entry",
        "ledger_sha256": sha256(ledger), "records": N_RECORDS, "profiles": N_PROFILES,
        "human_sha256": sha256(Path(args.human).expanduser()),
        "hgb_sha256": sha256(Path(args.hgb).expanduser()),
        "run_code_commit": run_meta.get("code_commit"),
        "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
        "bootstrap": "profiles resampled with replacement; all cells, items, and the HGB value move together",
        "interaction_definition": "differences of simple effects, not halved",
        "source_comparison": "E0 = mean of the four B0 cells minus HGB; D = B - E0",
        "source_verdict": verdict,
        "expected_score": "option probabilities renormalised over 1-5",
        "mass_flag_threshold": MASS_FLAG,
    }
    (out / "issp_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"records {N_RECORDS}, profiles {N_PROFILES}, run commit {run_meta.get('code_commit')}")
    print("Primary outcome by cell (M201 mean |r|):")
    for row in by_cell:
        print(f"  {row['cell']}  {row['m201_mean_abs_correlation']:.3f} "
              f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]  mass {row['mean_valid_option_mass']:.4f}")
    print("Effects:")
    for row in effect_rows:
        print(f"  {row['outcome']:32s} {row['effect']:4s} {row['estimate']:+.3f} "
              f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]")
    print(f"Source comparison: {verdict}")
    print("References:")
    for row in references:
        print(f"  {row['reference']}: M201 {row['m201_mean_abs_correlation']:.3f}, "
              f"M101 {row['m101_mean_abs_correlation']:.3f}, "
              f"all pairs {row['all_pairs_mean_abs_correlation']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
