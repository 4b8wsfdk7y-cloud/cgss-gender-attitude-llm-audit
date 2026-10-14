#!/usr/bin/env python3
# requires: Python 3.10+, numpy, the survey_llm_eval package
"""Parts 1 and 2 of PREREGISTRATION_V2.md from the greedy option-probability run.

Inputs are the profile-linked ledger written by 07_run_logprob_audit.py, the
matched respondents' answers, the weighted CGSS aggregates written by
08_export_human_benchmark.R, and v1 aggregate metrics. Optional inputs add the
exploratory label cell and the HGB expected-score comparator (see the Changes
section of the pre-registration). Only aggregate tables are written.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from survey_llm_eval.factorial import CELLS, item_ids
from survey_llm_eval.probabilities import (
    expected_score,
    log_score,
    mixture,
    ranked_probability_score,
    variance_decomposition,
)

SCALE = (1, 2, 3, 4, 5)
REFERENCE_CELL = "A1-B1-C1"
LABEL_CELL = "A1-B1-C2"
EFFECTS = ("A", "B", "C", "AB", "AC", "BC", "ABC")
VALID = {"1", "2", "3", "4", "5"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="option_probabilities_greedy.jsonl")
    parser.add_argument("--matched", required=True, help="profiles_pilot.csv (matched answers)")
    parser.add_argument("--human-dir", required=True, help="output of 08_export_human_benchmark.R")
    parser.add_argument("--v1-dir", required=True, help="v1 Qwen3.5 output directory")
    parser.add_argument("--ml-matched", required=True, help="v1 metrics_matched_300.csv")
    parser.add_argument("--ml-predictions", help="v1 predictions.csv, for the HGB comparator")
    parser.add_argument("--exploratory-ledger", help="ledger with the A1-B1-C2 label cell")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_ledger(path: Path, cells: list[str], profiles: list[tuple[str, str]] | None = None):
    """Option probabilities as an array (cell, profile, item, option)."""
    records = [json.loads(line) for line in path.open(encoding="utf-8")]
    records = [r for r in records if r["cell"] in cells]
    if profiles is None:
        profiles = sorted({(r["wave"], r["profile_id"]) for r in records})
    position = {pid: i for i, (_, pid) in enumerate(profiles)}
    items = list(item_ids())
    probs = np.full((len(cells), len(profiles), len(items), len(SCALE)), np.nan)
    mass = np.full((len(cells), len(profiles), len(items)), np.nan)
    for r in records:
        c, p, j = cells.index(r["cell"]), position[r["profile_id"]], items.index(r["item"])
        probs[c, p, j] = [r["probabilities"][str(k)] for k in SCALE]
        mass[c, p, j] = r["valid_option_mass"]
    if np.isnan(probs).any():
        raise SystemExit(f"{path} is incomplete for cells {cells}")
    return len(records), profiles, probs, mass


def as_dist(row: np.ndarray) -> dict[int, float]:
    return {k: float(v) for k, v in zip(SCALE, row)}


def total_variation(p: dict[int, float], q: dict[int, float]) -> float:
    return 0.5 * sum(abs(p.get(k, 0.0) - q.get(k, 0.0)) for k in SCALE)


def primary_outcome(expected: np.ndarray, wave_index: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Mean |r| of A425 with A421-A424 across profiles, by cell; per wave and averaged."""
    per_wave = []
    for idx in wave_index:
        x = expected[:, idx, :]
        x = x - x.mean(axis=1, keepdims=True)
        target = x[:, :, 4:5]
        r = (x[:, :, :4] * target).sum(1) / np.sqrt((x[:, :, :4] ** 2).sum(1) * (target ** 2).sum(1))
        per_wave.append(np.abs(r).mean(axis=1))
    per_wave = np.stack(per_wave, axis=1)
    return per_wave.mean(axis=1), per_wave


def factorial_effects(outcome: np.ndarray, cell_levels: list[tuple[int, int, int]]) -> dict[str, float]:
    """Main effects average over the other factors; interactions are differences
    of simple effects (AB = effect of A at B=1 minus at B=0, averaged over C;
    ABC = the AB difference at C=1 minus at C=0), not halved."""
    y = np.zeros((2, 2, 2))
    for value, (a, b, c) in zip(outcome, cell_levels):
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


def item_metrics(cells, probs, mass, profiles, wave_names, wave_index, matched, human):
    """Part 1 metrics and item-level secondary outcomes for every cell, wave, and item."""
    human_mean, human_var, human_marginal = human
    rows = []
    for c, cell in enumerate(cells):
        for w, idx in zip(wave_names, wave_index):
            for j, item in enumerate(i.lower() for i in item_ids()):
                dists = [as_dist(probs[c, p, j]) for p in idx]
                pop = mixture(dists)
                decomposition = variance_decomposition(dists)
                answers = [matched[profiles[p][1]][item] for p in idx]
                observed = [(d, int(a)) for d, a in zip(dists, answers) if a in VALID]
                obs_dist = {k: sum(o == k for _, o in observed) / len(observed) for k in SCALE}
                key = (w, item)
                rows.append({
                    "cell": cell, "wave": w, "item": item, "profiles": len(idx),
                    "matched_answers": len(observed),
                    "mixture_mean": expected_score(pop),
                    "human_mean": human_mean[key],
                    "mean_error": expected_score(pop) - human_mean[key],
                    "total_variation": total_variation(pop, human_marginal[key]),
                    **decomposition,
                    "human_variance": human_var[key],
                    "between_variance_ratio": decomposition["between_profile_variance"] / human_var[key],
                    "total_variance_ratio": decomposition["total_predictive_variance"] / human_var[key],
                    "matched_expected_mae": float(np.mean([abs(expected_score(d) - o) for d, o in observed])),
                    "matched_rps": float(np.mean([ranked_probability_score(d, o, SCALE) for d, o in observed])),
                    "matched_log_score": float(np.mean([log_score(d, o) for d, o in observed])),
                    "matched_total_variation": total_variation(pop, obs_dist),
                    "mean_valid_option_mass": float(mass[c, idx, j].mean()),
                })
    return rows


def cell_summary(rows: list[dict], cell: str) -> dict:
    rows = [r for r in rows if r["cell"] == cell]
    mass = float(np.mean([r["mean_valid_option_mass"] for r in rows]))
    return {
        "cell": cell,
        "mean_abs_mean_error": float(np.mean([abs(r["mean_error"]) for r in rows])),
        "mean_total_variation": float(np.mean([r["total_variation"] for r in rows])),
        "mean_between_variance_ratio": float(np.mean([r["between_variance_ratio"] for r in rows])),
        "mean_within_share": float(np.mean([r["within_share"] for r in rows])),
        "mean_matched_rps": float(np.mean([r["matched_rps"] for r in rows])),
        "mean_matched_log_score": float(np.mean([r["matched_log_score"] for r in rows])),
        "mean_valid_option_mass": mass,
        "flag_mass_below_0_90": mass < 0.90,
    }


def hgb_expected_scores(path: Path, profiles: list[tuple[str, str]]) -> np.ndarray:
    """Out-of-fold HGB expected scores (equality coding) for the matched profiles."""
    position = {pid: i for i, (_, pid) in enumerate(profiles)}
    expected = np.full((len(profiles), 5), np.nan)
    for row in read_csv(path):
        if (row["split"] == "oof" and row["model"] == "hist_gradient_boosting"
                and row["profile_id"] in position):
            j = int(row["item"][-1]) - 1
            expected[position[row["profile_id"]], j] = sum(
                k * float(row[f"p{k}"]) for k in SCALE
            )
    return expected


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger = Path(args.ledger)
    cells = [cell.name for cell in CELLS]
    records, profiles, probs, mass = load_ledger(ledger, cells)
    waves = np.array([wave for wave, _ in profiles])
    wave_names = sorted(set(waves))
    wave_index = [np.flatnonzero(waves == w) for w in wave_names]

    human_dir = Path(args.human_dir)
    moments = read_csv(human_dir / "human_moments.csv")
    human_mean = {(r["wave"], r["item"]): float(r["mean"]) for r in moments}
    human_var = {(r["wave"], r["item"]): float(r["variance"]) for r in moments}
    human_marginal: dict[tuple[str, str], dict[int, float]] = defaultdict(dict)
    for r in read_csv(human_dir / "human_marginals.csv"):
        human_marginal[(r["wave"], r["item"])][int(r["option"])] = float(r["probability"])
    human_corr = {(r["wave"], r["item_x"], r["item_y"]): float(r["correlation"])
                  for r in read_csv(human_dir / "human_correlations.csv")}
    human = (human_mean, human_var, human_marginal)

    matched = {r["profile_id"]: r for r in read_csv(Path(args.matched))}
    missing = [pid for _, pid in profiles if pid not in matched]
    if missing:
        raise SystemExit(f"{len(missing)} ledger profiles have no matched respondent")

    v1_decomp = {(r["wave"], r["item"]): r for r in read_csv(Path(args.v1_dir) / "metrics_variance_decomposition.csv")}
    v1_dist = {(r["wave"], r["item"]): r for r in read_csv(Path(args.v1_dir) / "metrics_distribution.csv")
               if r["condition"] == "neutral_verbal"}
    ml = {(r["model"], r["wave"], r["item"].removeprefix("eq_")): r for r in read_csv(Path(args.ml_matched))}

    item_rows = item_metrics(cells, probs, mass, profiles, wave_names, wave_index, matched, human)

    label = None
    if args.exploratory_ledger:
        n_label, _, label_probs, label_mass = load_ledger(
            Path(args.exploratory_ledger), [LABEL_CELL], profiles
        )
        item_rows += item_metrics([LABEL_CELL], label_probs, label_mass, profiles,
                                  wave_names, wave_index, matched, human)
        label = (n_label, label_probs)
    write_csv(out / "v2_item_metrics_by_cell.csv", item_rows)

    # Part 1: reference cell, with v1 and survey-trained comparators.
    comparison = []
    for r in (r for r in item_rows if r["cell"] == REFERENCE_CELL):
        key = (r["wave"], r["item"])
        row = {"wave": r["wave"], "item": r["item"]}
        for metric in ("mean_error", "total_variation", "within_share",
                       "between_variance_ratio", "total_variance_ratio",
                       "matched_expected_mae", "matched_rps", "matched_log_score",
                       "matched_total_variation"):
            row[f"v2_{metric}"] = r[metric]
        row["v1_mean_error"] = float(v1_dist[key]["mean_error"])
        row["v1_total_variation"] = float(v1_dist[key]["total_variation"])
        row["v1_within_share"] = float(v1_decomp[key]["within_share"])
        row["v1_between_variance_ratio"] = float(v1_decomp[key]["between_variance_ratio"])
        row["v1_total_variance_ratio"] = float(v1_decomp[key]["total_variance_ratio"])
        for prefix, model in (("v1", "qwen3_5_9b_neutral"), ("hgb", "hist_gradient_boosting")):
            source = ml[(model, r["wave"], r["item"])]
            row[f"{prefix}_matched_expected_mae"] = float(source["ordinal_expected_mae"])
            row[f"{prefix}_matched_rps"] = float(source["ranked_probability_score"])
            row[f"{prefix}_matched_log_score"] = float(source["log_loss"])
            row[f"{prefix}_matched_total_variation"] = float(source["total_variation"])
        comparison.append(row)
    write_csv(out / "v2_part1_reference_cell.csv", comparison)
    part1_summary = []
    for key in (k for k in comparison[0] if k not in ("wave", "item")):
        absolute = key.endswith("mean_error")
        values = [abs(r[key]) if absolute else r[key] for r in comparison]
        part1_summary.append({"metric": ("abs_" if absolute else "") + key,
                              "mean_over_15_wave_items": float(np.mean(values))})
    write_csv(out / "v2_part1_summary.csv", part1_summary)

    # Part 2: primary outcome, factorial effects, profile-cluster bootstrap.
    scale = np.array(SCALE, dtype=float)
    expected = probs @ scale
    levels = [(int(c.battery), int(c.history), int(c.json_format)) for c in CELLS]
    outcome, outcome_by_wave = primary_outcome(expected, wave_index)
    estimates = factorial_effects(outcome, levels)
    label_expected = label[1] @ scale if label else None
    label_outcome, label_by_wave = (primary_outcome(label_expected, wave_index)
                                    if label else (None, None))

    rng = np.random.default_rng(args.seed)
    boot_outcome = np.empty((args.replicates, len(cells)))
    boot_effects = np.empty((args.replicates, len(EFFECTS)))
    boot_label = np.empty(args.replicates)
    for b in range(args.replicates):
        draw = [rng.choice(idx, size=len(idx), replace=True) for idx in wave_index]
        boot_outcome[b], _ = primary_outcome(expected, draw)
        effects = factorial_effects(boot_outcome[b], levels)
        boot_effects[b] = [effects[name] for name in EFFECTS]
        if label:
            boot_label[b] = primary_outcome(label_expected, draw)[0][0]

    primary_rows = []
    for c, cell in enumerate(cells):
        low, high = np.percentile(boot_outcome[:, c], [2.5, 97.5])
        primary_rows.append({
            "cell": cell, "battery_visible": levels[c][0], "answer_history": levels[c][1],
            "json_output": levels[c][2], "a425_mean_abs_correlation": outcome[c],
            "ci95_low": low, "ci95_high": high,
            **{f"wave_{w}": outcome_by_wave[c, k] for k, w in enumerate(wave_names)},
        })
    if label:
        low, high = np.percentile(boot_label, [2.5, 97.5])
        primary_rows.append({
            "cell": f"{LABEL_CELL} (exploratory)", "battery_visible": 1, "answer_history": 1,
            "json_output": 1, "a425_mean_abs_correlation": label_outcome[0],
            "ci95_low": low, "ci95_high": high,
            **{f"wave_{w}": label_by_wave[0, k] for k, w in enumerate(wave_names)},
        })
    if args.ml_predictions:
        hgb = hgb_expected_scores(Path(args.ml_predictions), profiles)
        per_wave = []
        for idx in wave_index:
            x = hgb[idx][~np.isnan(hgb[idx]).any(axis=1)]
            per_wave.append(np.mean([abs(np.corrcoef(x[:, 4], x[:, j])[0, 1]) for j in range(4)]))
        primary_rows.append({
            "cell": "HGB expected scores (exploratory)", "battery_visible": "",
            "answer_history": "", "json_output": "",
            "a425_mean_abs_correlation": float(np.mean(per_wave)), "ci95_low": "", "ci95_high": "",
            **{f"wave_{w}": per_wave[k] for k, w in enumerate(wave_names)},
        })
    human_a425 = [np.mean([abs(human_corr[(w, f"a42{j}", "a425")]) for j in range(1, 5)])
                  for w in wave_names]
    primary_rows.append({
        "cell": "CGSS weighted (response level)", "battery_visible": "", "answer_history": "",
        "json_output": "", "a425_mean_abs_correlation": float(np.mean(human_a425)),
        "ci95_low": "", "ci95_high": "",
        **{f"wave_{w}": human_a425[k] for k, w in enumerate(wave_names)},
    })
    write_csv(out / "v2_part2_primary_by_cell.csv", primary_rows)

    effect_rows = []
    for k, name in enumerate(EFFECTS):
        low, high = np.percentile(boot_effects[:, k], [2.5, 97.5])
        effect_rows.append({"effect": name, "estimate": estimates[name], "ci95_low": low,
                            "ci95_high": high, "interval_excludes_zero": bool(low > 0 or high < 0)})
    write_csv(out / "v2_part2_effects.csv", effect_rows)
    secondary = [cell_summary(item_rows, cell) for cell in cells]
    write_csv(out / "v2_part2_secondary_by_cell.csv", secondary)

    if label:
        reference = cell_summary(item_rows, REFERENCE_CELL)
        labelled = cell_summary(item_rows, LABEL_CELL)
        diff = boot_label - boot_outcome[:, cells.index(REFERENCE_CELL)]
        low, high = np.percentile(diff, [2.5, 97.5])
        rows = [{"metric": "a425_mean_abs_correlation", LABEL_CELL: label_outcome[0],
                 REFERENCE_CELL: outcome[cells.index(REFERENCE_CELL)],
                 "difference": label_outcome[0] - outcome[cells.index(REFERENCE_CELL)],
                 "diff_ci95_low": low, "diff_ci95_high": high}]
        for key in (k for k in labelled if k not in ("cell", "flag_mass_below_0_90")):
            rows.append({"metric": key, LABEL_CELL: labelled[key], REFERENCE_CELL: reference[key],
                         "difference": labelled[key] - reference[key],
                         "diff_ci95_low": "", "diff_ci95_high": ""})
        write_csv(out / "v2_exploratory_label_cell.csv", rows)

    metadata = {
        "ledger_sha256": sha256(ledger), "records": records, "profiles": len(profiles),
        "profiles_per_wave": {w: int(len(i)) for w, i in zip(wave_names, wave_index)},
        "reference_cell": REFERENCE_CELL, "bootstrap_replicates": args.replicates,
        "bootstrap_seed": args.seed,
        "bootstrap": "profiles resampled with replacement within wave; all cells and items move together",
        "interaction_definition": "differences of simple effects, not halved",
    }
    if label:
        metadata["exploratory_ledger_sha256"] = sha256(Path(args.exploratory_ledger))
        metadata["exploratory_records"] = label[0]
    (out / "v2_analysis_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Part 1, reference cell {REFERENCE_CELL} (mean over 15 wave-items):")
    for row in part1_summary:
        print(f"  {row['metric']:38s} {row['mean_over_15_wave_items']:.3f}")
    print("Part 2, A425 mean |r| by cell:")
    for row in primary_rows:
        print(f"  {row['cell']:36s} {row['a425_mean_abs_correlation']:.3f}")
    print("Part 2, effects:")
    for row in effect_rows:
        print(f"  {row['effect']:4s} {row['estimate']:+.3f} [{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
