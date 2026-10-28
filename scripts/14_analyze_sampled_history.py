#!/usr/bin/env python3
# requires: Python 3.10+, numpy, the survey_llm_eval package
"""Sampled-history robustness check (PREREGISTRATION_V2.md, 2026-10-28 entry).

Inputs are the primary (modal-history) ledger and the sampled-history ledger
written by 07_run_logprob_audit.py --history sampled --paths 5. The answer drawn
on each path is rebuilt by replaying the seeded draws, which must reproduce the
history recorded for every later item. Cells without history get their draws
from the primary distributions with the same seeding scheme. Reported:

- R1: the primary outcome on drawn answers (response level), by cell, with
  factorial effects;
- R2: the primary outcome on expected scores conditional on each sampled
  history, averaged over paths, its difference from the modal-history value,
  and the factorial effects with the history cells replaced;
- secondary, descriptive: item-mean error and between-profile variance ratio
  from each profile's option distribution averaged over its paths.

Only aggregate tables are written.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
import warnings
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "analyze_logprob_audit", REPO_ROOT / "scripts" / "09_analyze_logprob_audit.py")
run09 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run09)

from survey_llm_eval.factorial import CELLS, item_ids  # noqa: E402

SCALE = run09.SCALE
EFFECTS = run09.EFFECTS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True, help="option_probabilities_greedy.jsonl")
    parser.add_argument("--sampled-ledger", required=True, help="option_probabilities_sampled.jsonl")
    parser.add_argument("--matched", required=True, help="profiles_pilot.csv (matched answers)")
    parser.add_argument("--human-dir", required=True, help="output of 08_export_human_benchmark.R")
    parser.add_argument("--v1-output", default=str(REPO_ROOT / "output"),
                        help="directory with the v1 aggregate metrics_*.csv files")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--paths", type=int, default=5)
    parser.add_argument("--draw-seed", type=int, default=20261013,
                        help="seed passed to 07_run_logprob_audit.py --seed")
    parser.add_argument("--replicates", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261013, help="bootstrap seed")
    return parser.parse_args()


def draw(rng: random.Random, distribution: dict[int, float]) -> int:
    """The draw made by 07_run_logprob_audit.choose_history_answer in sampled mode."""
    options = sorted(distribution)
    return rng.choices(options, weights=[distribution[o] for o in options], k=1)[0]


def load_sampled(path: Path, cells: list[str], profiles, paths: int, seed: int):
    """Probabilities (cell, profile, path, item, option), valid mass, and drawn answers.

    Raises if the ledger is incomplete or if replaying the seeded draws does not
    reproduce a recorded history.
    """
    items = list(item_ids())
    position = {pid: i for i, (_, pid) in enumerate(profiles)}
    shape = (len(cells), len(profiles), paths, len(items))
    probs = np.full(shape + (len(SCALE),), np.nan)
    mass = np.full(shape, np.nan)
    rows: dict[tuple[int, int, int, int], dict] = {}
    records = 0
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r["history_mode"] != "sampled" or r["cell"] not in cells:
            continue
        key = (cells.index(r["cell"]), position[r["profile_id"]], r["path"], items.index(r["item"]))
        rows[key] = r
        records += 1
    if len(rows) != np.prod(shape):
        raise SystemExit(f"{path} has {len(rows)} of {int(np.prod(shape))} sampled records")
    answers = np.zeros(shape, dtype=int)
    for c, cell in enumerate(cells):
        for p, (_, pid) in enumerate(profiles):
            for k in range(paths):
                rng = random.Random(f"{seed}|{pid}|{cell}|{k}")
                drawn: list[int] = []
                for j in range(len(items)):
                    r = rows[(c, p, k, j)]
                    if r["history"] != drawn:
                        raise SystemExit(f"replayed history differs: {pid} {cell} path {k} item {j}")
                    distribution = {int(o): v for o, v in r["probabilities"].items()}
                    probs[c, p, k, j] = [distribution[o] for o in SCALE]
                    mass[c, p, k, j] = r["valid_option_mass"]
                    drawn.append(draw(rng, distribution))
                answers[c, p, k] = drawn
    return records, probs, mass, answers


def draws_without_history(path: Path, cells: list[str], profiles, paths: int, seed: int) -> np.ndarray:
    """Drawn answers (cell, profile, path, item) for cells whose prompts have no history."""
    items = list(item_ids())
    position = {pid: i for i, (_, pid) in enumerate(profiles)}
    dist: dict[tuple[int, int, int], dict[int, float]] = {}
    for line in path.open(encoding="utf-8"):
        r = json.loads(line)
        if r["cell"] in cells:
            dist[(cells.index(r["cell"]), position[r["profile_id"]], items.index(r["item"]))] = {
                int(o): v for o, v in r["probabilities"].items()}
    answers = np.zeros((len(cells), len(profiles), paths, len(items)), dtype=int)
    for c, cell in enumerate(cells):
        for p, (_, pid) in enumerate(profiles):
            for k in range(paths):
                rng = random.Random(f"{seed}|{pid}|{cell}|{k}")
                answers[c, p, k] = [draw(rng, dist[(c, p, j)]) for j in range(len(items))]
    return answers


def response_level(answers: np.ndarray, wave_index: list[np.ndarray]):
    """R1 by cell: mean |r| of drawn A425 with drawn A421-A424 across profiles,
    per path and wave; undefined correlations are left out. Returns the
    path-and-wave average, the per-wave values, and the number of undefined
    correlations."""
    per_wave, undefined = [], 0
    with warnings.catch_warnings(), np.errstate(invalid="ignore", divide="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        for idx in wave_index:
            x = answers[:, idx].astype(float)                   # cell, profile, path, item
            x = x - x.mean(axis=1, keepdims=True)
            target = x[..., 4:5]
            num = (x[..., :4] * target).sum(axis=1)             # cell, path, item
            den = np.sqrt((x[..., :4] ** 2).sum(axis=1) * (target ** 2).sum(axis=1))
            r = np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)
            undefined += int(np.isnan(r).sum())
            per_path = np.nanmean(np.abs(r), axis=2)           # cell, path
            per_wave.append(np.nanmean(per_path, axis=1))
    per_wave = np.stack(per_wave, axis=1)
    return np.nanmean(per_wave, axis=1), per_wave, undefined


def sampled_expected(expected: np.ndarray, wave_index: list[np.ndarray]):
    """R2 by history cell: primary outcome per path from expected scores, averaged over paths."""
    n_cells, n_profiles, paths, n_items = expected.shape
    flat = expected.transpose(0, 2, 1, 3).reshape(n_cells * paths, n_profiles, n_items)
    outcome, by_wave = run09.primary_outcome(flat, wave_index)
    return (outcome.reshape(n_cells, paths).mean(axis=1),
            by_wave.reshape(n_cells, paths, -1).mean(axis=1))


def interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.nanpercentile(values, [2.5, 97.5])
    return float(low), float(high)


def main() -> int:
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    ledger, sampled_ledger = Path(args.ledger), Path(args.sampled_ledger)

    cells = [cell.name for cell in CELLS]
    levels = [(int(c.battery), int(c.history), int(c.json_format)) for c in CELLS]
    history_cells = [c.name for c in CELLS if c.history]
    plain_cells = [c.name for c in CELLS if not c.history]
    h_pos = [cells.index(c) for c in history_cells]
    p_pos = [cells.index(c) for c in plain_cells]

    records, profiles, probs, mass = run09.load_ledger(ledger, cells)
    waves = np.array([wave for wave, _ in profiles])
    wave_names = sorted(set(waves))
    wave_index = [np.flatnonzero(waves == w) for w in wave_names]
    scale = np.array(SCALE, dtype=float)
    modal_expected = probs @ scale

    n_sampled, s_probs, s_mass, s_answers = load_sampled(
        sampled_ledger, history_cells, profiles, args.paths, args.draw_seed)
    answers = np.zeros((len(cells), len(profiles), args.paths, len(item_ids())), dtype=int)
    answers[h_pos] = s_answers
    answers[p_pos] = draws_without_history(ledger, plain_cells, profiles, args.paths, args.draw_seed)
    s_expected = s_probs @ scale

    human_dir = Path(args.human_dir)
    moments = run09.read_csv(human_dir / "human_moments.csv")
    human_mean = {(r["wave"], r["item"]): float(r["mean"]) for r in moments}
    human_var = {(r["wave"], r["item"]): float(r["variance"]) for r in moments}
    human_marginal: dict[tuple[str, str], dict[int, float]] = {}
    for r in run09.read_csv(human_dir / "human_marginals.csv"):
        human_marginal.setdefault((r["wave"], r["item"]), {})[int(r["option"])] = float(r["probability"])
    human_corr = {(r["wave"], r["item_x"], r["item_y"]): float(r["correlation"])
                  for r in run09.read_csv(human_dir / "human_correlations.csv")}
    matched = {r["profile_id"]: r for r in run09.read_csv(Path(args.matched))}

    # Point estimates.
    modal, modal_by_wave = run09.primary_outcome(modal_expected, wave_index)
    r2_hist, r2_hist_by_wave = sampled_expected(s_expected, wave_index)
    r2 = modal.copy()
    r2[h_pos] = r2_hist
    r1, r1_by_wave, undefined = response_level(answers, wave_index)
    estimates = {
        "modal_expected": run09.factorial_effects(modal, levels),
        "sampled_expected": run09.factorial_effects(r2, levels),
        "response_level": run09.factorial_effects(r1, levels),
    }

    # Bootstrap: the same resampling scheme and seed as 09_analyze_logprob_audit.py.
    rng = np.random.default_rng(args.seed)
    boot = {name: np.empty((args.replicates, len(cells))) for name in estimates}
    boot_effects = {name: np.empty((args.replicates, len(EFFECTS))) for name in estimates}
    for b in range(args.replicates):
        idx = [rng.choice(i, size=len(i), replace=True) for i in wave_index]
        boot["modal_expected"][b] = run09.primary_outcome(modal_expected, idx)[0]
        boot["sampled_expected"][b] = boot["modal_expected"][b]
        boot["sampled_expected"][b, h_pos] = sampled_expected(s_expected, idx)[0]
        boot["response_level"][b] = response_level(answers, idx)[0]
        for name in estimates:
            effects = run09.factorial_effects(boot[name][b], levels)
            boot_effects[name][b] = [effects[e] for e in EFFECTS]

    cell_rows = []
    for c, cell in enumerate(cells):
        row = {"cell": cell, "battery_visible": levels[c][0], "answer_history": levels[c][1],
               "json_output": levels[c][2]}
        for name, value in (("modal_expected", modal), ("sampled_expected", r2),
                            ("response_level", r1)):
            low, high = interval(boot[name][:, c])
            row.update({name: float(value[c]), f"{name}_ci95_low": low, f"{name}_ci95_high": high})
        if c in h_pos:
            diff = boot["sampled_expected"][:, c] - boot["modal_expected"][:, c]
            low, high = interval(diff)
            row.update({"sampled_minus_modal": float(r2[c] - modal[c]),
                        "sampled_minus_modal_ci95_low": low, "sampled_minus_modal_ci95_high": high})
        else:
            row.update({"sampled_minus_modal": "", "sampled_minus_modal_ci95_low": "",
                        "sampled_minus_modal_ci95_high": ""})
        sampled_wave = r2_hist_by_wave[h_pos.index(c)] if c in h_pos else modal_by_wave[c]
        for k, w in enumerate(wave_names):
            row[f"modal_expected_{w}"] = float(modal_by_wave[c, k])
            row[f"sampled_expected_{w}"] = float(sampled_wave[k])
            row[f"response_level_{w}"] = float(r1_by_wave[c, k])
        cell_rows.append(row)
    run09.write_csv(out / "robustness_primary_by_cell.csv", cell_rows)

    effect_rows = []
    for name in estimates:
        for k, effect in enumerate(EFFECTS):
            low, high = interval(boot_effects[name][:, k])
            effect_rows.append({"outcome": name, "effect": effect,
                                "estimate": float(estimates[name][effect]),
                                "ci95_low": low, "ci95_high": high,
                                "interval_excludes_zero": bool(low > 0 or high < 0)})
    run09.write_csv(out / "robustness_effects.csv", effect_rows)

    # Response-level comparators, reported descriptively.
    human_a425 = [np.mean([abs(human_corr[(w, f"a42{j}", "a425")]) for j in range(1, 5)])
                  for w in wave_names]
    v1_output = Path(args.v1_output)
    envelope = {r["wave"]: r for r in run09.read_csv(v1_output / "metrics_human_sampling_envelope.csv")
                if r["metric"] == "a425_mean_abs_correlation"}
    v1_joint = {r["wave"]: float(r["joint_a425"])
                for r in run09.read_csv(v1_output / "metrics_coherence_comparison.csv")}
    comparators = [
        {"source": "CGSS weighted, full sample", "value": float(np.mean(human_a425)),
         **{f"wave_{w}": float(v) for w, v in zip(wave_names, human_a425)}},
        {"source": "CGSS samples of 100 per wave, median (v1 envelope)",
         "value": float(np.mean([float(envelope[w]["median"]) for w in wave_names])),
         **{f"wave_{w}": float(envelope[w]["median"]) for w in wave_names}},
        {"source": "CGSS samples of 100 per wave, 97.5th percentile (v1 envelope)",
         "value": float(np.mean([float(envelope[w]["p975"]) for w in wave_names])),
         **{f"wave_{w}": float(envelope[w]["p975"]) for w in wave_names}},
        {"source": "v1 joint calls, mean of five repeats",
         "value": float(np.mean([v1_joint[w] for w in wave_names])),
         **{f"wave_{w}": v1_joint[w] for w in wave_names}},
    ]
    run09.write_csv(out / "robustness_response_level_comparators.csv", comparators)

    # Secondary: each profile's distribution averaged over its paths.
    averaged = s_probs.mean(axis=2)
    averaged_mass = s_mass.mean(axis=2)
    human = (human_mean, human_var, human_marginal)
    modal_rows = run09.item_metrics(history_cells, probs[h_pos], mass[h_pos], profiles,
                                    wave_names, wave_index, matched, human)
    sampled_rows = run09.item_metrics(history_cells, averaged, averaged_mass, profiles,
                                      wave_names, wave_index, matched, human)
    item_rows = []
    for m, s in zip(modal_rows, sampled_rows):
        item_rows.append({"cell": m["cell"], "wave": m["wave"], "item": m["item"],
                          "human_mean": m["human_mean"],
                          "modal_mean": m["mixture_mean"], "sampled_mean": s["mixture_mean"],
                          "modal_between_variance_ratio": m["between_variance_ratio"],
                          "sampled_between_variance_ratio": s["between_variance_ratio"],
                          "sampled_mean_valid_option_mass": s["mean_valid_option_mass"]})
    run09.write_csv(out / "robustness_item_metrics.csv", item_rows)
    secondary = []
    for cell in history_cells:
        m = run09.cell_summary(modal_rows, cell)
        s = run09.cell_summary(sampled_rows, cell)
        secondary.append({"cell": cell,
                          "modal_mean_abs_mean_error": m["mean_abs_mean_error"],
                          "sampled_mean_abs_mean_error": s["mean_abs_mean_error"],
                          "modal_mean_between_variance_ratio": m["mean_between_variance_ratio"],
                          "sampled_mean_between_variance_ratio": s["mean_between_variance_ratio"],
                          "sampled_mean_valid_option_mass": s["mean_valid_option_mass"],
                          "flag_mass_below_0_90": s["flag_mass_below_0_90"]})
    run09.write_csv(out / "robustness_secondary_by_cell.csv", secondary)

    metadata = {
        "ledger_sha256": run09.sha256(ledger), "ledger_records": records,
        "sampled_ledger_sha256": run09.sha256(sampled_ledger), "sampled_records": n_sampled,
        "profiles": len(profiles), "paths": args.paths, "draw_seed": args.draw_seed,
        "history_cells": history_cells,
        "replay_check": "every recorded history equals the replayed seeded draws",
        "undefined_response_level_correlations": undefined,
        "bootstrap_replicates": args.replicates, "bootstrap_seed": args.seed,
        "bootstrap": "profiles resampled with replacement within wave; all paths, items, and cells move together",
    }
    (out / "robustness_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print("A425 mean |r| by cell: modal expected / sampled expected / response level")
    for row in cell_rows:
        print(f"  {row['cell']:9s} {row['modal_expected']:.3f}  {row['sampled_expected']:.3f}  "
              f"{row['response_level']:.3f} [{row['response_level_ci95_low']:.3f}, "
              f"{row['response_level_ci95_high']:.3f}]")
    for row in effect_rows:
        print(f"  {row['outcome']:17s} {row['effect']:4s} {row['estimate']:+.3f} "
              f"[{row['ci95_low']:+.3f}, {row['ci95_high']:+.3f}]")
    print(f"undefined response-level correlations: {undefined}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
