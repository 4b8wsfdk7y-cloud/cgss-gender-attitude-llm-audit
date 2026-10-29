#!/usr/bin/env python3
# requires: Python 3.10+, the packages in ml/requirements.txt
"""Survey-trained HGB with answer history (PREREGISTRATION_V2.md, 2026-10-29, 3b).

For A422-A425, fits the first audit's weighted histogram gradient boosting
model with its features and hyperparameters plus the respondent's own answers
to the earlier items (egalitarian coding), five-fold out of fold on all CGSS
respondents with valid answers on those items. Predictions for the matched
profiles are profile-linked and are written outside the repository; aggregate
out-of-fold metrics are written to the output directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from sklearn.model_selection import StratifiedKFold

REPO_ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "train_evaluate", REPO_ROOT / "ml" / "scripts" / "02_train_evaluate.py")
run02 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run02)

MODEL = "hist_gradient_boosting"
DEFAULT_PRIVATE = Path.home() / "Projects/restricted-data/surveyllm-v2/ml_history"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="human_ml_input.csv")
    parser.add_argument("--config", default=str(REPO_ROOT / "ml" / "config.json"))
    parser.add_argument("--private-dir", default=str(DEFAULT_PRIVATE))
    parser.add_argument("--output-dir", required=True, help="aggregate outputs")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    private = Path(args.private_dir).expanduser()
    if REPO_ROOT in private.resolve().parents or private.resolve() == REPO_ROOT:
        raise SystemExit("Profile-linked outputs must be written outside the repository")
    private.mkdir(parents=True, exist_ok=True)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    data_path = Path(args.data).expanduser()
    data = pd.read_csv(data_path)
    outcomes = config["outcomes"]                      # eq_a421 ... eq_a425

    prediction_rows, metric_rows = [], []
    for outcome_index, outcome in enumerate(outcomes, start=1):
        if outcome_index == 1:
            continue
        history = outcomes[: outcome_index - 1]
        item_config = dict(config, numeric_features=config["numeric_features"] + history)
        item_data = data.dropna(subset=[outcome, *history]).reset_index(drop=True)
        y = item_data[outcome].astype(int).to_numpy()
        splitter = StratifiedKFold(n_splits=config["folds"], shuffle=True,
                                   random_state=config["seed"] + outcome_index)
        probabilities = np.zeros((len(item_data), 5))
        for fold, (train_index, test_index) in enumerate(splitter.split(item_data, y), start=1):
            probabilities[test_index] = run02.fit_predict(
                MODEL, item_config, item_data.iloc[train_index], item_data.iloc[test_index], outcome)
            print(f"{outcome} with history: fold {fold}/{config['folds']}", flush=True)
        for wave, index in item_data.groupby("wave").groups.items():
            positions = np.asarray(list(index), dtype=int)
            metric_rows.append({
                "wave": str(wave), "item": outcome.removeprefix("eq_"), "model": f"{MODEL}_with_history",
                "history_items": "+".join(h.removeprefix("eq_") for h in history),
                "n": len(positions),
                **run02.prediction_metrics(
                    y[positions], probabilities[positions],
                    item_data.loc[positions, "analysis_weight"].to_numpy(dtype=float)),
            })
        matched = item_data["is_llm_profile"].to_numpy() == 1
        rows = item_data.loc[matched, ["wave", "profile_id", outcome]].rename(columns={outcome: "observed"})
        rows.insert(2, "item", outcome.removeprefix("eq_"))
        for k in range(5):
            rows[f"p{k + 1}"] = probabilities[matched, k]
        prediction_rows.append(rows)

    predictions = pd.concat(prediction_rows, ignore_index=True)
    predictions_path = private / "hgb_with_history_matched.csv"
    predictions.to_csv(predictions_path, index=False)
    pd.DataFrame(metric_rows).to_csv(out / "exploratory_hgb_history_oof_metrics.csv", index=False)
    metadata = {
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "config": config,
        "model": MODEL,
        "coding": "egalitarian (eq_*), history features are the earlier eq_* items",
        "matched_predictions": len(predictions),
        "matched_predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "versions": {"scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    (out / "exploratory_hgb_history_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"matched predictions: {len(predictions)} rows -> {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
