#!/usr/bin/env python3
# requires: Python 3.10+, the packages in ml/requirements.txt
"""Survey-trained HGB for the trust replication (PREREGISTRATION_V2.md, 2026-11-17, point 8).

For each of the eight CGSS 2012 trust items, fits the first audit's weighted
histogram gradient boosting model with the features and hyperparameters of
ml/config.json, five-fold out of fold on the eligible respondents written by
scripts/28_export_trust_ml_input.R. Only demographic features are used. The
out-of-fold predictions for the 300 matched profiles are profile-linked and are
written outside the repository; aggregate out-of-fold metrics are written to the
output directory.
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
ITEMS = ("n1601", "n1603", "n1605", "n1606", "n1611", "n1612", "n1613", "n1614")
DEFAULT_PRIVATE = Path.home() / "Projects/restricted-data/surveyllm-v2-trust/hgb"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True, help="trust_ml_input.csv")
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
    data = pd.read_csv(data_path, dtype={"prov": str, "profile_id": str})
    if len(data) != 5595 or int(data["is_llm_profile"].sum()) != 300:
        raise SystemExit("Unexpected input: expected 5,595 respondents and 300 matched profiles")

    matched = data["is_llm_profile"].to_numpy() == 1
    weights = data["analysis_weight"].to_numpy(dtype=float)
    prediction = data.loc[matched, ["profile_id"]].reset_index(drop=True)
    metric_rows = []
    for item_index, item in enumerate(ITEMS, start=1):
        outcome = f"t_{item}"
        y = data[outcome].astype(int).to_numpy()
        splitter = StratifiedKFold(n_splits=config["folds"], shuffle=True,
                                   random_state=config["seed"] + item_index)
        probabilities = np.zeros((len(data), 5))
        for fold, (train_index, test_index) in enumerate(splitter.split(data, y), start=1):
            probabilities[test_index] = run02.fit_predict(
                MODEL, config, data.iloc[train_index], data.iloc[test_index], outcome)
            print(f"{item}: fold {fold}/{config['folds']}", flush=True)
        if probabilities[:, 4].max() > 1e-12:
            raise SystemExit(f"{item}: probability on option 5 of a four-point item")
        metrics = run02.prediction_metrics(y, probabilities, weights)
        metric_rows.append({"item": item.upper(), "model": MODEL, "n": len(data),
                            "log_loss": metrics["log_loss"],
                            "ranked_probability_score": metrics["ranked_probability_score"],
                            "ordinal_expected_mae": metrics["ordinal_expected_mae"]})
        for k in range(4):
            prediction[f"{item.upper()}_p{k + 1}"] = probabilities[matched, k]

    predictions_path = private / "hgb_trust_matched.csv"
    prediction.to_csv(predictions_path, index=False)
    pd.DataFrame(metric_rows).to_csv(out / "trust_hgb_oof_metrics.csv", index=False)
    metadata = {
        "data_sha256": hashlib.sha256(data_path.read_bytes()).hexdigest(),
        "config": config,
        "model": MODEL,
        "features": config["numeric_features"] + config["categorical_features"],
        "coding": "trust items 1-4, higher = more trust",
        "respondents": len(data),
        "matched_profiles": int(matched.sum()),
        "matched_predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "versions": {"scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    (out / "trust_hgb_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"matched predictions: {len(prediction)} profiles -> {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
