#!/usr/bin/env python3
# requires: numpy, pandas, scikit-learn, matplotlib

from __future__ import annotations

import json
import math
import warnings
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


SCRIPT_PATH = Path(__file__).resolve()
ML_ROOT = SCRIPT_PATH.parents[1]
AUDIT_ROOT = ML_ROOT.parent
CONFIG_PATH = ML_ROOT / "config.json"
DATA_PATH = ML_ROOT / "data" / "human_ml_input.csv"
LLM_PATH = AUDIT_ROOT / "output" / "responses.csv"
OUTPUT_DIR = ML_ROOT / "output"
FIGURE_DIR = OUTPUT_DIR / "figures"
REPORT_DIR = ML_ROOT / "report"
CLASSES = np.arange(1, 6)

warnings.filterwarnings(
    "ignore",
    message=r".*encountered in matmul",
    category=RuntimeWarning,
    module=r"sklearn\.utils\.extmath",
)


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    keep = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    return float(np.average(values[keep], weights=weights[keep]))


def weighted_class_distribution(
    y: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    mass = np.array(
        [weights[y == value].sum() for value in CLASSES], dtype=float
    )
    return mass / mass.sum()


def align_probabilities(
    probabilities: np.ndarray, observed_classes: Iterable[int]
) -> np.ndarray:
    aligned = np.zeros((probabilities.shape[0], len(CLASSES)), dtype=float)
    for source_index, value in enumerate(observed_classes):
        aligned[:, int(value) - 1] = probabilities[:, source_index]
    return aligned


def validate_probabilities(probabilities: np.ndarray) -> None:
    if not np.isfinite(probabilities).all():
        raise ValueError("Predicted probabilities contain non-finite values.")
    if probabilities.min() < -1e-12 or probabilities.max() > 1 + 1e-12:
        raise ValueError("Predicted probabilities fall outside 0..1.")
    if not np.allclose(probabilities.sum(axis=1), 1, atol=1e-8):
        raise ValueError("Predicted probabilities do not sum to one.")


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    return weights / np.mean(weights)


def build_preprocessor(
    numeric_features: list[str], categorical_features: list[str]
) -> ColumnTransformer:
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, numeric_features),
            ("categorical", categorical, categorical_features),
        ],
        sparse_threshold=0,
    )


def build_model(name: str, config: dict) -> Pipeline:
    preprocessor = build_preprocessor(
        config["numeric_features"], config["categorical_features"]
    )
    if name == "multinomial_logit":
        estimator = LogisticRegression(
            C=config["logistic"]["C"],
            max_iter=config["logistic"]["max_iter"],
            solver=config["logistic"]["solver"],
            random_state=config["seed"],
        )
    elif name == "hist_gradient_boosting":
        options = config["hist_gradient_boosting"]
        estimator = HistGradientBoostingClassifier(
            learning_rate=options["learning_rate"],
            max_iter=options["max_iter"],
            max_leaf_nodes=options["max_leaf_nodes"],
            min_samples_leaf=options["min_samples_leaf"],
            l2_regularization=options["l2_regularization"],
            early_stopping=True,
            random_state=config["seed"],
        )
    else:
        raise ValueError(f"Unknown model: {name}")
    return Pipeline([("preprocess", preprocessor), ("model", estimator)])


def fit_predict(
    name: str,
    config: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    outcome: str,
) -> np.ndarray:
    if name == "weighted_prior":
        return np.tile(
            weighted_class_distribution(
                train[outcome].to_numpy(dtype=int),
                train["analysis_weight"].to_numpy(dtype=float),
            ),
            (len(test), 1),
        )
    model = build_model(name, config)
    features = config["numeric_features"] + config["categorical_features"]
    model.fit(
        train[features],
        train[outcome].astype(int),
        model__sample_weight=normalize_weights(
            train["analysis_weight"].to_numpy(dtype=float)
        ),
    )
    probabilities = model.predict_proba(test[features])
    probabilities = align_probabilities(
        probabilities, model.named_steps["model"].classes_
    )
    validate_probabilities(probabilities)
    return probabilities


def prediction_metrics(
    y: np.ndarray, probabilities: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    validate_probabilities(probabilities)
    expected = np.sum(probabilities * CLASSES[None, :], axis=1)
    argmax = CLASSES[np.argmax(probabilities, axis=1)]
    observed_distribution = weighted_class_distribution(y, weights)
    predicted_distribution = np.average(
        probabilities, axis=0, weights=weights
    )
    observed_mean = weighted_mean(y.astype(float), weights)
    predicted_mean = float(predicted_distribution @ CLASSES)
    observed_variance = weighted_mean(
        (y.astype(float) - observed_mean) ** 2, weights
    )
    predicted_second_moment = float(
        predicted_distribution @ (CLASSES.astype(float) ** 2)
    )
    predicted_variance = predicted_second_moment - predicted_mean**2
    cumulative_predicted = np.cumsum(probabilities, axis=1)[:, :-1]
    cumulative_observed = (
        y[:, None] <= np.arange(1, 5)[None, :]
    ).astype(float)
    row_rps = np.mean(
        (cumulative_predicted - cumulative_observed) ** 2, axis=1
    )
    one_hot = np.eye(5)[y - 1]
    row_brier = np.mean((probabilities - one_hot) ** 2, axis=1)
    return {
        "n": int(len(y)),
        "ordinal_expected_mae": weighted_mean(abs(expected - y), weights),
        "ordinal_argmax_mae": weighted_mean(abs(argmax - y), weights),
        "log_loss": float(
            log_loss(y, probabilities, labels=CLASSES, sample_weight=weights)
        ),
        "ranked_probability_score": weighted_mean(row_rps, weights),
        "multiclass_brier": weighted_mean(row_brier, weights),
        "human_mean": observed_mean,
        "predicted_mean": predicted_mean,
        "mean_error": predicted_mean - observed_mean,
        "total_variation": float(
            0.5 * np.abs(predicted_distribution - observed_distribution).sum()
        ),
        "human_variance": observed_variance,
        "predicted_variance": predicted_variance,
        "variance_ratio": predicted_variance / observed_variance,
    }


def discrete_metrics(
    y: np.ndarray, prediction: np.ndarray, weights: np.ndarray
) -> dict[str, float]:
    observed_distribution = weighted_class_distribution(y, weights)
    predicted_distribution = weighted_class_distribution(prediction, weights)
    observed_mean = weighted_mean(y.astype(float), weights)
    predicted_mean = weighted_mean(prediction.astype(float), weights)
    observed_variance = weighted_mean((y - observed_mean) ** 2, weights)
    predicted_variance = weighted_mean(
        (prediction - predicted_mean) ** 2, weights
    )
    return {
        "n": int(len(y)),
        "ordinal_expected_mae": weighted_mean(abs(prediction - y), weights),
        "ordinal_argmax_mae": weighted_mean(abs(prediction - y), weights),
        "log_loss": math.nan,
        "ranked_probability_score": math.nan,
        "multiclass_brier": math.nan,
        "human_mean": observed_mean,
        "predicted_mean": predicted_mean,
        "mean_error": predicted_mean - observed_mean,
        "total_variation": float(
            0.5 * np.abs(predicted_distribution - observed_distribution).sum()
        ),
        "human_variance": observed_variance,
        "predicted_variance": predicted_variance,
        "variance_ratio": predicted_variance / observed_variance,
    }


def run_oof(
    data: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_names = [
        "weighted_prior",
        "multinomial_logit",
        "hist_gradient_boosting",
    ]
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for outcome_index, outcome in enumerate(config["outcomes"], start=1):
        item_data = data.dropna(subset=[outcome]).reset_index(drop=True)
        y = item_data[outcome].astype(int).to_numpy()
        splitter = StratifiedKFold(
            n_splits=config["folds"],
            shuffle=True,
            random_state=config["seed"] + outcome_index,
        )
        predictions = {
            name: np.zeros((len(item_data), 5), dtype=float)
            for name in model_names
        }
        for fold, (train_index, test_index) in enumerate(
            splitter.split(item_data, y), start=1
        ):
            train = item_data.iloc[train_index]
            test = item_data.iloc[test_index]
            for name in model_names:
                predictions[name][test_index] = fit_predict(
                    name, config, train, test, outcome
                )
            print(
                f"OOF {outcome}: fold {fold}/{config['folds']} complete",
                flush=True,
            )
        for name, probabilities in predictions.items():
            for wave, wave_index in item_data.groupby("wave").groups.items():
                positions = np.asarray(list(wave_index), dtype=int)
                metric_rows.append(
                    {
                        "split": "oof",
                        "wave": str(wave),
                        "item": outcome,
                        "model": name,
                        **prediction_metrics(
                            y[positions],
                            probabilities[positions],
                            item_data.loc[
                                positions, "analysis_weight"
                            ].to_numpy(dtype=float),
                        ),
                    }
                )
            output = item_data[
                [
                    "wave",
                    "row_id",
                    "profile_id",
                    "is_llm_profile",
                    "analysis_weight",
                    outcome,
                ]
            ].copy()
            output = output.rename(columns={outcome: "observed"})
            output.insert(0, "split", "oof")
            output.insert(3, "item", outcome)
            output.insert(4, "model", name)
            for index, value in enumerate(CLASSES):
                output[f"p{value}"] = probabilities[:, index]
            prediction_rows.append(output)
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(metric_rows)


def run_temporal(
    data: pd.DataFrame, config: dict
) -> tuple[pd.DataFrame, pd.DataFrame]:
    model_names = [
        "weighted_prior",
        "multinomial_logit",
        "hist_gradient_boosting",
    ]
    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict] = []
    for outcome in config["outcomes"]:
        train = data.loc[
            (data["survey_year"] < 2021) & data[outcome].notna()
        ].copy()
        test = data.loc[
            (data["survey_year"] == 2021) & data[outcome].notna()
        ].copy()
        y = test[outcome].astype(int).to_numpy()
        weights = test["analysis_weight"].to_numpy(dtype=float)
        for name in model_names:
            probabilities = fit_predict(name, config, train, test, outcome)
            metric_rows.append(
                {
                    "split": "train_2012_2018_test_2021",
                    "wave": "2021",
                    "item": outcome,
                    "model": name,
                    **prediction_metrics(y, probabilities, weights),
                }
            )
            output = test[
                [
                    "wave",
                    "row_id",
                    "profile_id",
                    "is_llm_profile",
                    "analysis_weight",
                    outcome,
                ]
            ].copy()
            output = output.rename(columns={outcome: "observed"})
            output.insert(0, "split", "train_2012_2018_test_2021")
            output.insert(3, "item", outcome)
            output.insert(4, "model", name)
            for index, value in enumerate(CLASSES):
                output[f"p{value}"] = probabilities[:, index]
            prediction_rows.append(output)
        print(f"Temporal 2021 {outcome}: complete", flush=True)
    return pd.concat(prediction_rows, ignore_index=True), pd.DataFrame(metric_rows)


def matched_metrics(
    predictions: pd.DataFrame, llm: pd.DataFrame
) -> pd.DataFrame:
    rows: list[dict] = []
    matched_ml = predictions.loc[
        (predictions["split"] == "oof")
        & predictions["profile_id"].notna()
    ].copy()
    probability_columns = [f"p{value}" for value in CLASSES]
    for (wave, item, model), group in matched_ml.groupby(
        ["wave", "item", "model"]
    ):
        rows.append(
            {
                "source": "survey_trained_ml",
                "wave": str(wave),
                "item": item,
                "model": model,
                **prediction_metrics(
                    group["observed"].to_numpy(dtype=int),
                    group[probability_columns].to_numpy(dtype=float),
                    np.ones(len(group), dtype=float),
                ),
            }
        )

    llm = llm.loc[
        (llm["success"] == True)  # noqa: E712
        & (llm["condition"] == "neutral_verbal")
        & (llm["repeat"].between(1, 5))
    ].copy()
    for item_number in range(1, 6):
        item = f"eq_a42{item_number}"
        llm_column = f"a42{item_number}"
        if item_number <= 4:
            llm[f"pred_{item}"] = 6 - llm[llm_column]
        else:
            llm[f"pred_{item}"] = llm[llm_column]
    observed_lookup = (
        matched_ml[
            ["wave", "profile_id", "analysis_weight", "item", "observed"]
        ]
        .drop_duplicates()
    )
    llm_long = llm.melt(
        id_vars=["wave", "profile_id"],
        value_vars=[f"pred_{item}" for item in [
            "eq_a421", "eq_a422", "eq_a423", "eq_a424", "eq_a425"
        ]],
        var_name="item",
        value_name="prediction",
    )
    llm_long["item"] = llm_long["item"].str.replace("pred_", "", regex=False)
    llm_long["wave"] = llm_long["wave"].astype(str)
    llm_probabilities = (
        llm_long.assign(count=1)
        .pivot_table(
            index=["wave", "profile_id", "item"],
            columns="prediction",
            values="count",
            aggfunc="sum",
            fill_value=0,
        )
        .reindex(columns=CLASSES, fill_value=0)
        .div(5)
        .reset_index()
        .rename(columns={value: f"p{value}" for value in CLASSES})
    )
    observed_lookup["wave"] = observed_lookup["wave"].astype(str)
    llm_long = llm_probabilities.merge(
        observed_lookup,
        on=["wave", "profile_id", "item"],
        how="inner",
        validate="one_to_one",
    )
    for (wave, item), group in llm_long.groupby(["wave", "item"]):
        rows.append(
            {
                "source": "llm",
                "wave": str(wave),
                "item": item,
                "model": "qwen3_8b_neutral",
                **prediction_metrics(
                    group["observed"].to_numpy(dtype=int),
                    group[probability_columns].to_numpy(dtype=float),
                    np.ones(len(group), dtype=float),
                ),
            }
        )
    return pd.DataFrame(rows)


def temporal_matched_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    matched = predictions.loc[
        (predictions["split"] == "train_2012_2018_test_2021")
        & predictions["profile_id"].notna()
    ].copy()
    probability_columns = [f"p{value}" for value in CLASSES]
    for (wave, item, model), group in matched.groupby(
        ["wave", "item", "model"]
    ):
        rows.append(
            {
                "split": "train_2012_2018_test_matched_2021",
                "wave": str(wave),
                "item": item,
                "model": model,
                **prediction_metrics(
                    group["observed"].to_numpy(dtype=int),
                    group[probability_columns].to_numpy(dtype=float),
                    np.ones(len(group), dtype=float),
                ),
            }
        )
    output = pd.DataFrame(rows)
    expected_rows = 5 * 3
    if len(output) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} matched temporal metric rows; "
            f"received {len(output)}."
        )
    if output["n"].min() < 95 or output["n"].max() > 100:
        raise ValueError(
            "Matched temporal cells must retain the 100-profile cohort, "
            "allowing only item-specific missing human outcomes."
        )
    return output


def paired_bootstrap(
    predictions: pd.DataFrame,
    llm: pd.DataFrame,
    seed: int,
    repetitions: int = 2000,
) -> pd.DataFrame:
    ml = predictions.loc[
        (predictions["split"] == "oof")
        & (predictions["model"] == "hist_gradient_boosting")
        & predictions["profile_id"].notna()
    ].copy()
    probability_columns = [f"p{value}" for value in CLASSES]
    ml["ml_prediction"] = np.sum(
        ml[probability_columns].to_numpy(dtype=float) * CLASSES[None, :],
        axis=1,
    )
    llm = llm.loc[
        (llm["success"] == True)  # noqa: E712
        & (llm["condition"] == "neutral_verbal")
        & (llm["repeat"].between(1, 5))
    ].copy()
    parts = []
    for item_number in range(1, 6):
        part = llm[
            ["wave", "profile_id", "repeat", f"a42{item_number}"]
        ].rename(
            columns={f"a42{item_number}": "llm_prediction"}
        )
        part["item"] = f"eq_a42{item_number}"
        if item_number <= 4:
            part["llm_prediction"] = 6 - part["llm_prediction"]
        parts.append(part)
    llm_long = (
        pd.concat(parts, ignore_index=True)
        .groupby(["wave", "profile_id", "item"], as_index=False)
        .agg(
            llm_prediction=("llm_prediction", "mean"),
            completed_repeats=("repeat", "nunique"),
        )
    )
    if not (llm_long["completed_repeats"] == 5).all():
        raise ValueError("Paired bootstrap requires five completed LLM repeats.")
    llm_long["wave"] = llm_long["wave"].astype(str)
    ml["wave"] = ml["wave"].astype(str)
    paired = ml.merge(
        llm_long,
        on=["wave", "profile_id", "item"],
        how="inner",
        validate="one_to_one",
    )
    paired["difference"] = (
        abs(paired["llm_prediction"] - paired["observed"])
        - abs(paired["ml_prediction"] - paired["observed"])
    )
    profile_differences = (
        paired.groupby(["wave", "profile_id"], as_index=False)["difference"]
        .mean()
    )
    rng = np.random.default_rng(seed)
    bootstrap_values = np.empty(repetitions, dtype=float)
    wave_profiles = [
        group["difference"].to_numpy(dtype=float)
        for _, group in profile_differences.groupby("wave")
    ]
    for index in range(repetitions):
        sampled = [
            rng.choice(values, size=len(values), replace=True)
            for values in wave_profiles
        ]
        bootstrap_values[index] = np.concatenate(sampled).mean()
    return pd.DataFrame(
        [
            {
                "contrast": (
                    "qwen3_8b_five_draw_mean_minus_"
                    "hist_gradient_boosting"
                ),
                "metric": "paired_ordinal_absolute_error",
                "estimate": profile_differences["difference"].mean(),
                "ci_low": np.quantile(bootstrap_values, 0.025),
                "ci_high": np.quantile(bootstrap_values, 0.975),
                "bootstrap_repetitions": repetitions,
                "profiles": profile_differences["profile_id"].nunique(),
                "respondent_item_rows": len(paired),
            }
        ]
    )


def write_report(
    oof_metrics: pd.DataFrame,
    temporal_metrics: pd.DataFrame,
    temporal_matched: pd.DataFrame,
    matched: pd.DataFrame,
    bootstrap: pd.DataFrame,
) -> None:
    def markdown_table(frame: pd.DataFrame) -> str:
        formatted = frame.copy()
        for column in formatted.select_dtypes(include=[np.number]).columns:
            formatted[column] = formatted[column].map(
                lambda value: "NA" if pd.isna(value) else f"{value:.4f}"
            )
        headers = [str(column) for column in formatted.columns]
        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for row in formatted.itertuples(index=False, name=None):
            lines.append("| " + " | ".join(map(str, row)) + " |")
        return "\n".join(lines)

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    def aggregate(frame: pd.DataFrame, groups: list[str]) -> pd.DataFrame:
        return (
            frame.assign(absolute_mean_error=frame["mean_error"].abs())
            .groupby(groups, as_index=False)
            .agg(
                ordinal_expected_mae=("ordinal_expected_mae", "mean"),
                ranked_probability_score=("ranked_probability_score", "mean"),
                absolute_mean_error=("absolute_mean_error", "mean"),
                total_variation=("total_variation", "mean"),
                variance_ratio=("variance_ratio", "mean"),
            )
            .sort_values("ordinal_expected_mae")
        )

    oof_summary = aggregate(oof_metrics, ["model"])
    temporal_summary = aggregate(temporal_metrics, ["model"])
    temporal_matched_summary = aggregate(temporal_matched, ["model"])
    matched_summary = aggregate(matched, ["source", "model"])
    text = f"""# ML benchmark for the CGSS LLM audit

All five outcomes are oriented so that higher values indicate more egalitarian
attitudes. Machine-learning models receive exactly the demographic information
shown to the LLM. Results are predictive diagnostics, not causal estimates.

## Five-fold out-of-fold results

{markdown_table(oof_summary)}

## Temporal generalization: train 2012 and 2018, test 2021

{markdown_table(temporal_summary)}

## Temporal generalization on the same 100 Qwen profiles from 2021

This is the directly comparable temporal benchmark used in the manuscript.
Every model is evaluated on the identical 100 respondent records.

{markdown_table(temporal_matched_summary)}

## Same 300 profiles used in the Qwen3-8B audit

{markdown_table(matched_summary)}

## Paired uncertainty on the same profiles

Positive values mean that Qwen3-8B has larger absolute ordinal error than
HistGradientBoosting. Profiles are resampled within wave, retaining the five
items as a cluster.

{markdown_table(bootstrap)}

## Interpretation limits

The human-trained models use survey outcomes during training, whereas Qwen3-8B
is zero-shot. Their comparison tests whether a task-specific learner can
recover empirical regularities from the same profile features; it is not a
comparison of equal training budgets. Individual responses remain noisy, so
distributional recovery and calibration are more informative than exact match.
"""
    (REPORT_DIR / "ml_benchmark.md").write_text(text, encoding="utf-8")


def plot_matched(matched: pd.DataFrame) -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    summary = (
        matched.groupby(["model"], as_index=False)
        .agg(
            ordinal_mae=("ordinal_expected_mae", "mean"),
            mean_absolute_mean_error=("mean_error", lambda x: np.mean(abs(x))),
            total_variation=("total_variation", "mean"),
        )
        .sort_values("ordinal_mae")
    )
    labels = {
        "weighted_prior": "Weighted prior",
        "multinomial_logit": "Multinomial logit",
        "hist_gradient_boosting": "HistGradientBoosting",
        "qwen3_8b_neutral": "Qwen3-8B",
    }
    summary["label"] = summary["model"].map(labels).fillna(summary["model"])
    metrics = [
        ("ordinal_mae", "Ordinal MAE"),
        ("mean_absolute_mean_error", "Absolute mean error"),
        ("total_variation", "Total variation"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.8))
    colors = [
        "#4477AA" if model != "qwen3_8b_neutral" else "#CC6677"
        for model in summary["model"]
    ]
    for axis, (column, title) in zip(axes, metrics):
        axis.barh(summary["label"], summary[column], color=colors)
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
        axis.set_axisbelow(True)
    fig.suptitle("Same-profile comparison: survey-trained ML vs Qwen3-8B")
    fig.tight_layout()
    fig.savefig(
        FIGURE_DIR / "model_comparison.png",
        dpi=220,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(
        DATA_PATH,
        dtype={"wave": str, "prov": str, "profile_id": str},
    )
    llm = pd.read_csv(LLM_PATH, dtype={"wave": str, "profile_id": str})
    print(
        data[
            [
                "wave",
                "row_id",
                "female",
                "age",
                "educ_years",
                "eq_a421",
                "eq_a422",
                "eq_a423",
                "eq_a424",
                "eq_a425",
            ]
        ].head(7).to_string(index=False)
    )
    oof_predictions, oof_metrics = run_oof(data, config)
    temporal_predictions, temporal_metrics = run_temporal(data, config)
    predictions = pd.concat(
        [oof_predictions, temporal_predictions], ignore_index=True
    )
    matched = matched_metrics(oof_predictions, llm)
    temporal_matched = temporal_matched_metrics(temporal_predictions)
    bootstrap = paired_bootstrap(
        oof_predictions, llm, seed=config["seed"], repetitions=2000
    )

    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    oof_metrics.to_csv(OUTPUT_DIR / "metrics_oof.csv", index=False)
    temporal_metrics.to_csv(
        OUTPUT_DIR / "metrics_temporal_2021.csv", index=False
    )
    temporal_matched.to_csv(
        OUTPUT_DIR / "metrics_temporal_matched_2021.csv", index=False
    )
    matched.to_csv(OUTPUT_DIR / "metrics_matched_300.csv", index=False)
    bootstrap.to_csv(
        OUTPUT_DIR / "metrics_paired_bootstrap.csv", index=False
    )
    plot_matched(matched)
    write_report(
        oof_metrics, temporal_metrics, temporal_matched, matched, bootstrap
    )
    print("\nOOF mean metrics:")
    print(
        oof_metrics.groupby("model")[
            [
                "ordinal_expected_mae",
                "ranked_probability_score",
                "total_variation",
                "variance_ratio",
            ]
        ]
        .mean()
        .round(4)
    )
    print("\nMatched 300 mean metrics:")
    print(
        matched.groupby("model")[
            [
                "ordinal_expected_mae",
                "total_variation",
                "variance_ratio",
            ]
        ]
        .mean()
        .round(4)
    )


if __name__ == "__main__":
    main()
