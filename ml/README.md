# Survey-trained ML benchmark

This module compares Qwen3-8B synthetic responses with supervised models
trained on CGSS human responses. All models receive the same demographic
features. All five outcomes are coded from 1 to 5 with higher values indicating
more egalitarian attitudes.

## Environment

```bash
python3 -m venv ml/.venv
ml/.venv/bin/python -m pip install -r ml/requirements.txt
```

## Run

Run from the project root:

```bash
Rscript ml/scripts/01_export_data.R
ml/.venv/bin/python ml/scripts/02_train_evaluate.py
```

The first command prints seven rows before writing the full ML input. The
second command fits:

- a survey-weighted marginal prior;
- weighted multinomial logistic regression;
- weighted histogram gradient boosting.

Evaluation includes five-fold out-of-fold predictions, temporal transfer from
2012 and 2018 to 2021, and a direct comparison on the 300 profiles used in the
Qwen3-8B audit. Qwen metrics use the empirical distribution from five repeated
draws per profile. The main temporal comparison is restricted to the exact 100
matched 2021 profiles; full-wave 2021 metrics remain a supplementary
generalization check.

## Interpretation

This is a predictive benchmark, not a causal model. A supervised model's
success shows that empirical regularities can be learned from the available
profile features; it does not imply that demographic attributes cause the
attitudes. The LLM is zero-shot while the ML models observe survey outcomes
during training, so the comparison concerns reconstruction of survey
structure, not equal training budgets.
