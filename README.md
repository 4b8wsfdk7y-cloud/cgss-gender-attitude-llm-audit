# CGSS Gender-Attitude LLM Audit

[![R](https://img.shields.io/badge/R-4.5-276DC3)](https://www.r-project.org/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)](https://www.python.org/)
[![Data](https://img.shields.io/badge/data-restricted-red)](#restricted-data-setup)

**Current primary configuration:** Qwen3.5-9B served locally through LM Studio, 300 demographic profiles, five repeated draws per profile, and matched human/ML reference benchmarks.

This directory contains an estimand-indexed audit of whether a local LLM can
reproduce the distribution and social structure of five gender-attitude items
in CGSS 2012, 2018, and 2021. Human resampling envelopes, survey-trained
predictors, and a joint-response donor benchmark distinguish ordinary sampling
variation from generator-specific reconstruction error.

The LLM is treated as an object of validation, not as a replacement for human
survey respondents. Raw CGSS data and personally identifying information are
never sent to an external service.

## Tested environment

- R 4.5.2; package versions are recorded in
  `environment/R-session-info.txt`
- Python 3.10 or later; the LLM runners use the standard library only
- Supervised ML dependencies are pinned in `ml/requirements.txt`
- LM Studio with an OpenAI-compatible local endpoint
- Primary local model: `qwen/qwen3.5-9b`
- An earlier Ollama/Qwen3-8B configuration is retained as `config_ollama_qwen3_8b.json`

## Restricted-data setup

CGSS microdata and respondent-level derived files are not distributable with
the public archive. With authorized `.dta` files, build the benchmark locally:

```bash
export CGSS_RAW_DIR="/absolute/path/to/authorized/cgss/files"
Rscript scripts/00_build_authorized_benchmark.R
```

The directory named by `CGSS_RAW_DIR` must contain `CGSS2012.dta`,
`CGSS2018.dta`, and `CGSS2021.dta`. The generated
`data/source/dimension_pilot_results.rds` remains restricted. See
`PUBLIC_RELEASE_MANIFEST.md` before sharing the project.

## Reproduce the smoke test

Run commands from the project root.

```bash
Rscript scripts/01_prepare_profiles.R
python3 scripts/02_run_local_llm.py --config config.json --limit 10 --repeats 1
Rscript scripts/03_evaluate_audit.R
```

The preparation script prints the first ten profiles before writing the full
pilot sample. Inspect these rows before starting model inference.

## Run the full pilot

The configured pilot contains 100 profiles per wave. Run conditions
explicitly; invoking the runner without filters would execute all four
conditions and all three configured repetitions.

```bash
python3 scripts/02_run_local_llm.py \
  --config config.json \
  --conditions neutral_verbal original \
  --repeats 1
Rscript scripts/03_evaluate_audit.R
Rscript scripts/05_extended_validation.R
```

The Python runner is resumable. It appends one JSON record per completed call
and skips existing `profile_id × condition × repeat` keys on restart.

The current output contains repeat 1 for all 300 profiles under `original` and
five independently seeded repeats for all 300 profiles under
`neutral_verbal`. The `original` condition contains one full repeat. The
`paraphrased` and `original_verbal` conditions contain ten-profile smoke tests
only.

## Follow-up experiments

The repeated-draw analysis is resumable and preserves every item response for
the same profile within a repeat:

```bash
python3 scripts/02_run_local_llm.py \
  --config config.json \
  --conditions neutral_verbal \
  --repeats 5
```

The independent-item experiment presents exactly one item per fresh API call.
Every call repeats the full persona, starts with no conversation history, and
never exposes another item or a previous answer:

```bash
python3 scripts/04_run_independent_items.py
```

The primary repeated-draw estimand averages the five empirical category
distributions for each profile. Repeat 1 is retained as a sensitivity analysis
for continuity with the original pilot. Bootstrap resampling occurs at the
profile level within wave; all five items and all five repeats belonging to a
sampled profile remain together.

The independent-item contrast was specified before the full follow-up run:

```text
delta_r = mean_abs_correlation_joint - mean_abs_correlation_independent
```

A profile-bootstrap 95% interval for `delta_r` entirely above zero would
support context-induced coherence. Because the independent condition contains
one response per profile-item while the joint condition contains five, the
realized contrast is exploratory: it does not isolate prompt context from
decoding variability or model representation.

## Validation ladder

`scripts/05_extended_validation.R` adds three checks:

- a 1,000-replication human sampling reference envelope under the exact
  100-profile stratified design;
- survey-weighted Pearson correlations, with unweighted Pearson and
  polychoric sensitivity estimates;
- a stratified joint-donor benchmark that samples complete human response
  vectors while matching wave, sex, education group, and urban residence.

The human envelope is a reference distribution, not a confidence interval.
The joint donor is deliberately simple and outcome-informed; it tests whether
preserving joint human response vectors can recover relational structure, not
whether it is an optimal prediction model.

## Render the report

```bash
Rscript -e "rmarkdown::render('report/llm_audit_pilot.Rmd')"
```

Rendering HTML requires `pandoc`. Without it, the knitted Markdown report can
still be regenerated with:

```bash
Rscript -e "setwd('report'); knitr::knit(
  'llm_audit_pilot.Rmd', output = 'llm_audit_pilot.md', quiet = TRUE
)"
```

## Main outputs

The following list describes the restricted local working copy. The public
archive excludes respondent-derived profiles and profile-level response logs;
see `PUBLIC_RELEASE_MANIFEST.md`.

- `data/profiles_pilot.csv`: sampled profiles with held-out human responses
- `data/profiles_llm_input.csv`: model-facing profiles without human responses
- `output/responses.jsonl`: immutable raw call log (restricted local copy)
- `output/responses.csv`: analysis-ready model responses
- `output/responses_independent.jsonl`: immutable single-item call log
  (restricted local copy)
- `output/responses_independent.csv`: deduplicated single-item responses
- `output/metrics_*.csv`: validation metrics
- `output/figures/`: comparison figures
- `report/llm_audit_pilot.md`: knitted pilot report
- `paper/`: manuscript, online supplement, bibliography, figures, and final PDFs

## Portfolio artifacts

- [`paper/chinese-audit-paper.pdf`](paper/chinese-audit-paper.pdf): Chinese research paper.
- [`paper/main_showcase.pdf`](paper/main_showcase.pdf): English showcase manuscript.
- [`presentation/ppe-forum-presentation.pdf`](presentation/ppe-forum-presentation.pdf): conference presentation combining the survey study and LLM audit.
- `output/metrics_*.csv`: aggregate audit outputs that do not contain respondent-level records.

## Reproducibility boundary

The public repository can reproduce model calls, evaluation logic, figures, and manuscripts after an authorized user rebuilds the restricted benchmark. It cannot ship a turnkey copy of CGSS microdata. This is deliberate compliance with the survey's data-use restrictions, not a missing-file error.

## Citation

When reusing the audit design or code, cite this repository and the accompanying paper. CGSS, model providers, and third-party packages should be cited separately under their own terms.

The `original_verbal` condition is a scale-comprehension diagnostic. It
requires both a Chinese response label and its numeric score, allowing the
pipeline to detect whether the model silently reverses the stated Likert
coding.

The `neutral_verbal` condition removes the potentially leading
anti-social-desirability wording from the initial prompt. It is the candidate
primary condition; the earlier `original` condition is retained as a prompt
ablation rather than treated as the baseline.

## Interpretation boundary

Agreement in means does not establish that the model reproduces a population.
The audit separately evaluates marginal distributions, response variance,
item correlations, subgroup gradients, prompt sensitivity, and repeat
stability. Profile-level prediction errors are descriptive and are not treated
as estimates of individual latent attitudes.

In the completed core run, the neutral prompt reduces average mean error
relative to the original prompt but does not recover the survey benchmark.
The model strongly overpredicts traditional responses on A423 and A424,
underpredicts support for equal housework on A425, compresses response
variance, and makes A425 much more correlated with the other items than it is
in CGSS. Five repeated joint draws per profile recover some stochastic
variation but leave the mean variance ratio at 0.466. Across all three waves,
every reported Qwen marginal, heterogeneity, and relational diagnostic falls
outside the corresponding 95% human sampling envelope. A joint donor
preserving complete human response vectors recovers variance and covariance
far better, showing that sparse profile information alone is not a sufficient
explanation for the failure. This comparison does not isolate architecture
from pretraining, prompting, or lack of task-specific calibration.

Independent presentation reduces A425 coherence by 0.096 on average, but its
profile-bootstrap interval includes zero and A422/A424 become constant or
nearly constant in several cells. These outputs support an audit of model
behavior, not the use of synthetic respondents as survey substitutes.
