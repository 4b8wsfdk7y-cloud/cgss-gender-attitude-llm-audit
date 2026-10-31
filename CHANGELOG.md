# Changelog

## Unreleased

### Added

- Results for the second human-history cell, H-A0-B1-C0, analysed once all
  296 profiles were scored. `output/v2/exploratory_human_history_*.csv` and
  `exploratory_history_dependence.csv` now cover both cells; the H-A1-B1-C1
  rows are unchanged. README, README.zh-CN, the paper, and the results page
  report both cells.
- `scripts/21_plot_exploratory.R`, which draws four README figures from the
  aggregate tables: measurement-model loadings, item-mean errors by
  configuration, subgroup gradients, and the human-history cells.

### Changed

- The v2 section of README.zh-CN now follows the English README section by
  section, with the same tables and figures.
- The results page shows the four new figures under the matching tables.

### Added (2026-10-29 evening, pre-specified in `PREREGISTRATION_V2.md` before running)

- `scripts/22_calibrate_with_human_sample.py`: recalibrates each cell's option
  distributions on the answers of 50, 100, 200, or all 296 matched
  respondents, by category offsets (M1) or offsets and an inverse temperature
  (M2), and recomputes marginal error, between-profile variance, subgroup
  gradients, and the A425 history slope (`output/v2/exploratory_calibration_*.csv`).
- `scripts/23_probe_questionnaire_exposure.py`: tests whether the model
  prefers the CGSS order of the A42 battery among all 120 orders, under a
  CGSS and a neutral header, with a positive control, and whether it recalls
  the items (`output/v2/exploratory_exposure_*.csv`). One generated line is
  offensive and is replaced in the public recall table by a placeholder with
  its SHA-256; a rerun reproduces the line locally.
- README and README.zh-CN report both analyses. The paper and the results
  page do not yet include them.

### Changed (2026-10-30)

- README and README.zh-CN are shorter: the v1 summary and results sections
  are merged, and the v2 section is condensed into main results, the thinking
  context, measurement and subgroup structure, and the follow-up analyses.
  Values are unchanged; the full tables remain in `output/v2/`.
- The README figures are four composites drawn by
  `scripts/24_plot_readme_figures.R` (renamed from `24_plot_calibration.R`)
  with one palette: `readme_v1_overview.png`, `readme_v2_factorial.png`,
  `readme_v2_structure.png`, and `readme_v2_followups.png`. They replace eleven
  figures in the README. `v2_recalibration.png` was removed; the figures from
  scripts 10, 20, and 21 remain for the results page.
- The v1 overview figure is drawn from `output/metrics_distribution.csv` and
  `output/metrics_qwen_sampling_envelope.csv`. The repository has no script
  for the earlier `marginal_fidelity.png` and `benchmark_comparison.png`; both
  files are kept, and the results page uses its own copy of the first.
- The quick start in README.zh-CN now uses the commands that CI runs; it had
  named a `dev` extra and `pytest`, neither of which the package provides.

### Fixed

- Rounding errors in reported values: the A425 log scores of H-A1-B1-C1 and
  A1-B1-C1 are 3.44 and 5.55 (were 3.45 and 5.56), the H-A1-B1-C1 minus
  no-history contrast on A425 is +1.76 (was +1.77), and the CGSS
  urban-residence coefficient for A425 is 0.01 (was 0.02), with 0.19 (was
  0.20) for the v1 public-role coefficient. The tables in `output/v2/` were
  correct.

## 2.0.0 (2026-10-29)

### Added

- Option-probability audit: one forward pass per profile, item, and condition
  returns the probabilities of options 1–5 (`scripts/07_run_logprob_audit.py`,
  `src/survey_llm_eval/probabilities.py`).
- Preregistered 2 × 2 × 2 presentation experiment: item block visibility,
  answer history, and output format (`src/survey_llm_eval/factorial.py`,
  `PREREGISTRATION_V2.md`).
- Weighted human benchmark export and the Part 1–2 analysis
  (`scripts/08_export_human_benchmark.R`, `scripts/09_analyze_logprob_audit.py`)
  with aggregate tables in `output/v2/`.
- Figures for the factorial results (`scripts/10_plot_v2.R`).
- Exploratory cells V1-T1 and V1-T0, which score the v1 joint prompt with the
  reasoning block open or closed (`scripts/11_run_v1_context_cells.py`,
  `scripts/13_analyze_v1_context.py`).
- Part 3 measurement models: ordinal CFA, invariance tests, and graded-response
  models for CGSS and three synthetic populations
  (`scripts/12_measurement_models.R`).
- Preregistered robustness check with sampled answer histories: five seeded
  paths per profile in the four history cells
  (`scripts/07_run_logprob_audit.py --history sampled`,
  `scripts/14_analyze_sampled_history.py`, figure from
  `scripts/20_plot_robustness.R`).
- Exploratory analyses specified on 2026-10-29: human-history cells that use
  the matched respondent's own earlier answers
  (`scripts/15_run_human_history.py`), gradient boosting with answer history
  (`scripts/16_fit_hgb_with_history.py`), subgroup gradients
  (`scripts/17_subgroup_gradients.py`), their comparisons
  (`scripts/18_analyze_history_comparisons.py`), and item means in egalitarian
  coding (`scripts/19_item_means_egalitarian.py`).
- `ml/config.json`, which `ml/scripts/02_train_evaluate.py` reads.
- A results page in `docs/index.html`.

### Changed

- `LICENSE` is the standard MIT text; non-code material is described in
  `NOTICE.md`.
- The SMR submission files were removed from the working tree.
- The paper (`paper/main_showcase.tex`) has a new section on the follow-up
  audit; the abstract, limitations, and conclusion were updated. README and
  README.zh-CN describe the v1 thinking context and the Part 3 results.

### Corrected

- The v1 call description. v1 used a system prompt, returned integer JSON only
  (an integer response schema overrode the prompt's label request), and ran
  with thinking enabled, so its answers were generated inside an open
  `<think>` block and read from `reasoning_content`. The v1 and v2 weights are
  the same. See the 2026-10-27 entry of `PREREGISTRATION_V2.md`.
- The paper's weighted CGSS correlation between A425 and the four-item mean:
  0.05, 0.10, and 0.11 by wave (the earlier values, 0.05 to 0.08, were close
  to the unweighted ones).
- The paper and README now qualify the direction of the item-mean errors,
  which changes with the configuration.
- `ml/README.md` named Qwen3-8B as the comparison model; it is Qwen3.5-9B.

## 0.1.0 (2026-08-02)

- Frozen v1 audit of Qwen3.5-9B against CGSS 2012, 2018, and 2021: 300
  profiles, five sampled joint calls per profile, human-sampling envelopes,
  survey-trained and joint-donor baselines, and the stateless single-item
  experiment.
