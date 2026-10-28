# Changelog

## 2.0.0 (unreleased)

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

## 0.1.0 (2026-08-02)

- Frozen v1 audit of Qwen3.5-9B against CGSS 2012, 2018, and 2021: 300
  profiles, five sampled joint calls per profile, human-sampling envelopes,
  survey-trained and joint-donor baselines, and the stateless single-item
  experiment.
