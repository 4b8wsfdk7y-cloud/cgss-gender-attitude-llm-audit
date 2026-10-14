# Pre-specification for SurveyLLM-Eval v2

Fixed on 2026-10-13 (America/Chicago), before any v2 option probabilities
were computed for CGSS-derived profiles. Changes after this date will be
listed at the end of this file with their reasons.

## Scope

- Model: Qwen3.5-9B, MLX 4-bit weights from `lmstudio-community/Qwen3.5-9B-MLX-4bit`,
  thinking disabled through the chat template.
- Profiles: the same 300 profiles used in v1 (100 per CGSS wave: 2012, 2018,
  2021), with the same model-facing persona text.
- Scoring: one full forward pass per prompt. The option distribution is the
  probability of the tokens `1`–`5`, renormalized over the scale. The
  probability mass on these five tokens is recorded for every prompt.
- Prefix caching is not used for reported results, because it changed the
  modal option in 2 of 80 smoke-test prompts.

## Part 1. Option-level fidelity

Reference cell: A1-B1-C1 (all items visible, answered in order, JSON output),
which corresponds to the v1 joint condition.

For each wave and item:

1. **Marginal error.** Population distribution = average of profile-level
   option distributions. Compare with the weighted CGSS distribution using
   absolute mean error and total variation distance.
2. **Between-profile variance ratio.** Variance of profile-level expected
   scores divided by the weighted human variance.
3. **Within-profile share.** Share of total predictive variance that lies
   within profiles (law of total variance).
4. **Calibration against matched respondents.** Mean ranked probability score
   and mean log score of each profile's distribution for that profile's
   observed answer, compared with the v1 survey-trained reference models.

The v1 sampling-based estimates (68.2% within-profile share; between-profile
variance ratio 0.265) are compared with items 2–3 as a validation check.

## Part 2. 2 × 2 × 2 presentation experiment

Factors: A battery visibility, B answer history, C output format (integer vs
JSON). Cells with history use the modal earlier answers as context (primary)
and five seeded draws from the earlier distributions (robustness).

**Primary outcome.** For each cell, the mean absolute Pearson correlation
between the expected score on A425 and the expected scores on A421–A424,
computed across profiles within wave and averaged over waves.

**Effects.** Main effects of A, B, and C are the difference between the mean
outcome at level 1 and at level 0, averaged over the other two factors.
Two-way and three-way interactions are the corresponding contrasts.

**Inference.** Profile-cluster bootstrap stratified by wave, 2,000 replicates,
resampling profiles with all their items and cells. An effect is described as
supported when the 95% percentile interval excludes zero. All effects are
reported regardless of sign or significance.

**Secondary outcomes,** by cell: item-level marginal mean error, the
between-profile variance ratio, and the mean valid-option mass. Cells with a
mean valid-option mass below 0.90 are flagged.

**Robustness.** Repeat the primary outcome using response-level correlations
from the sampled-history runs, and compare modal versus sampled histories.

## Part 3. Measurement models

Planned after Parts 1–2 and specified separately before estimation:
ordinal CFA (WLSMV) comparing a one-factor five-item model with a model
treating A425 separately, multi-group invariance between CGSS and the
synthetic population, and graded-response discrimination for A425.

## Changes

None yet.
