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

**2026-10-14, output-projection precision (before any Part 1–2 outcome was
computed).** The greedy run first scored 1,072 prompts with the model's default
output projection, which returns bfloat16 logits for every position. At logit
magnitudes between 16 and 32, bfloat16 resolves steps of 0.125, so the option
probabilities depended on which matmul kernel ran: on 40 of these prompts,
applying the same projection to the final position alone changed probabilities
by up to 0.031. The projection is now applied to the final position only, in
float32; final-position and all-position float32 projections then agree to
within 1.0e-5. On the same 40 prompts, float32 and bfloat16 probabilities
differed by 0.0086 on average (maximum 0.019), and the modal option changed for
one prompt. The run was restarted from the first profile. The 1,072 bfloat16
records are kept outside the repository for this comparison and are not used in
reported analyses. Two changes that leave the arithmetic unchanged were made at
the same time: a cap on MLX's buffer cache, and scoring each distinct prompt once
per profile (the first A0-B1 prompt is identical to the A0-B0 prompt for A421).

**2026-10-14, after the Part 1–2 results were computed: one correction and
three additions.**

1. *Correction.* Part 1 described cell A1-B1-C1 as corresponding to the v1
   joint condition. It shares that condition's battery visibility, answer
   order, and JSON output, but v1 also required a verbal label before each
   score (`{"A421":{"label":"完全不同意","score":1}}`). The two runs also differ
   in decoding (v1 sampled at temperature 0.7 with top-p 0.9; v2 reads the
   untempered distribution), in answer history (v1 histories were the model's
   own sampled answers; the primary v2 run uses modal answers), and possibly in
   weights (LM Studio `qwen/qwen3.5-9b` versus the MLX 4-bit conversion). The
   comparison with the v1 values of 68.2% and 0.265 is therefore reported as
   descriptive, not as a validation check.
2. *Exploratory cell A1-B1-C2,* specified here before it was run. It is the
   A1-B1 prompt with the v1 answer format: a verbal label, then the score. The
   option distribution is the distribution over the five labels, each scored
   as the sum of its token log-probabilities in one full forward pass. The
   closing quote is excluded because the tokenizer merges it with the
   following comma; the five label token sequences are checked to be
   prefix-free. Modal labels form the answer history. The cell is analysed
   with the Part 1 metrics and the Part 2 primary outcome, and compared with
   A1-B1-C1 using the same profile-cluster bootstrap. It does not enter the
   factorial effects.
3. *Exploratory comparator,* added after the Part 2 results: the primary
   outcome computed from the out-of-fold expected scores of the v1 histogram
   gradient boosting models for the same matched profiles. The weighted CGSS
   correlation is a response-level quantity and is not on the same scale as a
   correlation between profile-level expected scores.
4. *Definition used for interactions,* fixed in the analysis code before it
   was first run: a two-way interaction is a difference of simple effects
   averaged over the third factor (for example, the effect of A at B = 1 minus
   the effect of A at B = 0), and the three-way interaction is the difference
   between the A × B contrasts at C = 1 and at C = 0. The contrasts are not
   halved.
