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

**2026-10-27, after a check of the v1 call path: corrections to the
2026-10-14 entry, two exploratory cells, and the Part 3 specification.** This
entry was written before the two cells below were run and before any
measurement model was estimated. One diagnostic was run before it and is
reported here as such.

1. *Corrections.* The 2026-10-14 correction misdescribed v1 in three ways,
   established from the v1 runner (`02_run_local_llm.py`), the v1 response
   files, and the LM Studio server logs of the v1 run (2026-07-09, UTC).
   - v1 used a system prompt: "你正在参加一项社会调查模拟。严格遵守用户要求，只返回JSON，不解释推理。"
     The v2 system prompt drops "只返回JSON".
   - v1 did not produce verbal labels. Its prompt asked for a label and a
     score, but each request carried an integer-only JSON schema
     (`response_format`), and LM Studio's constrained decoding followed the
     schema. All 1,500 v1 joint responses are integer JSON of the form
     `{"A421": 4, "A422": 2, ...}` (55 characters, 46 completion tokens).
     Cell A1-B1-C2 therefore does not reproduce the v1 output format; it
     remains an exploratory cell about label formats.
   - v1 ran with thinking enabled. LM Studio's model definition for
     `qwen/qwen3.5-9b` sets the template variable `enable_thinking` to true by
     default, and Qwen3.5's chat template has no `/no_think` switch, so the
     generation prompt ended in an open `<think>` block. In every logged v1
     call, `content` is empty and the JSON is in `reasoning_content`
     (45 reasoning tokens); the v1 runner read `reasoning_content` when
     `content` was empty. v2 renders the template with thinking disabled,
     which closes an empty `<think></think>` block before the answer.
   - The weights are the same. LM Studio's download records contain only
     `lmstudio-community/Qwen3.5-9B-MLX-4bit`, the directory v2 loads.
   - The same model definition sets top-k 20 and a presence penalty of 1.5 as
     defaults. v1 requests set only temperature (0.7), top-p (0.9), and the
     token limit; the logs do not show whether the remaining defaults were
     applied. Top-k 20 cannot bind on a five-option scale.
2. *Diagnostic run before this entry.* For the 300 v1 joint prompts
   (user-prompt hashes checked against the v1 records), A421 was scored with
   thinking open and with thinking closed, otherwise identical. Its results
   motivated the cells below and are reported as a diagnostic.
3. *Exploratory cells V1-T1 and V1-T0* (`scripts/11_run_v1_context_cells.py`).
   Both use the v1 joint prompt exactly: the v1 system prompt, then a user
   message of `/no_think`, a newline, and `prompts/neutral_verbal_zh.txt` with
   the persona. V1-T1 renders the chat template with thinking enabled (the v1
   context); V1-T0 with thinking disabled. The assistant text is the integer
   JSON enforced by the v1 schema, with modal earlier answers as history, and
   each item is scored in one full forward pass. Both cells are analysed with
   the Part 1 metrics and the Part 2 primary outcome. Three contrasts are
   estimated with the profile-cluster bootstrap: V1-T1 minus V1-T0 (the
   thinking context, other v1 settings held fixed), V1-T0 minus A1-B1-C1 (the
   remaining prompt differences, thinking closed), and V1-T1 minus A1-B1-C1.
   Item means of V1-T1 are compared descriptively with the v1 sampled means.
   The cells do not enter the factorial effects.

### Part 3 specification (fixed 2026-10-27, before estimation)

*Populations.* Five items, recoded so that higher values are more egalitarian
on every item: A421–A424 are reversed (6 − x) and A425 is unchanged.

- Human: the CGSS 2012, 2018, and 2021 analysis samples, respondents with all
  five items in 1–5 and a positive weight. Weights are rescaled within wave so
  that each wave has the same total weight, matching the synthetic design of
  100 profiles per wave.
- S1 (primary synthetic): modal answers in the reference cell A1-B1-C1,
  one response per profile (n = 300).
- S2: modal answers in A0-B0-C0 (no battery, no history, integer), n = 300.
- S3: the v1 sampled joint responses, analysed as five replicate data sets of
  300 (repeat 1, …, repeat 5). For S3, the median and range over the five
  replicates are reported.
- Synthetic populations are unweighted. Waves are pooled.

*Sparse categories.* Within each analysis, the category with the smallest
count in any population of that analysis is merged with its neighbour toward
the scale midpoint (3) if that count is below 10; a sparse middle category is
merged with whichever neighbour has more observations across the populations.
The merge is repeated until every remaining category reaches 10 in every
population, and the same coding is applied to every population in the
analysis. An item left with a single category is dropped
from that analysis. All merges and drops are reported.

*Single-population models* (lavaan, ordered items, WLSMV, delta
parameterization, factor variance fixed to 1; human estimates use the
rescaled weights):

- M1: one factor, five items.
- M2: one factor, A421–A424 only.

Reported for each population: scaled CFI, TLI, and RMSEA, and SRMR for M1 and
M2; standardized loadings with standard errors for M1. *Primary estimand:* the
standardized A425 loading in M1. Synthetic-minus-human differences are
reported with 95% intervals from the two standard errors (independent
samples).

*Invariance* (human versus S1, and human versus each S3 replicate), M1,
two groups, theta parameterization, identification following Wu and
Estabrook (2016) via `semTools::measEq.syntax`: configural, then equal
thresholds, then equal thresholds and loadings (the order Wu and Estabrook
recommend for ordinal items). Each step is compared with the
previous one by the scaled chi-square difference test (`lavTestLRT`), and the
changes in scaled CFI and RMSEA are reported. A step is described as not
supported when the difference test gives p < .05 and CFI falls by more than
0.01. Because the human sample is about a hundred times larger than the
synthetic one, the tests are read together with the size of the loading
differences.

*Graded response model* (mirt, one dimension, human estimates with survey
weights): discrimination parameters for all five items in each population,
with standard errors. Reported for A425: its discrimination and its ratio to
the mean discrimination of A421–A424.

*Output.* Aggregate tables only, in `output/v2/part3_*.csv`.

**2026-10-27, after the Part 3 models were first estimated.** The first run
stopped because a model did not converge in a synthetic population whose modal
answers make several polychoric correlations nearly 1. Without changing any
specified model, the script now (1) records a model that fails or does not
converge as not estimable and continues; (2) flags improper solutions
(negative residual variances, or a standardized loading above 1 in absolute
value) in `part3_fit.csv`; (3) writes the polychoric correlations of every
population as a descriptive table, `part3_polychoric.csv`; and (4) reports fit
measures for every estimable invariance step, with difference tests only
between adjacent estimable steps. Graded-response models that do not converge
are recorded as missing; the very large discriminations estimated for
near-deterministic populations are reported but not interpreted. The human
population has 31,177 respondents with all five items, against 31,856 in the
per-item benchmark.

**2026-10-28, sampled-history robustness check: implementation and analysis,
fixed before the run.** The original specification names five seeded draws
and a response-level outcome. The details below were fixed before any
sampled-history probabilities were computed for analysis. A smoke test on
three profiles (one per wave, cell A1-B1-C1) checked the code path: first-item
probabilities equal the primary run's, and every history is reproduced by
replaying the seeded draws. Its outputs are not analysed.

1. *Runs.* The four cells with history (A0-B1-C0, A0-B1-C1, A1-B1-C0,
   A1-B1-C1), all 300 profiles, five paths per profile, thinking disabled,
   using `scripts/07_run_logprob_audit.py --history sampled --paths 5` with
   seed 20261013 (the mode has been in the script since the original
   specification). On each path, the answer to each item is drawn from that
   item's renormalized option distribution by a generator seeded with the
   seed, profile, cell, and path, and it enters the history of the next item;
   the A425 answer is drawn the same way. Prompts that recur within a profile
   are scored once. The first item of every path has no history and is the
   primary run's prompt.
2. *Cells without history.* Their prompts do not depend on history, so no new
   forward passes are made. Their response-level answers are drawn from the
   primary distributions with the same seeding scheme, one draw per item in
   order.
3. *R1, response-level outcome.* For each cell, path, and wave, the mean
   absolute Pearson correlation between the drawn A425 answers and the drawn
   A421–A424 answers across the 100 profiles, averaged over paths and then
   over waves. A correlation involving an item that is constant in a wave is
   undefined; it is left out of the average, and the number of such cases is
   reported. Factorial effects are computed from these values as in Part 2.
   Three response-level values are reported descriptively for comparison: the
   weighted CGSS value (0.069), the v1 human-sampling envelope for samples of
   100 per wave (95% upper bounds 0.198–0.234), and v1 (0.270; 0.232–0.335 by
   wave), which differs in its thinking context and prompt.
4. *R2, expected scores under sampled histories.* For each history cell and
   path, the primary outcome computed from the expected scores conditional on
   that path's history, averaged over the five paths. Reported: the value for
   each history cell, its difference from the modal-history value, and the
   factorial effects with the history cells replaced by their sampled-history
   values (cells without history unchanged).
5. *Inference.* The Part 2 profile-cluster bootstrap: 2,000 replicates within
   wave, seed 20261013; each resampled profile carries all its paths, items,
   and cells.
6. *Secondary, descriptive.* By cell, the mean absolute error of item means
   against CGSS and the between-profile variance ratio, computed from each
   profile's option distribution averaged over its five paths.
7. *Output.* Aggregate tables only, in `output/v2/robustness_*.csv`, written
   by `scripts/14_analyze_sampled_history.py`.
