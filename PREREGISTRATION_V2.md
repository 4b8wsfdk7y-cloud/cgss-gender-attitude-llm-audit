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

**2026-10-29, three exploratory analyses added after the preregistered ones.**
Fixed before any of them was run; none changes a preregistered estimand. The
first audit's subgroup gradients were already known (`output/metrics_subgroup_gradients.csv`).
All three use the bootstrap of Part 2 (profiles resampled within wave, 2,000
replicates, seed 20261013) unless stated otherwise.

1. *Human-history cells H-A1-B1-C1 and H-A0-B1-C0*
   (`scripts/15_run_human_history.py`). The prompts of A1-B1-C1 and
   A0-B1-C0, except that the history for each item holds the matched CGSS
   respondent's own answers to the earlier items instead of the model's.
   Profiles whose respondent lacks a valid answer on any of A421–A425 are
   excluded (4 of 300). A421 has no history and is taken from the primary
   run. Thinking disabled, one full forward pass per prompt. For the 296
   profiles:
   - (a) *Matched calibration* on A422–A425: mean log score and ranked
     probability score of the respondent's own answer, against the same
     cell with modal history and the matching cell without history
     (A1-B0-C1, A0-B0-C0), with bootstrap intervals for the differences.
   - (b) *History dependence of A425*: within wave, the Pearson correlation
     across profiles between the expected A425 score and the mean of the
     respondent's A421–A424 answers in egalitarian coding (6 − x), averaged
     over waves; compared with the same quantity for the survey-trained
     model with history (3b), and descriptively with the response-level
     correlation among the same respondents and in weighted CGSS.
2. *Subgroup gradients* (`scripts/17_subgroup_gradients.py`). For each of the
   eight factorial cells and V1-T1 and V1-T0, within wave: least squares
   across the 100 profiles of the egalitarian-coded expected score on female,
   years of education, age in decades (centred within wave), and urban
   residence, the first audit's specification. Outcomes: the public-role
   index (mean of A421–A424), household (A425), and each item. Benchmark:
   the first audit's weighted full-CGSS estimates. Reported per cell and
   averaged over waves: each coefficient, its difference from CGSS with a 95%
   interval (bootstrap for the model, a normal draw from the CGSS standard
   error), and the public-minus-household difference in the female and
   education coefficients.
3. *Survey-trained comparators.*
   - (a) *Response-level coherence of HGB*: five draws per profile from the
     first audit's out-of-fold HGB distributions, seeded as in the 2026-10-28
     entry, scored with its outcome R1.
   - (b) *HGB with history* (`scripts/16_fit_hgb_with_history.py`): for
     A422–A425, weighted HGB with the first audit's features and
     hyperparameters plus the respondent's answers to the earlier items,
     five-fold out of fold on all CGSS respondents with valid answers on
     those items (the first audit's splitter and seeds), giving predictions
     for the 296 matched profiles. Compared with HGB without history on the
     outcomes of 1(a) and 1(b).

The analyses of 1 and 3 are in `scripts/18_analyze_history_comparisons.py`.
Outputs are aggregate tables in `output/v2/exploratory_*.csv`.

**2026-10-29, human-history cells: run order, before any human-history
probabilities were analysed.** The first launch scored the two cells profile
by profile, at about a third of the speed of the earlier runs. To have one
cell complete for all 296 profiles, the run was restarted with
`--cells H-A1-B1-C1` (and a smaller MLX buffer-cache limit, which does not
enter the computation); the ledger is append-only and records already written
were kept. H-A0-B1-C0 is completed afterwards with the same script. Item 1 is
reported for H-A1-B1-C1 first (`scripts/18_analyze_history_comparisons.py
--human-cells H-A1-B1-C1`); H-A0-B1-C0 is analysed only once all its 296
profiles are scored, and no partial H-A0-B1-C0 result is reported.

*Addendum to item 1(b), written before the human-history cells were run.*
Also reported: the least-squares slope of A425 on the mean of the
respondent's A421–A424 answers, within wave and averaged over waves. Unlike a
correlation, the slope of an expected score and the slope of observed answers
estimate the same conditional mean, so the model, HGB, the matched
respondents, and weighted CGSS can be compared on one scale.

**2026-10-29 (evening), two further exploratory analyses, fixed before either
was run.** Neither changes a preregistered estimand.

1. *Recalibration with a small human sample*
   (`scripts/22_calibrate_with_human_sample.py`). The question is whether a
   researcher who holds the real answers of a few respondents with the same
   profiles could correct the errors reported above by recalibrating the
   model's option distributions.
   - *Profiles and cells.* The 296 profiles whose matched respondent answered
     all five items; uncalibrated values are recomputed on the same 296. The
     eight factorial cells, and A425 in the two human-history cells.
   - *Calibration sample.* n = 50, 100, or 200 matched respondents, drawn by
     simple random sampling without weights (profiles were drawn with
     probability proportional to weight within strata, so the matched
     respondents are close to self-weighting), and all 296 as the most
     favourable case.
   - *Methods,* fitted separately for each cell and item, pooled over waves,
     by penalised maximum likelihood on the sample's answers:
     (M1) category offsets, q ∝ p·exp(b_k) with b_1 = 0;
     (M2) offsets and an inverse temperature s, q ∝ p^s·exp(b_k).
     Penalty Σ b_k² / (2·5²) + (s − 1)² / 2, so that a category absent from a
     small sample does not send an offset to minus infinity.
   - *Uncertainty.* 1,000 replicates, seed 20261029. In each, profiles are
     resampled within wave as in Part 2, a calibration sample of size n is
     drawn from the resampled profiles, and the outcomes are computed on all
     resampled profiles. Reported: the median and the 2.5th and 97.5th
     percentiles over replicates.
   - *Outcomes,* with the definitions used above, averaged over waves:
     (a) mean absolute error of item means and mean total variation against
     weighted CGSS, averaged over items (Part 1);
     (b) between-profile variance ratio, averaged over items (Part 1);
     (c) subgroup gradients (item 2 of the morning entry): all four
     coefficients for the public-role index and A425, and the
     public-minus-household difference in the education coefficient,
     against the first audit's CGSS estimates (A425 on urban residence:
     0.01; education difference: +0.036);
     (d) in the human-history cells, the slope of A425 on the respondent's
     A421–A424 mean (addendum to item 1(b)), against weighted CGSS (0.106)
     and HGB with history (0.087).
   - *Expectations, stated in advance.* (a) falls as n grows under both
     methods. Under M1, (c) and (d) change little, because one offset per
     item moves every profile in the same direction. M2 can shrink all
     between-profile differences, so it may move (c) and (d) toward CGSS at
     the cost of (b) and of the public-role gradients. No direction is
     predicted for M2; what happens to the real public-role gradients is
     reported alongside.
   - *Output.* Aggregate tables in `output/v2/exploratory_calibration_*.csv`.

2. *Questionnaire-exposure probe* (`scripts/23_probe_questionnaire_exposure.py`).
   Whether the model has seen the A42 battery is untested. This probe tests
   only exposure to the questionnaire text (item wording and order), not to
   published response distributions, which is the contamination that would
   matter most for the audit; a null result does not rule out exposure.
   - *Texts.* A header, then the five items numbered 1–5, one per line,
     without final punctuation (wording as in `src/survey_llm_eval/factorial.py`).
     CGSS header: `中国综合社会调查（CGSS）问卷` / `A42. 您是否同意下列说法？`.
     Neutral header: `某市居民生活状况调查问卷` / `Q12. 您是否同意下列说法？`.
     Raw text, no chat template, one full forward pass per text, float32
     output projection.
   - *(a) Order test, primary.* For each header, the summed log-probability
     of the battery lines for all 120 orders of the five items. Reported: the
     rank of the CGSS order (1 = most probable), the one-sided permutation
     p-value rank/120, and the margin of the CGSS order over the median
     order. Reading fixed in advance: evidence of exposure to texts listing
     the battery in CGSS order requires p ≤ 0.05 under the CGSS header and a
     larger margin under the CGSS header than under the neutral header. A
     high rank under both headers reflects how plausible the order is, not
     exposure.
   - *(b) Free recall, secondary.* For each header and k = 1–4: the first k
     items in CGSS order, then `k+1. `, greedy decoding of up to 32 tokens,
     stopping at a newline. Reported: the generated line, whether it equals
     item k+1, and whether it equals any of the five items. A421 and A423
     are common sayings, so reproducing them is uninformative; exact
     reproduction of A422, A424, or A425 counts as evidence of exposure to
     texts quoting the battery.
   - *Positive control.* The same order test with the header
     `社会主义核心价值观：` and the first five values (富强、民主、文明、和谐、自由)
     separated by `、`, whose order the model should know. If the canonical
     order does not rank first, the probe is not sensitive enough to read.
   - *Output.* `output/v2/exploratory_exposure_*.csv`; contains no CGSS
     data.

**2026-10-31, a second model, fixed before any of its probabilities were
computed for the study profiles.** Exploratory; it changes no preregistered
estimand and adds no hypothesis.

1. *Model.* Gemma 4 12B, instruction-tuned (Google, Apache 2.0), in the MLX
   4-bit conversion `mlx-community/gemma-4-12B-it-4bit` (Hugging Face revision
   `73bcf09092aa277861d5a191b989b666f7f32e8f`). Its developer and training data
   differ from Qwen's, its size is close to Qwen3.5-9B, its chat template has a
   thinking switch, and the options 1–5 are single tokens. Loading it requires
   mlx-lm 0.32 or later, which requires Python 3.11, so it runs in a separate
   environment (Python 3.11.15, mlx-lm 0.32.2, mlx 0.32.4); the Qwen
   environment is unchanged.
2. *Output projection.* Gemma 4 applies a final-logit soft cap, c·tanh(x/c)
   with c = 30, which the float32 final-position projection of the 2026-10-14
   entry left out. The projection in `scripts/07_run_logprob_audit.py` and
   `scripts/23_probe_questionnaire_exposure.py` now applies the cap when the
   model configuration defines one. Qwen3.5 defines none; its projection is the
   same function as before and gave identical probabilities on test prompts.
   Checks on invented personas, not CGSS profiles: on 30 prompts, probabilities
   without the cap differed from the model's own forward pass by up to 0.296,
   and with the cap by up to 0.046, the bfloat16 precision issue of the
   2026-10-14 entry. The smallest probability mass on the five options was
   0.9996. With thinking disabled, the chat template closes an empty thought
   channel before the answer.
3. *Run.* The eight factorial cells with modal answer histories, all 300
   profiles, thinking disabled: `scripts/07_run_logprob_audit.py --history
   greedy`, seed 20261013, one full forward pass per prompt, as in the primary
   Qwen run. Outputs go to a separate directory outside the repository.
4. *Analysis.* The Part 1 metrics, and the Part 2 primary outcome and
   factorial effects with the Part 2 bootstrap, computed for Gemma with the
   definitions used for Qwen and reported beside the Qwen values. No direction
   is predicted. With one model per developer, differences between the two
   models are described and not attributed to the developer or to training
   data. Further analyses for Gemma (Part 3, the robustness check, the
   exploratory analyses) are specified in a later entry before they are run.
5. *Thinking-context cells.* V1-T1 is not yet defined for Gemma. Qwen3.5's
   template with thinking enabled ends in an open `<think>` block, so V1-T1
   scores the answer inside the thinking segment, as in v1. Gemma 4's template
   with thinking enabled adds `<|think|>` to the system turn and ends at the
   start of the model turn without opening a thought channel. The Gemma
   versions of V1-T1 and V1-T0 will be fixed in a later entry before those
   cells are run.

**2026-10-31, thinking-context cells for Gemma, fixed before they were run**
(item 5 of the previous entry).

1. *Cells.* V1-T1 and V1-T0 as specified in item 3 of the 2026-10-27 entry,
   with Gemma 4 12B. V1-T0 renders the chat template with thinking disabled,
   which closes an empty thought channel before the answer. V1-T1 renders it
   with thinking enabled and then appends `<|channel>thought\n`, the text with
   which the model opens its thought channel when thinking is enabled (its
   first three generated tokens in a check on an invented persona). The
   integer JSON is thus scored inside the thinking segment, as for Qwen3.5.
   Scoring the answer at the start of the model turn with thinking enabled but
   no thought channel open is not run.
2. *Implementation.* `scripts/11_run_v1_context_cells.py` appends the opening
   only when the thinking-enabled template ends at the start of a Gemma 4
   model turn. For Qwen3.5 the prompts are unchanged: the hashes of 40 Qwen
   V1-T1 and V1-T0 prompts equal those in the Qwen records. A smoke test on
   three profiles (one per wave) checked the code path and the v1 user-prompt
   hashes; its outputs are not analysed.
3. *Run and analysis.* All 300 profiles, after the factorial run has
   finished, with outputs in a separate directory outside the repository. The
   three contrasts of the 2026-10-27 entry, with the Part 1 metrics, the
   Part 2 primary outcome, and the profile-cluster bootstrap, where A1-B1-C1
   is Gemma's own cell. The comparison of V1-T1 item means with the v1
   sampled means is not made, because v1 ran only with Qwen. No direction is
   predicted.

**2026-10-31, human-history cells and subgroup gradients for Gemma, fixed
before either was run** (item 4 of the first 2026-10-31 entry). Exploratory;
no preregistered estimand changes. Before this entry, interim Part 1–2 results
for Gemma were inspected on the 276 profiles then complete; no human-history
or subgroup result had been computed for Gemma.

1. *Human-history cells.* H-A1-B1-C1 and H-A0-B1-C0 as in item 1 of the first
   2026-10-29 entry, with Gemma 4 12B, thinking disabled, one full forward
   pass per prompt (`scripts/15_run_human_history.py`, unchanged). The same
   296 profiles; A421 is taken from Gemma's primary run. The run starts after
   the thinking-context cells have finished and, following the Qwen run order
   (second 2026-10-29 entry), scores H-A1-B1-C1 for all 296 profiles before
   H-A0-B1-C0. Analysis: items 1(a) and 1(b), with the slope of the addendum,
   in `scripts/18_analyze_history_comparisons.py` with Gemma's ledgers. The
   survey-trained comparators of item 3 do not depend on the model and are
   reused unchanged. Neither cell is analysed before both are complete.
2. *Subgroup gradients.* Item 2 of the first 2026-10-29 entry for Gemma's eight
   factorial cells and its V1-T1 and V1-T0 cells, with the same benchmark
   (`scripts/17_subgroup_gradients.py`, unchanged).
3. *Reporting.* Aggregate tables for Gemma are written to a separate directory
   and do not replace the Qwen tables in `output/v2`. Values are reported
   beside Qwen's. No direction is predicted, and differences between the two
   models are described, not attributed to the developer or to training data.
4. *Not run for Gemma:* Part 3, the sampled-history robustness check of the
   2026-10-28 entry, and the two analyses of the 2026-10-29 (evening) entry.

**2026-11-15, downstream statistics, recorded after they were computed.**
Exploratory; no preregistered estimand changes. The plan below was written in
a separate note before any of these quantities was computed and is copied here
in substance; it was not committed before the run, so the results are
exploratory and are reported as such.

1. *Question.* If a researcher treated the synthetic respondents as survey data
   and ran the analyses common in public-management survey research, which
   conclusions would differ from those reached with CGSS?
2. *Populations.* H-full: CGSS 2012, 2018, 2021 with complete A421–A425, a
   positive weight, and the control variables, waves counting equally. H-300:
   1,000 samples of 100 respondents per wave drawn with probability
   proportional to the weight (seed 20261115), the sampling range at the
   synthetic sample size. H-matched: the 296 matched respondents with complete
   answers. S-drawn: Qwen3.5-9B, each factorial cell, the drawn answers of the
   2026-10-28 entry (five paths; history cells condition on earlier drawn
   answers), one dataset of 300 per path. S-modal: the most probable answer
   per item and cell. Higher values are more egalitarian on every item, as in
   Part 3.
3. *Quantities per dataset.* Cronbach's alpha for the five items and for
   A421–A424; the A425 corrected item–total correlation; a one-factor CFA (ML,
   items treated as continuous, survey weights for H-full) with CFI, RMSEA,
   SRMR, and the standardized A425 loading; OLS of A425 on the mean of
   A421–A424, alone and with female, years of education, age in decades, urban
   residence, and wave dummies.
4. *Decision rules reported.* Whether deleting A425 raises alpha; whether A425
   is kept by an item–total correlation of at least 0.30 or a loading of at
   least 0.40; whether the five-item alpha reaches 0.70.
5. *Comparison.* Each synthetic value against the 2.5–97.5 percentiles of
   H-300 and against H-full.
6. *Not done.* Gemma 4 12B (no sampled-history ledger, so its history cells
   have no drawn answers); an ordinal CFA.

Scripts: `scripts/25_export_drawn_answers.py` (profile-level output, written
outside the repository) and `scripts/26_downstream_statistics.R`. Aggregate
tables: `output/v2/exploratory_downstream_*.csv` and
`exploratory_downstream_metadata.json`.

**2026-11-17, a second item battery (trust replication), fixed before any of
its probabilities were computed for study profiles.** Exploratory replication;
it changes no preregistered estimand.

1. *Question.* Does the effect of answer history on cross-item coherence
   (Part 2) recur on a battery from another domain whose human structure is
   known?
2. *Battery.* Eight of the sixteen CGSS 2012 trust items (N16, "您对下面这些人的
   信任程度如何？"), in questionnaire order: N1601 亲戚, N1603 邻居, N1605 陌生人,
   N1606 医生, N1611 老师, N1612 地方政府官员, N1613 中央政府官员, N1614 警察.
   The selection was made on 2026-11-17 after computing weighted Pearson
   correlations of all sixteen items on the 5,139 respondents with complete
   answers; no model output for these items existed then. In those data the
   items form three blocks (interpersonal; occupations; public authorities),
   the stranger item correlates 0.04–0.23 with the other fifteen, and trust in
   central officials exceeds trust in local officials by 0.40 on the 1–4
   scale (correlation 0.59). The eight items keep two interpersonal items, the
   stranger item, two public-service occupations, and three public
   authorities.
3. *Prompt and scale.* System prompt, introduction, and persona construction
   as for the gender battery; stem "请评价你对以下这些人的信任程度："; scale
   "1=完全不信任；2=不太信任；3=比较信任；4=非常信任", so higher scores mean more
   trust. CGSS codes run the other way and are recoded as 5 − code. The
   options 1–4 are single tokens; the distribution is their probability
   renormalized over the four options, and the mass on them is recorded.
   Implementation: `TRUST` in `src/survey_llm_eval/factorial.py`; the gender
   prompts are unchanged (checked string by string for all cells).
4. *Profiles.* 300 CGSS 2012 respondents with complete demographics (coded as
   in `scripts/00_build_authorized_benchmark.R` for 2012), valid answers on the
   eight items, and a positive weight (5,595 eligible), sampled with the scheme
   of `scripts/01_prepare_profiles.R`: strata by sex × education group × urban
   residence, largest-remainder allocation proportional to the weight, and
   weighted sampling without replacement within strata, seed 20261117
   (`scripts/27_build_trust_profiles.R`). One wave only; outputs are written
   outside the repository.
5. *Run.* Qwen3.5-9B with the weights of the Scope section, thinking
   disabled, the eight factorial cells with modal answer histories, one full
   forward pass per prompt, seed 20261013: `scripts/07_run_logprob_audit.py
   --battery trust --history greedy`. A smoke test on three profiles checks
   the option mass and timing; its probabilities are not analysed.
6. *Primary outcome.* For each cell, the mean absolute Pearson correlation
   between the expected score on N1605 and the expected scores on the other
   seven items, across the 300 profiles. Main effects and interactions of A,
   B, and C as in Part 2, with a profile-cluster bootstrap of 2,000
   replicates (one wave, so no strata). The gender battery predicts a positive
   main effect of B; it is reported whatever its sign or interval.
7. *Secondary outcomes, by cell.* (a) The correlation across profiles between
   the expected scores on N1612 and N1613, and the population mean difference
   N1613 − N1612 compared with the weighted CGSS difference. (b) The mean
   correlation within the blocks {N1601, N1603}, {N1606, N1611}, and {N1612,
   N1613, N1614}, and the mean correlation between items of different blocks.
   (c) Item-level absolute mean error against the weighted CGSS 2012 means.
   (d) Mean valid-option mass; cells below 0.90 are flagged.
8. *References.* Weighted CGSS 2012 individual-level values on the eligible
   respondents; and a survey-trained HGB with the features and
   hyperparameters of `ml/config.json`, fitted on the eligible respondents,
   five-fold out of fold, whose expected scores for the 300 profiles give the
   information-matched value of each outcome. The HGB is fitted after the
   model run and before any comparison.
9. *Not part of this entry.* Human-history cells, sampled histories, a second
   model, and larger models; any of them would be fixed in a later entry
   before it is run.

**2026-11-18, trust replication: an interim look, and the run's outcome,
recorded after the analysis.** No analysis choice of the 2026-11-17 entry
changed.

1. *Interim look.* On 2026-11-17 at about 22:15 (UTC−6), with 70 of the 300
   profiles complete, the primary outcome was computed by cell from the
   partial ledger and compared with the weighted CGSS 2012 value. This came
   before the HGB was fitted and so departs from the order of point 8 of the
   2026-11-17 entry. The interim values (B main effect +0.18; 0.23–0.71
   without answer history and 0.61–0.81 with it; CGSS 0.09) were not used
   to change the battery, prompts, profiles, outcomes, or analysis code.
2. *Run.* 19,200 records, one per cell, profile, and item, with no duplicates;
   code commit 4fbd6a7; minimum valid-option mass 0.985, so no cell is
   flagged. The HGB was then fitted (`scripts/28_export_trust_ml_input.R`,
   `scripts/29_fit_trust_hgb.py`), and the analysis of points 6–8 was run
   (`scripts/30_analyze_trust_replication.py`). Aggregate tables:
   `output/v2/trust/`.
3. *Primary outcome.* The B main effect is +0.087 (95% interval 0.056 to
   0.118), positive as for the gender battery (+0.379) but smaller, and the
   interactions are large (ABC −0.573). The B simple effects are +0.38 at
   A1-C0 and between −0.08 and +0.08 in the other three combinations of A
   and C. Seven of the eight cells lie between 0.57 and 0.67, including three
   of the four cells without answer history; A1-B0-C0 is 0.19. The weighted
   CGSS value is 0.093 and the information-matched HGB value 0.139.
4. *Secondary outcomes.* The B main effect on the N1612–N1613 correlation is
   +0.001 (−0.032 to 0.033) and on the mean between-block correlation −0.198
   (−0.219 to −0.179). In every cell the central-minus-local mean difference
   (0.89–1.90) exceeds the weighted CGSS difference (0.43).
5. *Note.* numpy emits overflow and divide-by-zero warnings from the matrix
   product that forms the HGB expected scores; the probabilities are finite
   and the product equals an element-wise sum, so the warnings do not affect
   the values.

**2026-11-18, a second gender battery (ISSP 2012 module in CGSS 2012), fixed
before any of its probabilities were computed for study profiles.**
Exploratory replication; it changes no preregistered estimand. It was added
after the trust results above were known, for the reason given in point 1.

1. *Question.* On the trust battery the main effect of answer history was
   positive but small, and most of the excess coherence was present without
   history; on A421–A425 it came from history. This entry asks whether a
   second gender-attitude battery behaves like A421–A425, which would point
   to the attitude domain, or like the trust battery, which would point to
   properties of the particular battery.
2. *Battery.* The seven agree–disagree gender-role items of the ISSP 2012
   Family and Changing Gender Roles module as fielded in CGSS 2012 (section
   M, answered by 5,946 respondents), in questionnaire order: m101 有工作的妈妈
   也能和小孩关系亲密, m102 小孩上学前妈妈工作会影响小孩, m103 女的全职工作家庭
   生活会受影响, m104 女人真正想要的是家庭和孩子, m105 家庭妇女和工作一样有成就
   感, m201 夫妻双方都应该挣钱养家, m202 男人挣钱、女人顾家. Items m301 and m302
   use another response scale and are not included. On 2026-11-18, before any
   model output for these items existed, weighted Pearson correlations were
   computed on the 5,607 eligible respondents of point 4: m201 has a mean
   absolute correlation of 0.053 with the other six (the others 0.098–0.184),
   the largest absolute correlation is 0.45 (m102–m103), and the mean over
   the 21 pairs is 0.134. m201 is the target item because its value is the
   lowest, the role A425 has in the first battery. m202 is close in content
   to A421 (weighted absolute correlation 0.37 among the section M
   respondents with valid answers on both); m101 and m201
   are worded in the egalitarian direction, the others in the traditional
   one.
3. *Prompt and scale.* System prompt, introduction, persona construction, and
   stem as for A421–A425; scale "1=非常不同意；2=不同意；3=无所谓同意不同意；
   4=同意；5=非常同意", with the CGSS section M labels, so higher scores mean more
   agreement. CGSS codes run the other way and are recoded as 6 − code;
   "无法选择" (8) is missing. Implementation: `ISSP` in
   `src/survey_llm_eval/factorial.py`; the gender and trust prompts are
   unchanged.
4. *Profiles.* 300 CGSS 2012 respondents with complete demographics (coded as
   for the trust battery), valid answers 1–5 on all seven items, and a
   positive weight (5,607 eligible), sampled with the scheme of the trust
   entry, seed 20261118 (`scripts/31_build_issp_profiles.R`). Outputs are
   written outside the repository.
5. *Run.* Qwen3.5-9B as for the trust battery, thinking disabled, the eight
   cells with modal answer histories, one full forward pass per prompt, seed
   20261013: `scripts/07_run_logprob_audit.py --battery issp --history
   greedy`, 16,800 records. A smoke test on three profiles checks the option
   mass and timing; its probabilities are not analysed. Until the run has
   finished, only the record count and the log are read.
6. *Primary outcome.* For each cell, the mean absolute Pearson correlation
   between the expected score on m201 and the expected scores on the other
   six items, across the 300 profiles. Main effects and interactions of A, B,
   and C as in Part 2, with a profile-cluster bootstrap of 2,000 replicates.
   The first gender battery predicts a positive main effect of B; it is
   reported whatever its sign or interval.
7. *Source comparison.* E0 is the mean of the primary outcome over the four
   cells without history minus its HGB value (point 9), and D = B − E0, with
   the HGB value recomputed on every bootstrap draw. If the 95% interval of D
   lies above zero, answer history is reported as the larger source of the
   excess; if it lies below zero, the excess present without history is; if
   it includes zero, the comparison is undetermined. The rule was written
   after the results of both earlier batteries were known; their point
   values, computed for context only, are D = +0.36 for A421–A425 (B 0.379,
   E0 0.022) and D = −0.28 for the trust battery (B 0.087, E0 0.370). The
   first gender battery predicts D > 0.
8. *Secondary outcomes, by cell.* (a) The mean absolute correlation over all
   21 item pairs, and its B main effect. (b) The m101 outcome defined as in
   point 6, and its B main effect. (c) Item-level absolute mean error against
   the weighted CGSS 2012 means. (d) Mean valid-option mass; cells below 0.90
   are flagged.
9. *References.* Weighted CGSS 2012 values on the eligible respondents, and
   the survey-trained HGB of the trust entry (features and hyperparameters of
   `ml/config.json`, five-fold out of fold on the eligible respondents,
   `scripts/29_fit_trust_hgb.py --battery issp`), fitted after the run and
   before any comparison. The matched respondents' own answers are reported
   as a descriptive row. Analysis: `scripts/32_analyze_issp_replication.py`.
10. *Not part of this entry.* A questionnaire-exposure probe for these items,
    human-history cells, sampled histories, and other models.

**2026-11-22, item-pair structure without history, recorded after it was
computed (exploratory)**

After the ISSP run, a pair-level comparison was planned in a separate note
(its text and hash are stored in `output/v2/exploratory_pairs_metadata.json`)
and then computed with `scripts/33_exploratory_pair_structure.py`. For the 59
item pairs of the three batteries (Qwen3.5-9B), the mean absolute correlation
over the four cells without history was regressed on the HGB pair correlation:
slope 0.19 (0.14–0.25), R² 0.04, residual SD 0.23. By the note's rules this is
"scattered", so the reading that the model adds a fixed level of covariation
is not supported. Within A421–A425 the slope is 0.56 and all ten signs agree
with HGB; in the trust battery strangers are grouped with doctors, teachers,
officials, and police (0.55–0.70; HGB 0.05–0.17); M105 is placed on the
egalitarian side (negative with M102–M104 and M202), where weighted CGSS 2012
answers and HGB place it on the traditional side; M201 has the HGB direction
but 0.56–0.80 in magnitude. With history the slope is 0.03. These results
suggested the hypothesis tested in the next entry.

**2026-11-22, wording experiment, fixed before any reworded item was scored**

Known at the time: the results of the previous entry, and, from a dry run of
the analysis script on the original ledgers, the original-wording values of
the five indicators below in the two cells used here (M105 −0.708, M201 0.703,
A423 −0.218, A422 0.618, N1605 0.619).

1. *Hypothesis.* Without answer history, Qwen3.5-9B relates items through the
   surface cues of their wording rather than through the structure of
   respondents' answers. Changing a cue while keeping the meaning should move
   the reworded item's correlations with the other items; replacing words by
   near-synonyms without changing a cue should not.
2. *Items* (`WORDING_VARIANTS` in `src/survey_llm_eval/factorial.py`; item
   IDs, stem, scale, and persona unchanged). Cue items: M105 "对女人来说，当好
   家庭妇女也是一种成就。" (the equality word 一样 removed); M201 "一个家光靠一个
   人挣钱不够，夫妻都要工作。" (the normative 双方都应该 replaced by a practical
   reason); A423 "对女人来说，嫁个好丈夫比自己工作干得好更重要。" (the idiom
   written out). Controls: A422 "男人的能力生来就比女人强。"; N1605 "不认识的
   人". That the rewordings keep the meaning is the researcher's judgement;
   no respondent answered them.
3. *Run.* Qwen3.5-9B as in the earlier entries, cells A0-B0-C0 and A0-B0-C1
   only, where the prompt holds the target item alone, the same 300 profiles
   per battery, one full forward pass per prompt: `scripts/07_run_logprob_audit.py
   --battery BATTERY_wording --cells A0-B0-C0 A0-B0-C1 --items ...`, 3,000
   records. A smoke test on one profile per wave checks option mass and that
   prompts for unchanged items reproduce the hashes of the main runs; its
   probabilities are not analysed. Until the run has finished only the record
   count and the log are read.
4. *Indicators.* Each reworded item is correlated across profiles with the
   original scores of the other items from the main runs, per cell, then
   averaged over the two cells (A421–A425 within wave, averaged over waves).
   M105: mean signed correlation with M102, M103, M104, M202. M201: mean
   absolute correlation with the other six. A423: mean signed correlation with
   A421, A422, A424. A422: mean absolute correlation with the other four.
   N1605: mean absolute correlation with the other seven. Profile-cluster
   bootstrap, 2,000 replicates, seed 20261013; original and reworded scores
   move together.
5. *Predictions.* M105: reworded value above 0 (95% interval). M201: reworded
   minus original below 0. A423: reworded minus original above 0. A422 and
   N1605: the interval of reworded minus original lies within ±0.10. The
   hypothesis is reported as supported if all five hold, not supported if none
   holds, and partly supported otherwise, with each prediction reported.
6. *Secondary.* Valid-option mass per item and cell; values below 0.90 are
   flagged. Analysis: `scripts/34_analyze_wording_experiment.py`.
7. *Not part of this entry.* Cells with battery listing or history, other
   models, and human answers to the reworded items.
