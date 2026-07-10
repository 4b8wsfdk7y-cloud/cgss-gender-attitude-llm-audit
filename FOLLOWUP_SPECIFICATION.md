# Follow-up specification

Specification fixed before completion of the full follow-up model calls on
2026-07-02 (Asia/Shanghai).

## Repeated-draw estimand

- Primary LLM distribution: five independently seeded joint-prompt draws for
  each of 300 profiles.
- Each profile's five draws define an empirical predictive category
  distribution.
- Repeat 1 remains a sensitivity analysis for continuity with the initial
  pilot.
- Total predictive variance is decomposed into between-profile variance in
  profile-specific means and average within-profile stochastic variance.

## Independent-item manipulation

- Each item is presented in a fresh API call.
- The complete persona is repeated in every call.
- No conversation history, other item, or prior answer is available.
- Primary contrast:

  `delta_r = mean_abs_correlation_joint - mean_abs_correlation_independent`

- A profile-bootstrap 95% interval for `delta_r` entirely above zero supports
  context-induced coherence.
- If independent-item correlations remain above the CGSS benchmark, the
  evidence supports a mixture of contextual and model-level coherence.
- If an item is constant or nearly constant, its correlation is reported as
  undefined; lower correlation is not treated as structural recovery without
  heterogeneity recovery.

## Uncertainty

- Bootstrap unit: profile, stratified within survey wave.
- All five items and all five repeats for a sampled profile remain together.
- Repeats and item rows are never resampled independently.
- The full-wave CGSS benchmark is treated as fixed, so intervals describe
  uncertainty in the matched profile design and model generation rather than
  every stage of the original multistage survey.

## Temporal benchmark

- Survey-trained temporal models are trained on 2012 and 2018.
- The manuscript comparison uses predictions for the exact 100 matched 2021
  profiles used by Qwen3-8B.
- Full-2021 temporal metrics are retained only as an external-generalization
  supplement and are not directly compared with the matched Qwen sample.
