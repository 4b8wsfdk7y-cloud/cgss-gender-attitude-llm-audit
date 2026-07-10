# requires: dplyr, readr, tidyr, ggplot2, patchwork, scales

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(patchwork)
  library(scales)
})

set.seed(20260702)

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1]]))
} else {
  normalizePath("reproduce.R")
}
paper_root <- dirname(script_path)
audit_root <- normalizePath(file.path(paper_root, ".."))
figure_dir <- file.path(paper_root, "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

paths <- list(
  profiles = file.path(audit_root, "data", "profiles_pilot.csv"),
  responses = file.path(audit_root, "output", "responses.csv"),
  distribution = file.path(audit_root, "output", "metrics_distribution.csv"),
  variance = file.path(audit_root, "output", "metrics_variance.csv"),
  correlations = file.path(audit_root, "output", "metrics_correlations.csv"),
  gradients = file.path(audit_root, "output", "metrics_subgroup_gradients.csv"),
  repeated = file.path(audit_root, "output", "metrics_repeated_predictive.csv"),
  variance_decomposition = file.path(
    audit_root, "output", "metrics_variance_decomposition.csv"
  ),
  coherence = file.path(
    audit_root, "output", "metrics_coherence_comparison.csv"
  ),
  coherence_bootstrap = file.path(
    audit_root, "output", "metrics_coherence_bootstrap_by_wave.csv"
  ),
  correlation_sensitivity = file.path(
    audit_root, "output", "metrics_correlation_sensitivity.csv"
  ),
  sampling_envelope = file.path(
    audit_root, "output", "metrics_human_sampling_envelope.csv"
  ),
  qwen_envelope = file.path(
    audit_root, "output", "metrics_qwen_sampling_envelope.csv"
  ),
  joint_donor = file.path(
    audit_root, "output", "metrics_joint_donor_baseline.csv"
  ),
  core_bootstrap = file.path(
    audit_root, "output", "metrics_core_bootstrap.csv"
  ),
  matched = file.path(audit_root, "ml", "output", "metrics_matched_300.csv"),
  temporal = file.path(
    audit_root, "ml", "output", "metrics_temporal_matched_2021.csv"
  ),
  benchmark = file.path(
    audit_root, "data", "source", "dimension_pilot_results.rds"
  )
)
stopifnot(all(file.exists(unlist(paths))))

theme_smr <- function(base_size = 9) {
  theme_minimal(base_size = base_size, base_family = "sans") +
    theme(
      plot.title = element_text(face = "bold", size = base_size + 1),
      plot.subtitle = element_text(size = base_size - 1, color = "grey30"),
      plot.caption = element_text(size = base_size - 2, color = "grey35"),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      axis.title = element_text(size = base_size),
      strip.text = element_text(face = "bold"),
      legend.position = "bottom",
      legend.title = element_blank(),
      plot.margin = margin(5, 7, 5, 5)
    )
}

palette <- c(
  "CGSS" = "#222222",
  "Qwen3-8B" = "#0072B2",
  "Original prompt" = "#D55E00",
  "Neutral prompt" = "#0072B2",
  "Weighted prior" = "#999999",
  "Multinomial logit" = "#009E73",
  "Histogram gradient boosting" = "#0072B2"
)
items <- paste0("a42", 1:5)
item_labels <- c(
  a421 = "A421",
  a422 = "A422",
  a423 = "A423",
  a424 = "A424",
  a425 = "A425"
)

# ---- Figure 1: diagnostic workflow ----
workflow <- tibble(
  x = 1:4,
  stage = c(
    "1  Marginal fidelity",
    "2  Heterogeneity fidelity",
    "3  Relational fidelity",
    "4  Predictive benchmark"
  ),
  question = c(
    "Are category shares\nand means calibrated?",
    "Is disagreement\npreserved?",
    "Are covariance and\nsubgroup gradients\npreserved?",
    "Does the generator beat\noutcome-informed\nreferences?"
  ),
  metric = c(
    "Mean error · Total variation",
    "Variance ratio",
    "Correlation RMSE\nGradient error",
    "Matched OMAE\nTemporal transfer"
  )
)

p_workflow <- ggplot(workflow, aes(x, 1)) +
  geom_segment(
    data = tibble(x = 1:3, xend = 2:4),
    aes(x = x + .33, xend = xend - .33, y = 1.11, yend = 1.11),
    linewidth = .8, arrow = arrow(length = unit(0.12, "inches")),
    color = "grey40", inherit.aes = FALSE
  ) +
  geom_label(
    aes(label = stage), nudge_y = .22, size = 3.2, fontface = "bold",
    linewidth = .35, label.padding = unit(.18, "lines"), fill = "white"
  ) +
  geom_text(aes(label = question), nudge_y = -.02, size = 2.55, lineheight = .92) +
  geom_text(
    aes(label = metric), nudge_y = -.27, size = 2.35,
    color = "#444444", fontface = "italic"
  ) +
  annotate(
    "text", x = 2.5, y = .49,
    label = "Deployment gate: stay within the human-sampling envelope at every required level",
    size = 3.2, fontface = "bold"
  ) +
  coord_cartesian(xlim = c(.55, 4.45), ylim = c(.42, 1.36), clip = "off") +
  labs(
    title = "Population-fidelity audit for synthetic survey respondents",
    subtitle = "Each stage diagnoses a distinct failure; success at an earlier stage does not imply success later."
  ) +
  theme_void(base_family = "sans", base_size = 9) +
  theme(
    plot.title = element_text(face = "bold", size = 11),
    plot.subtitle = element_text(size = 8.5, color = "grey30"),
    plot.margin = margin(8, 10, 8, 10)
  )

ggsave(
  file.path(figure_dir, "audit_design.pdf"), p_workflow,
  device = "pdf", width = 7.2, height = 3.0, units = "in"
)

# ---- Figure 2: marginal fidelity and prompt ablation ----
distribution <- read_csv(paths$distribution, show_col_types = FALSE) |>
  filter(condition %in% c("original", "neutral_verbal")) |>
  mutate(
    human_eq = if_else(item == "a425", human_mean, 6 - human_mean),
    llm_eq = if_else(item == "a425", llm_mean, 6 - llm_mean),
    eq_error = llm_eq - human_eq,
    condition = recode(
      condition,
      original = "Original prompt",
      neutral_verbal = "Neutral prompt"
    ),
    item_f = factor(item, levels = items, labels = item_labels[items]),
    wave = factor(wave, levels = c(2012, 2018, 2021))
  )

neutral_means <- distribution |>
  filter(condition == "Neutral prompt") |>
  select(wave, item_f, human_eq, llm_eq) |>
  pivot_longer(
    c(human_eq, llm_eq), names_to = "source", values_to = "mean"
  ) |>
  mutate(source = recode(source, human_eq = "CGSS", llm_eq = "Qwen3-8B"))

p_means <- ggplot(
  neutral_means, aes(item_f, mean, color = source, group = source)
) +
  geom_line(linewidth = .65) +
  geom_point(size = 2) +
  facet_wrap(~ wave, nrow = 1) +
  scale_color_manual(values = palette) +
  scale_y_continuous(limits = c(1, 5), breaks = 1:5) +
  labs(
    title = "A. Item means under the neutral prompt",
    x = NULL, y = "Mean (higher = more egalitarian)"
  ) +
  theme_smr() +
  theme(axis.text.x = element_text(size = 7))

p_errors <- ggplot(
  distribution,
  aes(item_f, abs(eq_error), color = condition, group = condition)
) +
  stat_summary(fun = mean, geom = "line", linewidth = .7) +
  stat_summary(fun = mean, geom = "point", size = 2.1) +
  scale_color_manual(values = palette) +
  scale_y_continuous(limits = c(0, 2.25), breaks = seq(0, 2, .5)) +
  labs(
    title = "B. Prompt ablation: error remains large",
    x = NULL, y = "Absolute mean error, averaged across waves"
  ) +
  theme_smr() +
  theme(axis.text.x = element_text(size = 7))

responses <- read_csv(paths$responses, show_col_types = FALSE) |>
  filter(
    success,
    condition %in% c("original", "neutral_verbal"),
    `repeat` == 1
  ) |>
  select(profile_id, wave, condition, all_of(items)) |>
  pivot_longer(all_of(items), names_to = "item", values_to = "raw_score") |>
  mutate(
    eq_score = if_else(item == "a425", raw_score, 6 - raw_score)
  ) |>
  select(profile_id, wave, item, condition, eq_score) |>
  pivot_wider(names_from = condition, values_from = eq_score) |>
  mutate(prompt_shift = abs(original - neutral_verbal))

prompt_shift <- responses |>
  group_by(wave, item) |>
  summarise(
    mean_abs_shift = mean(prompt_shift),
    agreement = mean(original == neutral_verbal),
    .groups = "drop"
  ) |>
  mutate(
    item_f = factor(item, levels = items, labels = item_labels[items]),
    wave = factor(wave, levels = c(2012, 2018, 2021))
  )

p_shift <- ggplot(
  prompt_shift, aes(item_f, mean_abs_shift, fill = wave)
) +
  geom_col(position = position_dodge(width = .75), width = .68, color = "white") +
  scale_fill_manual(values = c("2012" = "#B3CDE3", "2018" = "#6497B1", "2021" = "#005B96")) +
  scale_y_continuous(limits = c(0, 1.25), breaks = seq(0, 1.2, .3)) +
  labs(
    title = "C. Same-profile response shifts",
    x = NULL, y = "Mean absolute response shift"
  ) +
  theme_smr() +
  theme(axis.text.x = element_text(size = 7))

marginal_figure <- (p_means / (p_errors | p_shift)) +
  plot_annotation(
    title = "Marginal fidelity is item-specific and prompt-sensitive",
    caption = paste0(
      "Notes: A421 = family role; A422 = male ability; A423 = marriage; ",
      "A424 = dismissal; A425 = housework.\n",
      "The original prompt discouraged socially desirable answers; the neutral prompt removed\n",
      "that instruction and added verbal labels. All panels use the same 300 profiles."
    ),
    theme = theme(plot.caption = element_text(size = 6.5, hjust = 0))
  )

ggsave(
  file.path(figure_dir, "marginal_fidelity.pdf"), marginal_figure,
  device = "pdf", width = 7.2, height = 7.0, units = "in"
)

# ---- Monte Carlo: marginal matching does not identify joint structure ----
benchmark <- readRDS(paths$benchmark)
human_2021 <- benchmark$samples[["2021"]] |>
  select(all_of(items), weight) |>
  filter(if_all(all_of(items), ~ !is.na(.x)), !is.na(weight), weight > 0)

target_probs <- lapply(items, function(item) {
  mass <- tapply(human_2021$weight, human_2021[[item]], sum)
  out <- rep(0, 5)
  out[as.integer(names(mass))] <- mass
  out / sum(out)
})
names(target_probs) <- items
target_means <- vapply(
  target_probs, function(p) sum((1:5) * p), numeric(1)
)
target_vars <- vapply(
  target_probs,
  function(p) sum(((1:5) - sum((1:5) * p))^2 * p),
  numeric(1)
)
target_cor <- cov.wt(
  as.matrix(human_2021[items]),
  wt = human_2021$weight,
  cor = TRUE
)$cor

draw_categories <- function(u, probability) {
  findInterval(u, c(0, cumsum(probability)), rightmost.closed = TRUE)
}

simulate_once <- function(scenario, n = 300, coherence = .65) {
  if (scenario == "Row bootstrap") {
    index <- sample.int(
      nrow(human_2021), n, replace = TRUE, prob = human_2021$weight
    )
    generated <- as.matrix(human_2021[index, items])
  } else if (scenario == "Independent marginals") {
    generated <- sapply(items, function(item) {
      sample(1:5, n, replace = TRUE, prob = target_probs[[item]])
    })
  } else if (scenario == "Coherent factor") {
    common <- rnorm(n)
    latent <- sapply(seq_along(items), function(j) {
      sqrt(coherence) * common + sqrt(1 - coherence) * rnorm(n)
    })
    generated <- sapply(seq_along(items), function(j) {
      draw_categories(pnorm(latent[, j]), target_probs[[items[[j]]]])
    })
    colnames(generated) <- items
  } else {
    stop("Unknown scenario.")
  }

  generated_means <- colMeans(generated)
  generated_vars <- apply(generated, 2, var)
  generated_cor <- cor(generated)
  generated_probs <- sapply(seq_along(items), function(j) {
    tabulate(generated[, j], nbins = 5) / n
  })
  tv <- vapply(seq_along(items), function(j) {
    .5 * sum(abs(generated_probs[, j] - target_probs[[items[[j]]]]))
  }, numeric(1))
  tibble(
    scenario = scenario,
    abs_mean_error = mean(abs(generated_means - target_means)),
    total_variation = mean(tv),
    mean_variance_ratio = mean(generated_vars / target_vars),
    correlation_rmse = sqrt(mean((generated_cor - target_cor)^2)),
    a425_mean_abs_correlation = mean(abs(generated_cor[1:4, 5]))
  )
}

simulation <- bind_rows(lapply(
  c("Row bootstrap", "Independent marginals", "Coherent factor"),
  function(scenario) {
    bind_rows(replicate(
      1000, simulate_once(scenario), simplify = FALSE
    )) |>
      mutate(replication = row_number())
  }
))

simulation_summary <- simulation |>
  group_by(scenario) |>
  summarise(
    across(
      c(
        abs_mean_error, total_variation, mean_variance_ratio,
        correlation_rmse, a425_mean_abs_correlation
      ),
      list(mean = mean, mcse = ~ sd(.x) / sqrt(n())),
      .names = "{.col}_{.fn}"
    ),
    replications = n(),
    .groups = "drop"
  )

# ---- Figure 3: structural fidelity, gradients, and diagnostic simulation ----
variance <- read_csv(paths$variance_decomposition, show_col_types = FALSE) |>
  mutate(
    item_f = factor(item, levels = items, labels = item_labels[items]),
    wave = factor(wave, levels = c(2012, 2018, 2021)),
    within_variance_ratio = within_profile_variance / human_variance
  ) |>
  select(
    wave, item_f,
    `Between profiles` = between_variance_ratio,
    `Within profiles` = within_variance_ratio
  ) |>
  pivot_longer(
    c(`Between profiles`, `Within profiles`),
    names_to = "component", values_to = "variance_ratio"
  ) |>
  mutate(
    component = factor(
      component, levels = c("Within profiles", "Between profiles")
    )
  )

p_variance <- ggplot(
  variance, aes(item_f, variance_ratio, fill = component)
) +
  geom_hline(yintercept = 1, linetype = "dashed", color = "grey45") +
  geom_col(width = .7) +
  facet_wrap(~ wave, nrow = 1) +
  scale_fill_manual(values = c(
    "Between profiles" = "#0072B2",
    "Within profiles" = "#B3CDE3"
  )) +
  scale_y_continuous(
    limits = c(0, 1.35), breaks = seq(0, 1.25, .25),
    expand = expansion(mult = c(0, .02))
  ) +
  labs(
    title = "A. Repeated draws still compress variance",
    x = NULL, y = "Variance component / human variance"
  ) +
  theme_smr() +
  theme(
    axis.text.x = element_text(size = 7),
    legend.position = "bottom"
  )

coherence <- read_csv(paths$coherence, show_col_types = FALSE)
coherence_intervals <- read_csv(
  paths$coherence_bootstrap, show_col_types = FALSE
) |>
  filter(metric %in% c("joint_a425", "independent_a425")) |>
  mutate(source = recode(
    metric,
    joint_a425 = "Joint prompt",
    independent_a425 = "Independent items"
  )) |>
  select(wave, source, ci_low, ci_high)

weighted_human_correlations <- read_csv(
  paths$correlation_sensitivity, show_col_types = FALSE
) |>
  filter(method == "survey_weighted_pearson") |>
  transmute(
    wave,
    source = "CGSS",
    correlation = a425_mean_abs_correlation
  )

correlations <- coherence |>
  select(
    wave,
    `Joint prompt` = joint_a425,
    `Independent items` = independent_a425
  ) |>
  pivot_longer(
    c(`Joint prompt`, `Independent items`),
    names_to = "source", values_to = "correlation"
  ) |>
  bind_rows(weighted_human_correlations) |>
  left_join(coherence_intervals, by = c("wave", "source")) |>
  mutate(wave = factor(wave, levels = c(2012, 2018, 2021)))

p_cor <- ggplot(
  correlations, aes(wave, correlation, color = source, group = source)
) +
  geom_line(linewidth = .75) +
  geom_point(size = 2.2) +
  geom_errorbar(
    aes(ymin = ci_low, ymax = ci_high),
    width = .08, linewidth = .45, na.rm = TRUE
  ) +
  scale_color_manual(values = c(
    "CGSS" = "#222222",
    "Joint prompt" = "#D55E00",
    "Independent items" = "#0072B2"
  )) +
  scale_y_continuous(limits = c(0, .6), breaks = seq(0, .6, .1)) +
  labs(
    title = "B. Independent items weakly reduce coherence",
    x = "CGSS wave", y = "Mean |correlation|: A425 vs. A421-A424"
  ) +
  theme_smr()

gradients <- read_csv(paths$gradients, show_col_types = FALSE) |>
  filter(
    condition == "neutral_verbal",
    outcome %in% c("public", "household")
  ) |>
  mutate(
    outcome = recode(outcome, public = "Public-item index", household = "Equal housework"),
    predictor = recode(
      predictor,
      female = "Female",
      educ_years = "Education",
      age_decade = "Age (decade)",
      urban_residence = "Urban"
    )
  )
gradient_rmse <- sqrt(mean((gradients$llm_estimate - gradients$human_estimate)^2))

p_gradient <- ggplot(
  gradients,
  aes(human_estimate, llm_estimate, shape = outcome, color = predictor)
) +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "grey40") +
  geom_point(size = 2.15, alpha = .9) +
  facet_wrap(~ wave, nrow = 1) +
  scale_color_manual(values = c(
    "Female" = "#0072B2", "Education" = "#D55E00",
    "Age (decade)" = "#009E73", "Urban" = "#CC79A7"
  )) +
  guides(shape = "none") +
  coord_equal(xlim = c(-.15, .55), ylim = c(-.15, .55)) +
  labs(
    title = "C. Demographic gradients are only partially recovered",
    subtitle = sprintf("Coefficient RMSE = %.3f across 24 comparisons", gradient_rmse),
    x = "CGSS coefficient", y = "Qwen3-8B coefficient"
  ) +
  theme_smr(base_size = 8)

sim_plot <- simulation_summary |>
  mutate(
    scenario = factor(
      scenario,
      levels = c("Row bootstrap", "Independent marginals", "Coherent factor")
    )
  )

p_sim <- ggplot(
  sim_plot,
  aes(total_variation_mean, correlation_rmse_mean, label = scenario, color = scenario)
) +
  geom_point(size = 3) +
  geom_text(
    nudge_y = .035, size = 2.7, show.legend = FALSE,
    check_overlap = FALSE
  ) +
  scale_color_manual(values = c(
    "Row bootstrap" = "#009E73",
    "Independent marginals" = "#E69F00",
    "Coherent factor" = "#D55E00"
  )) +
  scale_x_continuous(limits = c(0, .09), breaks = seq(0, .08, .02)) +
  scale_y_continuous(limits = c(0, .48), breaks = seq(0, .4, .1)) +
  labs(
    title = "D. Monte Carlo: marginal fit can hide structural failure",
    subtitle = "1,000 replications; n = 300; all scenarios use CGSS marginals",
    x = "Mean total variation", y = "Correlation-matrix RMSE"
  ) +
  theme_smr(base_size = 8) +
  theme(legend.position = "none")

structural_figure <- (p_variance | p_cor) / (p_gradient | p_sim) +
  plot_annotation(
    title = "Synthetic responses are too concentrated and too coherent",
    caption = paste0(
      "Notes: Dashed lines mark exact recovery. Subgroup regressions adjust simultaneously ",
      "for sex, education, age, and urban residence.\n",
      "Circles denote equal housework and triangles the public-item index. ",
      "Monte Carlo standard errors are smaller than the plotted symbols."
    ),
    theme = theme(plot.caption = element_text(size = 6.3, hjust = 0))
  )

ggsave(
  file.path(figure_dir, "structural_fidelity.pdf"), structural_figure,
  device = "pdf", width = 7.2, height = 8.0, units = "in"
)

# ---- Figure 4: matched and temporal reference models ----
matched <- read_csv(paths$matched, show_col_types = FALSE)
repeated_metrics <- read_csv(paths$repeated, show_col_types = FALSE) |>
  transmute(
    source = "llm",
    wave,
    item,
    model = "qwen3_8b_neutral",
    n,
    ordinal_expected_mae,
    mean_error,
    total_variation,
    variance_ratio
  )
matched_primary <- bind_rows(
  matched |>
    filter(source == "survey_trained_ml"),
  repeated_metrics
)
matched_summary <- matched_primary |>
  group_by(source, model) |>
  summarise(
    ordinal_mae = weighted.mean(ordinal_expected_mae, n),
    absolute_mean_error = weighted.mean(abs(mean_error), n),
    total_variation = weighted.mean(total_variation, n),
    variance_ratio = weighted.mean(variance_ratio, n),
    .groups = "drop"
  ) |>
  mutate(model_label = recode(
    model,
    weighted_prior = "Weighted prior",
    multinomial_logit = "Multinomial logit",
    hist_gradient_boosting = "Histogram gradient boosting",
    qwen3_8b_neutral = "Qwen3-8B"
  ))

matched_long <- matched_summary |>
  select(model_label, ordinal_mae, absolute_mean_error, total_variation) |>
  pivot_longer(-model_label, names_to = "metric", values_to = "value") |>
  mutate(
    metric = recode(
      metric,
      ordinal_mae = "Ordinal MAE",
      absolute_mean_error = "Absolute mean error",
      total_variation = "Total variation"
    ),
    model_label = factor(
      model_label,
      levels = c(
        "Weighted prior", "Multinomial logit",
        "Histogram gradient boosting", "Qwen3-8B"
      )
    )
  )

p_matched <- ggplot(
  matched_long, aes(model_label, value, fill = model_label)
) +
  geom_col(width = .7) +
  facet_wrap(~ metric, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = c(
    "Weighted prior" = "#999999",
    "Multinomial logit" = "#009E73",
    "Histogram gradient boosting" = "#0072B2",
    "Qwen3-8B" = "#D55E00"
  )) +
  labs(
    title = "A. Same 300 profiles: outcome-informed references recover the population",
    x = NULL, y = NULL
  ) +
  theme_smr() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 28, hjust = 1, size = 7)
  )

temporal <- read_csv(paths$temporal, show_col_types = FALSE) |>
  group_by(model) |>
  summarise(
    ordinal_mae = weighted.mean(ordinal_expected_mae, n),
    total_variation = weighted.mean(total_variation, n),
    variance_ratio = weighted.mean(variance_ratio, n),
    .groups = "drop"
  ) |>
  mutate(model_label = recode(
    model,
    weighted_prior = "Weighted prior",
    multinomial_logit = "Multinomial logit",
    hist_gradient_boosting = "Histogram gradient boosting"
  ))

qwen_2021 <- repeated_metrics |>
  filter(wave == 2021) |>
  summarise(
    ordinal_mae = weighted.mean(ordinal_expected_mae, n),
    total_variation = weighted.mean(total_variation, n),
    variance_ratio = weighted.mean(variance_ratio, n)
  ) |>
  mutate(model_label = "Qwen3-8B")

temporal_long <- bind_rows(temporal, qwen_2021) |>
  pivot_longer(
    c(ordinal_mae, total_variation, variance_ratio),
    names_to = "metric", values_to = "value"
  ) |>
  mutate(
    metric = recode(
      metric,
      ordinal_mae = "Ordinal MAE",
      total_variation = "Total variation",
      variance_ratio = "Variance ratio"
    ),
    model_label = factor(
      model_label,
      levels = c(
        "Weighted prior", "Multinomial logit",
        "Histogram gradient boosting", "Qwen3-8B"
      )
    )
  )

p_temporal <- ggplot(
  temporal_long, aes(model_label, value, fill = model_label)
) +
  geom_col(width = .7) +
  facet_wrap(~ metric, scales = "free_y", nrow = 1) +
  scale_fill_manual(values = c(
    "Weighted prior" = "#999999",
    "Multinomial logit" = "#009E73",
    "Histogram gradient boosting" = "#0072B2",
    "Qwen3-8B" = "#D55E00"
  )) +
  labs(
    title = "B. Temporal transfer: train on 2012-2018, evaluate 2021",
    subtitle = "Qwen3-8B is zero-shot; reference models use historical outcomes only.",
    x = NULL, y = NULL
  ) +
  theme_smr() +
  theme(
    legend.position = "none",
    axis.text.x = element_text(angle = 28, hjust = 1, size = 7)
  )

benchmark_figure <- (p_matched / p_temporal) +
  plot_annotation(
    title = "Outcome-informed benchmarks separate sparse-profile limits from generator distortion",
    caption = paste0(
      "Notes: Lower is better for MAE, absolute mean error, and total variation; ",
      "a variance ratio of 1 indicates recovery.\n",
      "Reference models observe survey outcomes during training, whereas Qwen3-8B does not."
    ),
    theme = theme(plot.caption = element_text(size = 6.5, hjust = 0))
  )

ggsave(
  file.path(figure_dir, "benchmark_comparison.pdf"), benchmark_figure,
  device = "pdf", width = 7.2, height = 7.1, units = "in"
)

# ---- Console audit: values cited in the manuscript ----
cat("\nFive-row preview: neutral-prompt egalitarian means\n")
print(
  distribution |>
    filter(condition == "Neutral prompt") |>
    select(wave, item, human_eq, llm_eq, eq_error) |>
    slice_head(n = 5)
)

cat("\nPrompt-ablation summary (300 matched profiles)\n")
print(
  distribution |>
    group_by(condition) |>
    summarise(
      mean_absolute_item_error = mean(abs(eq_error)),
      mean_total_variation = mean(total_variation),
      .groups = "drop"
    )
)
cat(sprintf(
  "Mean same-profile absolute prompt shift: %.3f\n",
  mean(responses$prompt_shift)
))

cat("\nMonte Carlo summary\n")
print(simulation_summary, width = Inf)

cat("\nSubgroup gradient audit\n")
cat(sprintf("Coefficient RMSE: %.4f\n", gradient_rmse))
cat(sprintf(
  "Sign agreement: %.3f\n",
  mean(sign(gradients$llm_estimate) == sign(gradients$human_estimate))
))

cat("\nMatched-model summary\n")
print(matched_summary, width = Inf)

cat("\nStratified joint-donor baseline\n")
print(
  read_csv(paths$joint_donor, show_col_types = FALSE),
  width = Inf
)

cat("\nHuman sampling envelope and Qwen decisions\n")
print(
  read_csv(paths$qwen_envelope, show_col_types = FALSE),
  n = Inf, width = Inf
)

cat("\nTemporal-transfer summary including Qwen3-8B\n")
print(
  bind_rows(temporal, qwen_2021) |>
    select(model_label, ordinal_mae, total_variation, variance_ratio),
  width = Inf
)

cat("\nGenerated figures\n")
print(list.files(figure_dir, pattern = "\\.pdf$", full.names = TRUE))
