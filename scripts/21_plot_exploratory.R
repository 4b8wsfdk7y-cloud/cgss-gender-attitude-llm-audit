# requires: dplyr, readr, tidyr, ggplot2, scales, ragg
# Results-page figures (docs/assets/) for the v2 measurement models and the exploratory analyses.
# Reads only the aggregate tables written by scripts/12_measurement_models.R,
# scripts/17_subgroup_gradients.py, scripts/18_analyze_history_comparisons.py,
# and scripts/19_item_means_egalitarian.py; no restricted data are needed.
# Theme and palette follow scripts/10_plot_v2.R.
#
# Usage: Rscript scripts/21_plot_exploratory.R [V2_DIR] [FIGURE_DIR]
#   defaults: output/v2 output/figures

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(scales)
})

args <- commandArgs(trailingOnly = TRUE)
v2_dir <- if (length(args) >= 1) args[[1]] else file.path("output", "v2")
figure_dir <- if (length(args) >= 2) args[[2]] else file.path("output", "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

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

theme_dot <- function() {
  theme_smr() +
    theme(
      panel.grid.major.x = element_line(color = "grey90", linewidth = .3),
      panel.grid.major.y = element_blank(),
      plot.caption = element_text(hjust = 0, lineheight = 1.05),
      plot.title.position = "plot",
      plot.caption.position = "plot"
    )
}

save_figure <- function(plot, name, height) {
  ggsave(
    file.path(figure_dir, name), plot,
    device = ragg::agg_png, width = 7.2, height = height, units = "in", dpi = 300, bg = "white"
  )
}

cgss_color <- "#0072B2"
history_palette <- c(
  "Earlier answers in context" = "#D55E00",
  "No earlier answers" = "#E69F00"
)
factorial_cells <- c(
  "A0-B0-C0", "A0-B0-C1", "A1-B0-C0", "A1-B0-C1",
  "A0-B1-C0", "A0-B1-C1", "A1-B1-C0", "A1-B1-C1"
)
item_labels <- c(a421 = "A421", a422 = "A422", a423 = "A423", a424 = "A424", a425 = "A425")

# ---- Figure 1: measurement-model loadings ----
loadings <- read_csv(file.path(v2_dir, "part3_loadings.csv"), show_col_types = FALSE) |>
  filter(population %in% c("human", "S3")) |>
  mutate(
    item = factor(item_labels[item], levels = rev(item_labels)),
    source = factor(
      if_else(population == "human", "CGSS respondents (weighted)", "v1 sampled answers, 5 replicates"),
      levels = c("CGSS respondents (weighted)", "v1 sampled answers, 5 replicates")
    )
  )
stopifnot(sum(loadings$population == "human") == 5, sum(loadings$population == "S3") == 25)
# Named values: ggplot2 orders a discrete scale by the first layer it trains on.
loading_palette <- c("CGSS respondents (weighted)" = cgss_color, "v1 sampled answers, 5 replicates" = "#D55E00")
loading_shapes <- c("CGSS respondents (weighted)" = 18, "v1 sampled answers, 5 replicates" = 16)
a425_range <- range(loadings$loading[loadings$population == "S3" & loadings$item == "A425"])
a425_cgss <- loadings$loading[loadings$population == "human" & loadings$item == "A425"]

p_loadings <- ggplot(loadings, aes(x = loading, y = item, color = source, shape = source)) +
  annotate(
    "rect", xmin = -Inf, xmax = Inf, ymin = .5, ymax = 1.5, fill = "grey50", alpha = .07
  ) +
  geom_point(
    data = \(d) filter(d, population == "S3"),
    size = 1.9, alpha = .8, position = position_jitter(width = 0, height = .12, seed = 20261013)
  ) +
  geom_point(data = \(d) filter(d, population == "human"), size = 3.2) +
  annotate(
    "text", x = a425_range[2], y = 1,
    label = sprintf("A425: CGSS %.2f, v1 replicates %.2f-%.2f", a425_cgss, a425_range[1], a425_range[2]),
    hjust = -0.08, size = 2.6, color = "grey20"
  ) +
  scale_color_manual(values = loading_palette, limits = names(loading_palette)) +
  scale_shape_manual(values = loading_shapes, limits = names(loading_shapes)) +
  scale_x_continuous(
    limits = c(0, 1), breaks = seq(0, 1, .2),
    labels = label_number(accuracy = .1), expand = expansion(mult = c(0, .02))
  ) +
  labs(
    title = "The housework item joins the common factor in the synthetic answers",
    subtitle = "Standardized loadings in ordinal one-factor models (WLSMV), items coded so that higher is more egalitarian",
    x = "Standardized loading", y = NULL,
    caption = paste0(
      "CGSS: weighted complete cases, n = 31,177. v1: five replicates of 300 profiles. ",
      "Every 95% interval for the A425 difference from CGSS lies above zero.\n",
      "The v2 modal-answer populations are close to deterministic, give improper solutions, ",
      "and are not shown."
    )
  ) +
  theme_dot()

save_figure(p_loadings, "v2_measurement_loadings.png", 3.5)

# ---- Figure 2: direction of the item-mean errors ----
config_labels <- c(
  "v1 sampled (pooled)" = "v1 sampled answers",
  "V1-T1" = "V1-T1: v1 prompt, thinking open",
  "V1-T0" = "V1-T0: v1 prompt, thinking closed",
  setNames(factorial_cells, factorial_cells)
)
group_palette <- c(
  "First audit and v1 prompt" = "#CC79A7",
  "Target item only (A0)" = "#56B4E9",
  "Full item block (A1)" = "#D55E00"
)

item_means <- read_csv(file.path(v2_dir, "v2_item_means_egalitarian.csv"), show_col_types = FALSE) |>
  filter(cell %in% names(config_labels)) |>
  mutate(
    group = factor(
      case_when(
        cell %in% c("v1 sampled (pooled)", "V1-T1", "V1-T0") ~ "First audit and v1 prompt",
        startsWith(cell, "A0") ~ "Target item only (A0)",
        TRUE ~ "Full item block (A1)"
      ),
      levels = names(group_palette)
    ),
    config = factor(config_labels[cell], levels = rev(config_labels)),
    item = factor(item_labels[item], levels = item_labels)
  )
stopifnot(nrow(item_means) == 11 * 5)

p_means <- ggplot(item_means, aes(x = error, y = config, color = group)) +
  geom_vline(xintercept = 0, color = "grey25", linewidth = .4) +
  geom_segment(aes(x = 0, xend = error, yend = config), linewidth = .5, alpha = .6) +
  geom_point(size = 1.7) +
  facet_wrap(~item, nrow = 1) +
  scale_color_manual(values = group_palette) +
  scale_x_continuous(breaks = seq(-2, 2, 1), limits = c(-2, 1.5)) +
  labs(
    title = "The sign of the item-mean error depends on the configuration",
    subtitle = "Mean in egalitarian coding minus the wave-averaged weighted CGSS mean (positive = more egalitarian than CGSS)",
    x = "Error against CGSS (response-scale points)", y = NULL,
    caption = paste0(
      "A = full five-item block shown; B = the model's earlier answers kept in context; ",
      "C = JSON output. Factorial cells use T = 1 expected scores.\n",
      "V1-T1 and V1-T0 score the v1 joint prompt with the reasoning block open or closed. ",
      "CGSS means: A421 2.80, A422 3.17, A423 3.03, A424 3.99, A425 3.91."
    )
  ) +
  theme_dot() +
  theme(panel.spacing.x = unit(.9, "lines"))

save_figure(p_means, "v2_item_mean_errors.png", 4.3)

# ---- Figure 3: subgroup gradients ----
predictor_labels <- c(
  female = "Female",
  educ_years = "Years of education",
  urban_residence = "Urban residence"
)
outcome_labels <- c(public = "Public roles (A421-A424)", household = "Housework (A425)")

gradients <- read_csv(file.path(v2_dir, "exploratory_subgroup_gradients.csv"), show_col_types = FALSE) |>
  filter(
    cell %in% factorial_cells,
    outcome %in% names(outcome_labels),
    predictor %in% names(predictor_labels)
  ) |>
  mutate(
    predictor = factor(predictor_labels[predictor], levels = predictor_labels),
    outcome = factor(outcome_labels[outcome], levels = rev(outcome_labels)),
    history = factor(
      if_else(grepl("B1", cell), "Earlier answers in context", "No earlier answers"),
      levels = names(history_palette)
    )
  )
stopifnot(nrow(gradients) == 8 * 2 * 3)
benchmarks <- gradients |>
  distinct(predictor, outcome, human_estimate, v1_sampled_estimate) |>
  pivot_longer(
    c(human_estimate, v1_sampled_estimate), names_to = "source", values_to = "estimate"
  ) |>
  mutate(source = recode(
    source,
    human_estimate = "CGSS respondents (weighted)",
    v1_sampled_estimate = "v1 sampled answers"
  ))
stopifnot(nrow(benchmarks) == 2 * 3 * 2)

p_gradients <- ggplot(gradients, aes(x = model_estimate, y = outcome)) +
  geom_vline(xintercept = 0, color = "grey70", linewidth = .3) +
  geom_point(
    aes(color = history), size = 1.8, alpha = .85,
    position = position_jitter(width = 0, height = .14, seed = 20261013)
  ) +
  geom_point(
    data = benchmarks, aes(x = estimate, shape = source, fill = source),
    size = 2.8, color = "grey10", stroke = .5
  ) +
  facet_wrap(~predictor, nrow = 1, scales = "free_x") +
  scale_color_manual(values = history_palette, name = NULL) +
  scale_shape_manual(values = c(23, 24), name = NULL) +
  scale_fill_manual(values = c(cgss_color, "white"), name = NULL) +
  guides(color = guide_legend(order = 1), shape = guide_legend(order = 2), fill = guide_legend(order = 2)) +
  labs(
    title = "The model gives housework the social gradients of the public-role items",
    subtitle = "Coefficients from within-wave regressions of egalitarian-coded scores, averaged over waves; dots are the eight factorial cells",
    x = "Coefficient (response-scale points)", y = NULL,
    caption = paste0(
      "Female relative to male, per year of education, urban relative to rural residence; ",
      "age in decades is also in the model.\n",
      "Model: expected scores of 300 profiles per cell. CGSS: weighted full-sample regression ",
      "with the same specification."
    )
  ) +
  theme_dot() +
  theme(legend.box = "vertical", legend.spacing.y = unit(0, "pt"), panel.spacing.x = unit(1, "lines"))

save_figure(p_gradients, "v2_subgroup_gradients.png", 3.7)

# ---- Figure 4: human-history cells ----
source_map <- tribble(
  ~source, ~group, ~history, ~label,
  "H-A1-B1-C1 (respondent history)", "Item block, JSON", "Respondent's history", "H-A1-B1-C1: respondent's history",
  "A1-B1-C1 (model's modal history)", "Item block, JSON", "Model's modal history", "A1-B1-C1: model's modal history",
  "A1-B0-C1 (no history)", "Item block, JSON", "No history", "A1-B0-C1: no history",
  "H-A0-B1-C0 (respondent history)", "Single item, integer", "Respondent's history", "H-A0-B1-C0: respondent's history",
  "A0-B1-C0 (model's modal history)", "Single item, integer", "Model's modal history", "A0-B1-C0: model's modal history",
  "A0-B0-C0 (no history)", "Single item, integer", "No history", "A0-B0-C0: no history",
  "HGB with respondent history", "Survey-trained HGB", "Respondent's history", "HGB: respondent's history",
  "HGB without history", "Survey-trained HGB", "No history", "HGB: no history"
)
hh_palette <- c(
  "Respondent's history" = "#009E73",
  "Model's modal history" = "#D55E00",
  "No history" = "#E69F00"
)
metric_labels <- c(
  log_score = "A425 log score (lower is better)",
  slope = "Slope of A425 on the A421\u2013A424 mean"
)

dependence <- read_csv(file.path(v2_dir, "exploratory_history_dependence.csv"), show_col_types = FALSE)
calibration_hh <- read_csv(
  file.path(v2_dir, "exploratory_human_history_calibration.csv"), show_col_types = FALSE
)
cgss_slope <- dependence$slope[grepl("^CGSS weighted", dependence$source)]
matched_slope <- dependence$slope[dependence$source == "Matched respondents (observed answers)"]
stopifnot(length(cgss_slope) == 1, length(matched_slope) == 1)

history_points <- bind_rows(
  dependence |>
    inner_join(source_map, by = "source") |>
    transmute(source, metric = "slope", value = slope, low = slope_ci95_low, high = slope_ci95_high),
  calibration_hh |>
    inner_join(source_map, by = "source") |>
    transmute(source, metric = "log_score", value = log_score_a425, low = NA_real_, high = NA_real_)
) |>
  inner_join(source_map, by = "source") |>
  mutate(
    group = factor(group, levels = unique(source_map$group)),
    label = factor(label, levels = rev(source_map$label)),
    history = factor(history, levels = names(hh_palette)),
    metric = factor(metric_labels[metric], levels = metric_labels)
  )
stopifnot(nrow(history_points) == 16)

references <- tibble(
  metric = factor(metric_labels, levels = metric_labels),
  value = c(log(5), cgss_slope)
)
# Invisible points that fix each panel's x range.
ranges <- tibble(
  metric = factor(rep(metric_labels, each = 2), levels = metric_labels),
  value = c(1, 6, 0, .8),
  label = factor(source_map$label[1], levels = levels(history_points$label)),
  group = factor(source_map$group[1], levels = levels(history_points$group))
)

# The y scale follows the first layer it is trained on, so the intervals (all
# rows) come first and the range-setting blank layer last.
p_history <- ggplot(history_points, aes(x = value, y = label, color = history)) +
  geom_linerange(aes(xmin = low, xmax = high), linewidth = .8, na.rm = TRUE) +
  geom_vline(
    data = references[1, ], aes(xintercept = value), linetype = "dashed", color = "grey25", linewidth = .4
  ) +
  geom_vline(
    data = references[2, ], aes(xintercept = value), linetype = "dashed", color = cgss_color, linewidth = .4
  ) +
  geom_point(size = 2.2) +
  geom_label(
    aes(label = sprintf("%.2f", value)), vjust = -0.45, size = 2.4, fill = "white",
    border.colour = NA, label.padding = unit(.06, "lines"), show.legend = FALSE
  ) +
  geom_blank(data = ranges, aes(x = value, y = label), inherit.aes = FALSE) +
  facet_grid(group ~ metric, scales = "free", space = "free_y", switch = "y") +
  scale_color_manual(values = hh_palette) +
  scale_y_discrete(expand = expansion(add = c(.5, .8))) +
  labs(
    title = "Given a respondent's real answers, the model ties housework to them",
    subtitle = paste0(
      "296 matched profiles. Right: slope of the expected A425 score on the respondent's own ",
      "A421\u2013A424 mean, 95% profile-bootstrap intervals.\n",
      sprintf("Dashed lines: uniform guess, %.2f (left); weighted CGSS slope, %.2f (right)", log(5), cgss_slope)
    ),
    x = NULL, y = NULL,
    caption = paste0(
      "Respondent's history: the matched CGSS respondent's own earlier answers replace the ",
      "model's in the prompt. Egalitarian coding for the slopes.\n",
      sprintf("Slope among the matched respondents' observed answers: %.2f. ", matched_slope),
      "HGB: weighted gradient boosting, out of fold, with or without the earlier answers as features."
    )
  ) +
  theme_dot() +
  theme(
    strip.placement = "outside",
    strip.text.y.left = element_text(angle = 0, hjust = 1, face = "bold"),
    strip.text.x = element_text(size = 8),
    panel.spacing.x = unit(1.2, "lines"),
    panel.border = element_rect(fill = NA, color = "grey85", linewidth = .3)
  )

save_figure(p_history, "v2_human_history.png", 4.6)

# ---- Console check: values drawn in the figures ----
cat("A425 loadings: CGSS", round(a425_cgss, 3), "| v1", round(a425_range, 3), "\n")
item_means |>
  select(config, item, error) |>
  mutate(error = round(error, 2)) |>
  pivot_wider(names_from = item, values_from = error) |>
  print(n = Inf)
gradients |>
  group_by(predictor, outcome) |>
  summarise(
    model_min = round(min(model_estimate), 3), model_max = round(max(model_estimate), 3),
    cgss = round(first(human_estimate), 3), .groups = "drop"
  ) |>
  print(n = Inf)
history_points |>
  select(metric, label, value, low, high) |>
  mutate(across(c(value, low, high), \(x) round(x, 3))) |>
  arrange(metric, desc(label)) |>
  print(n = Inf)
