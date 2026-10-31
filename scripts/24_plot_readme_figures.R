# requires: dplyr, readr, tidyr, ggplot2, scales, patchwork, ragg
# The four README figures. Reads only aggregate tables in output/ and
# output/v2/; no restricted data are needed.
#
# All four share one theme and one palette. In the v2 figures colour shows
# what the model has in context when it answers: no earlier answers, its own
# earlier answers, or the matched respondent's own earlier answers. The three
# hues pass the all-pairs colour-vision checks as a set. CGSS is always black;
# comparators (v1, HGB, human sampling) are grey and carry their own shapes.
#
# Usage: Rscript scripts/24_plot_readme_figures.R [OUTPUT_DIR] [FIGURE_DIR]
#   defaults: output output/figures

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(scales)
  library(patchwork)
})

args <- commandArgs(trailingOnly = TRUE)
output_dir <- if (length(args) >= 1) args[[1]] else "output"
figure_dir <- if (length(args) >= 2) args[[2]] else file.path("output", "figures")
v2_dir <- file.path(output_dir, "v2")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

read <- function(...) read_csv(file.path(...), show_col_types = FALSE)

# ---- Palette and theme ----
context_palette <- c(
  "No earlier answers" = "#2a78d6",
  "Model's own earlier answers" = "#eb6834",
  "Respondent's own earlier answers" = "#1baf7a"
)
ink <- "#0b0b0b"
ink_secondary <- "#52514e"
ink_muted <- "#898781"
grid_color <- "#e1e0d9"
band_fill <- "#e9e8e3"

theme_readme <- function(base_size = 8.5) {
  theme_minimal(base_size = base_size, base_family = "sans") +
    theme(
      text = element_text(color = ink),
      plot.title = element_text(face = "bold", size = base_size + 0.5, margin = margin(b = 4)),
      plot.subtitle = element_text(color = ink_secondary, margin = margin(b = 6)),
      plot.caption = element_text(color = ink_secondary, size = base_size - 1, hjust = 0, lineheight = 1.1),
      plot.title.position = "plot",
      plot.caption.position = "plot",
      axis.text = element_text(color = ink_secondary),
      axis.title = element_text(color = ink_secondary, size = base_size - 0.5),
      panel.grid.major = element_line(color = grid_color, linewidth = .3),
      panel.grid.minor = element_blank(),
      panel.spacing = unit(10, "pt"),
      strip.text = element_text(hjust = 0, color = ink, size = base_size - 0.5, margin = margin(b = 3)),
      legend.position = "bottom",
      legend.title = element_blank(),
      legend.text = element_text(color = ink, size = base_size - 0.5),
      legend.key.size = unit(9, "pt"),
      legend.margin = margin(0, 0, 0, 0),
      legend.box.spacing = unit(4, "pt"),
      plot.margin = margin(4, 6, 4, 4)
    )
}

annotate_figure <- function(title, caption) {
  plot_annotation(
    title = title, caption = caption,
    theme = theme_readme() + theme(plot.title = element_text(face = "bold", size = 10.5))
  )
}

save_figure <- function(plot, name, height) {
  ggsave(
    file.path(figure_dir, name), plot,
    device = ragg::agg_png, width = 7.2, height = height, units = "in", dpi = 300, bg = "white"
  )
}

items <- c(a421 = "A421", a422 = "A422", a423 = "A423", a424 = "A424", a425 = "A425")
design_note <- "A = all five items shown, B = earlier answers in context, C = JSON output (1 = on, 0 = off)."

# Cells top to bottom: the four with answer history first.
cell_levels <- rev(c(
  "A1-B1-C1", "A1-B1-C0", "A0-B1-C1", "A0-B1-C0",
  "A1-B0-C1", "A1-B0-C0", "A0-B0-C1", "A0-B0-C0"
))
cell_context <- function(cell) {
  factor(
    case_when(
      startsWith(cell, "H-") ~ "Respondent's own earlier answers",
      grepl("B1", cell) ~ "Model's own earlier answers",
      TRUE ~ "No earlier answers"
    ),
    levels = names(context_palette)
  )
}
scale_context <- function(used = names(context_palette)) {
  scale_color_manual(values = context_palette, breaks = used, limits = names(context_palette))
}

# ---- Figure 1: the v1 audit ----
v1_colors <- c("CGSS (weighted)" = ink, "Qwen3.5-9B, v1" = "#2a78d6")

v1_means <- read(output_dir, "metrics_distribution.csv") |>
  filter(condition == "neutral_verbal") |>
  mutate(
    # A421-A424 are reverse-coded so that higher is more egalitarian on every item.
    human = if_else(item == "a425", human_mean, 6 - human_mean),
    model = if_else(item == "a425", llm_mean, 6 - llm_mean),
    item = factor(items[item], levels = items),
    wave = factor(wave)
  )
stopifnot(nrow(v1_means) == 15)
v1_means_long <- v1_means |>
  select(wave, item, `CGSS (weighted)` = human, `Qwen3.5-9B, v1` = model) |>
  pivot_longer(-c(wave, item), names_to = "source", values_to = "mean") |>
  mutate(source = factor(source, levels = names(v1_colors)))

p1a <- ggplot(v1_means, aes(x = item)) +
  geom_segment(aes(xend = item, y = human, yend = model), color = "#c3c2b7", linewidth = .7) +
  geom_point(data = v1_means_long, aes(y = mean, color = source), size = 2.1) +
  facet_wrap(~wave, nrow = 1) +
  scale_color_manual(values = v1_colors) +
  scale_y_continuous(limits = c(2, 4.6), breaks = 2:4) +
  labs(
    title = "a  Item means, coded so that higher is more egalitarian",
    x = NULL, y = "Mean (1\u20135)"
  ) +
  theme_readme() +
  theme(panel.grid.major.x = element_blank(), legend.position = "top", legend.justification = "left")

diagnostic_labels <- c(
  absolute_mean_error = "Absolute mean error",
  total_variation = "Total variation",
  variance_ratio = "Variance ratio (target 1)",
  correlation_rmse = "Correlation RMSE"
)
envelope <- read(output_dir, "metrics_qwen_sampling_envelope.csv") |>
  filter(metric %in% names(diagnostic_labels)) |>
  mutate(
    metric = factor(diagnostic_labels[metric], levels = diagnostic_labels),
    wave = factor(wave),
    status = factor(
      if_else(outside_human_envelope, "Qwen3.5-9B, outside the human range", "Qwen3.5-9B, inside the human range"),
      levels = c("Qwen3.5-9B, outside the human range", "Qwen3.5-9B, inside the human range")
    )
  )
stopifnot(nrow(envelope) == 12, sum(envelope$outside_human_envelope) == 11)
# Invisible points that fix each panel's x range.
envelope_ranges <- tibble(
  metric = factor(rep(diagnostic_labels, each = 2), levels = diagnostic_labels),
  value = c(0, .75, 0, .3, .6, 1.2, 0, .2),
  wave = factor(2012)
)

p1b <- ggplot(envelope, aes(y = wave)) +
  geom_blank(data = envelope_ranges, aes(x = value)) +
  geom_linerange(aes(xmin = p025, xmax = p975), color = band_fill, linewidth = 4.2) +
  geom_point(aes(x = median), shape = 124, size = 2.6, color = ink_muted) +
  geom_point(aes(x = qwen_estimate, shape = status), color = "#2a78d6", size = 2.1, stroke = .9) +
  facet_wrap(~metric, nrow = 1, scales = "free_x") +
  scale_shape_manual(values = c(16, 1), drop = FALSE) +
  scale_x_continuous(breaks = breaks_extended(3), expand = expansion(mult = c(.03, .08))) +
  scale_y_discrete(limits = c("2021", "2018", "2012")) +
  labs(
    title = "b  Core diagnostics against 1,000 human samples of the same design",
    subtitle = "Grey bar: 95% range of the human samples; tick: their median",
    x = NULL, y = NULL
  ) +
  theme_readme() +
  theme(
    panel.grid.major.y = element_blank(), panel.spacing = unit(16, "pt"),
    legend.position = "top", legend.justification = "left"
  )

figure1 <- (p1a / p1b) +
  plot_layout(heights = c(1.15, 1)) +
  annotate_figure(
    "v1: Qwen3.5-9B against CGSS 2012, 2018, and 2021",
    "300 profiles (100 per wave), five sampled answers per profile and item. 11 of the 12 core diagnostics fall outside the human range."
  )
save_figure(figure1, "readme_v1_overview.png", 4.9)

# ---- Figure 2: the v2 factorial experiment ----
panel2_levels <- c(
  "A425 coherence, expected scores",
  "A425 coherence, drawn answers",
  "Matched log score (lower is better)"
)
robustness <- read(v2_dir, "robustness_primary_by_cell.csv")
secondary <- read(v2_dir, "v2_part2_secondary_by_cell.csv")
comparators <- read(v2_dir, "robustness_response_level_comparators.csv")
part1 <- read(v2_dir, "v2_part1_summary.csv")
primary <- read(v2_dir, "v2_part2_primary_by_cell.csv")
stopifnot(nrow(robustness) == 8, all(cell_levels %in% secondary$cell))

hgb_coherence <- primary$a425_mean_abs_correlation[primary$cell == "HGB expected scores (exploratory)"]
hgb_log_score <- part1$mean_over_15_wave_items[part1$metric == "hgb_matched_log_score"]
human_median <- comparators$value[grepl("median", comparators$source)]
human_upper <- comparators$value[grepl("97.5th", comparators$source)]
cgss_full <- comparators$value[grepl("full sample", comparators$source)]
stopifnot(
  length(hgb_coherence) == 1, length(hgb_log_score) == 1,
  length(human_median) == 1, length(human_upper) == 1, length(cgss_full) == 1
)

history_kind <- c("Most probable earlier answers", "Sampled earlier answers")
points2 <- bind_rows(
  robustness |>
    transmute(panel = panel2_levels[1], cell, kind = history_kind[1],
              value = modal_expected, low = modal_expected_ci95_low, high = modal_expected_ci95_high),
  robustness |>
    filter(answer_history == 1) |>
    transmute(panel = panel2_levels[1], cell, kind = history_kind[2],
              value = sampled_expected, low = NA_real_, high = NA_real_),
  robustness |>
    transmute(panel = panel2_levels[2], cell, kind = history_kind[1],
              value = response_level, low = NA_real_, high = NA_real_),
  secondary |>
    filter(cell %in% cell_levels) |>
    transmute(panel = panel2_levels[3], cell, kind = history_kind[1],
              value = mean_matched_log_score, low = NA_real_, high = NA_real_)
) |>
  mutate(
    context = cell_context(cell),
    cell = factor(cell, levels = cell_levels),
    panel = factor(panel, levels = panel2_levels),
    kind = factor(kind, levels = history_kind)
  )
stopifnot(nrow(points2) == 28)

references2 <- tibble(
  panel = factor(panel2_levels[c(1, 2, 3, 3)], levels = panel2_levels),
  value = c(hgb_coherence, cgss_full, log(5), hgb_log_score),
  label = c(
    sprintf("HGB %.2f", hgb_coherence),
    sprintf("CGSS %.2f", cgss_full),
    sprintf("Uniform guess %.2f", log(5)),
    sprintf("HGB %.2f", hgb_log_score)
  ),
  color = c(ink_secondary, ink, ink_secondary, ink_secondary),
  hjust = c(-.08, -.08, -.06, 1.08)
)
band2 <- tibble(panel = factor(panel2_levels[2], levels = panel2_levels), xmin = 0, xmax = human_upper)
ranges2 <- tibble(
  panel = factor(rep(panel2_levels, each = 2), levels = panel2_levels),
  value = c(0, 1, 0, .65, .6, 3.2),
  cell = factor(cell_levels[1], levels = cell_levels)
)

p2 <- ggplot(points2, aes(x = value, y = cell)) +
  geom_blank(data = ranges2) +
  geom_rect(
    data = band2, aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf),
    inherit.aes = FALSE, fill = band_fill
  ) +
  geom_vline(data = references2, aes(xintercept = value), color = references2$color, linewidth = .4) +
  geom_text(
    data = references2, aes(x = value, y = 8.75, label = label, hjust = hjust),
    color = references2$color, size = 2.4, inherit.aes = FALSE
  ) +
  geom_linerange(aes(xmin = low, xmax = high, color = context), linewidth = .6, na.rm = TRUE) +
  geom_point(aes(color = context, shape = kind), size = 2, stroke = .8) +
  facet_wrap(~panel, nrow = 1, scales = "free_x") +
  scale_context(names(context_palette)[1:2]) +
  scale_shape_manual(values = setNames(c(16, 1), history_kind), breaks = history_kind[2]) +
  scale_x_continuous(
    breaks = \(x) if (max(x) > 2) 1:3 else pretty(x, 3), expand = expansion(mult = c(0, .03))
  ) +
  scale_y_discrete(expand = expansion(add = c(.6, 1.1))) +
  labs(x = NULL, y = NULL) +
  guides(color = guide_legend(order = 1), shape = guide_legend(order = 2)) +
  theme_readme() +
  theme(panel.grid.major.y = element_blank(), panel.spacing = unit(16, "pt"), legend.box = "horizontal")

figure2 <- p2 +
  annotate_figure(
    "v2: the eight presentation conditions",
    paste0(
      design_note, " 300 profiles; 95% profile-bootstrap intervals.\n",
      "Drawn answers: five seeded paths per profile; their correlations are on a different scale from expected scores.\n",
      sprintf("Grey band: human samples of 100 per wave, up to the 97.5th percentile (%.2f).", human_upper)
    )
  )
save_figure(figure2, "readme_v2_factorial.png", 3.85)

# ---- Figure 3: measurement and subgroup structure ----
source_colors <- c(
  context_palette[1:2],
  "CGSS (weighted)" = ink,
  "v1 sampled answers" = ink_secondary
)
source_shapes <- c(16, 16, 18, 2)
names(source_shapes) <- names(source_colors)
scale_sources <- function() {
  list(
    scale_color_manual(values = source_colors, limits = names(source_colors)),
    scale_shape_manual(values = source_shapes, limits = names(source_shapes))
  )
}

loadings <- read(v2_dir, "part3_loadings.csv") |>
  filter(population %in% c("human", "S3")) |>
  mutate(
    item = factor(items[item], levels = rev(items)),
    source = if_else(population == "human", "CGSS (weighted)", "v1 sampled answers"),
    # Five v1 replicates per item, spread vertically so that none hides another.
    offset = if_else(population == "human", 0, (replicate - 3) * .09)
  )
stopifnot(sum(loadings$population == "human") == 5, sum(loadings$population == "S3") == 25)
a425_v1 <- range(loadings$loading[loadings$population == "S3" & loadings$item == "A425"])
a425_cgss <- loadings$loading[loadings$population == "human" & loadings$item == "A425"]

p3a <- ggplot(loadings, aes(x = loading, y = as.numeric(item) + offset, color = source, shape = source)) +
  geom_point(size = 1.7, stroke = .6) +
  annotate(
    "text", x = a425_v1[2] + .03, y = 1,
    label = sprintf("A425: CGSS %.2f, v1 %.2f\u2013%.2f", a425_cgss, a425_v1[1], a425_v1[2]),
    hjust = 0, size = 2.4, color = ink
  ) +
  scale_sources() +
  scale_x_continuous(limits = c(0, 1), breaks = seq(0, 1, .2), expand = expansion(mult = c(0, .02))) +
  scale_y_continuous(breaks = seq_along(levels(loadings$item)), labels = levels(loadings$item)) +
  labs(title = "a  Standardized loadings, one-factor ordinal model", x = NULL, y = NULL) +
  theme_readme() +
  theme(panel.grid.major.y = element_blank(), legend.position = "none")

predictor_labels <- c(
  female = "Female",
  educ_years = "Years of education",
  urban_residence = "Urban residence"
)
outcome_labels <- c(public = "Public roles (A421\u2013A424)", household = "Housework (A425)")
gradients <- read(v2_dir, "exploratory_subgroup_gradients.csv") |>
  filter(cell %in% cell_levels, outcome %in% names(outcome_labels), predictor %in% names(predictor_labels)) |>
  mutate(
    predictor = factor(predictor_labels[predictor], levels = predictor_labels),
    outcome = factor(outcome_labels[outcome], levels = rev(outcome_labels)),
    source = as.character(cell_context(cell)),
    offset = if_else(grepl("B1", cell), -.2, .2)
  )
stopifnot(nrow(gradients) == 8 * 2 * 3)
gradient_benchmarks <- gradients |>
  distinct(predictor, outcome, `CGSS (weighted)` = human_estimate, `v1 sampled answers` = v1_sampled_estimate) |>
  pivot_longer(-c(predictor, outcome), names_to = "source", values_to = "model_estimate") |>
  mutate(offset = 0)
stopifnot(nrow(gradient_benchmarks) == 12)

p3b <- ggplot(
  bind_rows(gradients, gradient_benchmarks),
  aes(x = model_estimate, y = as.numeric(outcome) + offset, color = source, shape = source)
) +
  geom_vline(xintercept = 0, color = "#c3c2b7", linewidth = .4) +
  geom_point(data = \(d) filter(d, source %in% names(context_palette)), size = 1.7) +
  geom_point(data = \(d) filter(d, !source %in% names(context_palette)), size = 2.1, stroke = .7) +
  facet_wrap(~predictor, nrow = 1, scales = "free_x") +
  scale_sources() +
  scale_x_continuous(breaks = breaks_extended(3), expand = expansion(mult = c(.02, .05))) +
  scale_y_continuous(
    breaks = seq_along(levels(gradients$outcome)), labels = levels(gradients$outcome),
    expand = expansion(add = .45)
  ) +
  labs(
    title = "b  Subgroup coefficients, egalitarian coding (dots: the eight v2 conditions)",
    x = NULL, y = NULL
  ) +
  theme_readme() +
  theme(panel.grid.major.y = element_blank())

figure3 <- (p3a / p3b) +
  plot_layout(heights = c(1, 1)) +
  annotate_figure(
    "Measurement and subgroup structure against CGSS",
    paste0(
      "a: CGSS n = 31,177, weighted complete cases; v1: five replicates of 300 profiles. ",
      "v2 modal answers give improper solutions and are not shown.\n",
      "b: within-wave regressions on sex, years of education, age, and urban residence, averaged over waves."
    )
  )
save_figure(figure3, "readme_v2_structure.png", 4.6)

# ---- Figure 4: respondents' own answers and recalibration ----
dependence <- read(v2_dir, "exploratory_history_dependence.csv")
cgss_slope <- dependence$slope[grepl("^CGSS weighted", dependence$source)]
stopifnot(length(cgss_slope) == 1)
slope_map <- tribble(
  ~source, ~label, ~model,
  "H-A1-B1-C1 (respondent history)", "H-A1-B1-C1", "Qwen3.5-9B",
  "A1-B1-C1 (model's modal history)", "A1-B1-C1", "Qwen3.5-9B",
  "A1-B0-C1 (no history)", "A1-B0-C1", "Qwen3.5-9B",
  "H-A0-B1-C0 (respondent history)", "H-A0-B1-C0", "Qwen3.5-9B",
  "A0-B1-C0 (model's modal history)", "A0-B1-C0", "Qwen3.5-9B",
  "A0-B0-C0 (no history)", "A0-B0-C0", "Qwen3.5-9B",
  "HGB with respondent history", "HGB, with history", "Survey-trained HGB",
  "HGB without history", "HGB, no history", "Survey-trained HGB"
)
slopes4 <- dependence |>
  inner_join(slope_map, by = "source") |>
  mutate(
    context = factor(
      case_when(
        grepl("respondent history", source) ~ "Respondent's own earlier answers",
        grepl("modal history", source) ~ "Model's own earlier answers",
        TRUE ~ "No earlier answers"
      ),
      levels = names(context_palette)
    ),
    label = factor(label, levels = rev(slope_map$label)),
    model = factor(model, levels = c("Qwen3.5-9B", "Survey-trained HGB"))
  )
stopifnot(nrow(slopes4) == 8)

p4a <- ggplot(slopes4, aes(x = slope, y = label, color = context, shape = model)) +
  geom_vline(xintercept = cgss_slope, color = ink, linewidth = .4) +
  annotate("text", x = cgss_slope, y = 8.75, label = sprintf("CGSS %.2f", cgss_slope),
           hjust = -.08, size = 2.4, color = ink) +
  geom_linerange(aes(xmin = slope_ci95_low, xmax = slope_ci95_high), linewidth = .6) +
  geom_point(size = 2, stroke = .8) +
  scale_context() +
  scale_shape_manual(values = c(16, 15)) +
  scale_x_continuous(limits = c(0, .8), breaks = seq(0, .8, .2), expand = expansion(mult = c(0, .02))) +
  scale_y_discrete(expand = expansion(add = c(.6, 1.1))) +
  labs(
    title = "a  Slope of A425 on the respondent's\nA421\u2013A424 mean",
    x = NULL, y = NULL
  ) +
  guides(color = guide_legend(order = 1), shape = guide_legend(order = 2)) +
  theme_readme() +
  theme(panel.grid.major.y = element_blank(), legend.box = "vertical", legend.spacing.y = unit(2, "pt"))

recal_levels <- c(
  "Item-mean error (MAE)",
  "A425 slope, as in a",
  "A425 on urban residence",
  "A425 on female"
)
method_labels <- c(uncalibrated = "None", M1 = "M1", M2 = "M2")
at_n100 <- function(data) filter(data, method == "uncalibrated" | n == 100)
calibration_gradients <- read(v2_dir, "exploratory_calibration_gradients.csv") |>
  filter(outcome == "household", predictor %in% c("urban_residence", "female")) |>
  at_n100()
calibration_slopes <- read(v2_dir, "exploratory_calibration_history_slope.csv") |> at_n100()
recal <- bind_rows(
  read(v2_dir, "exploratory_calibration_summary.csv") |>
    at_n100() |>
    transmute(panel = recal_levels[1], cell, method, value = mean_abs_mean_error_median),
  calibration_slopes |>
    transmute(panel = recal_levels[2], cell, method, value = slope_median),
  calibration_gradients |>
    filter(predictor == "urban_residence") |>
    transmute(panel = recal_levels[3], cell, method, value = model_median),
  calibration_gradients |>
    filter(predictor == "female") |>
    transmute(panel = recal_levels[4], cell, method, value = model_median)
) |>
  mutate(
    context = cell_context(cell),
    panel = factor(panel, levels = recal_levels),
    method = factor(method_labels[method], levels = method_labels)
  )
stopifnot(nrow(recal) == 3 * (8 + 2 + 8 + 8))
recal_references <- tibble(
  panel = factor(recal_levels[2:4], levels = recal_levels),
  value = c(
    calibration_slopes$cgss_weighted_slope[[1]],
    calibration_gradients |> filter(predictor == "urban_residence") |> pull(cgss_estimate) |> first(),
    calibration_gradients |> filter(predictor == "female") |> pull(cgss_estimate) |> first()
  )
)

p4b <- ggplot(recal, aes(x = method, y = value, color = context, group = cell)) +
  geom_hline(data = recal_references, aes(yintercept = value), color = ink, linewidth = .4) +
  geom_line(linewidth = .5, alpha = .75) +
  geom_point(size = 1.5) +
  facet_wrap(~panel, ncol = 2, scales = "free_y") +
  scale_context() +
  scale_y_continuous(limits = c(0, NA), breaks = breaks_extended(3), expand = expansion(mult = c(0, .08))) +
  scale_x_discrete(expand = expansion(add = .25)) +
  labs(
    title = "b  Recalibration on the answers of\n100 matched respondents",
    x = NULL, y = NULL
  ) +
  theme_readme() +
  theme(panel.grid.major.x = element_blank(), legend.position = "none")

figure4 <- (p4a | p4b) +
  plot_layout(widths = c(1, 1.45), guides = "collect") +
  annotate_figure(
    "Follow-ups: the respondent's own answers, and recalibration on a few of them",
    paste0(
      "a: H- cells put the matched respondent's own earlier answers in the prompt; 296 profiles; 95% intervals.\n",
      "b: M1 = category offsets, M2 = offsets and an inverse temperature; medians over 1,000 replicates; ",
      "lines join one condition.\nBlack lines: weighted CGSS."
    )
  ) &
  theme(legend.box = "vertical")
save_figure(figure4, "readme_v2_followups.png", 4.4)

# ---- Console checks against the tables ----
print(v1_means |> transmute(wave, item, human = round(human, 2), model = round(model, 2)), n = 15)
print(envelope |> transmute(wave, metric, qwen = round(qwen_estimate, 3), p975 = round(p975, 3), status))
points2 |>
  group_by(panel, kind, context) |>
  summarise(min = round(min(value), 2), max = round(max(value), 2), .groups = "drop") |>
  print()
print(references2 |> select(panel, label))
print(slopes4 |> transmute(label, slope = round(slope, 2), low = round(slope_ci95_low, 2), high = round(slope_ci95_high, 2)))
recal |>
  group_by(panel, method) |>
  summarise(min = round(min(value), 2), max = round(max(value), 2), .groups = "drop") |>
  print(n = 12)
print(recal_references)
