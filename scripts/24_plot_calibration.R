# requires: dplyr, readr, tidyr, ggplot2, scales, ragg
# README figure for the recalibration analysis. Reads only the aggregate tables
# written by scripts/22_calibrate_with_human_sample.py; no restricted data are
# needed. Theme and palette follow scripts/21_plot_exploratory.R.
#
# Usage: Rscript scripts/24_plot_calibration.R [V2_DIR] [FIGURE_DIR]
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
      plot.caption = element_text(size = base_size - 2, color = "grey35", hjust = 0, lineheight = 1.05),
      panel.grid.minor = element_blank(),
      panel.grid.major.x = element_blank(),
      axis.title = element_text(size = base_size),
      strip.text = element_text(face = "bold", hjust = 0),
      legend.position = "bottom",
      legend.title = element_blank(),
      plot.title.position = "plot",
      plot.caption.position = "plot",
      plot.margin = margin(5, 7, 5, 5)
    )
}

cgss_color <- "#0072B2"
method_palette <- c(
  "Uncalibrated" = "grey45",
  "Offsets (M1)" = "#D55E00",
  "Offsets and temperature (M2)" = "#009E73"
)
method_labels <- c(uncalibrated = "Uncalibrated", M1 = "Offsets (M1)", M2 = "Offsets and temperature (M2)")
size_levels <- c("None", "50", "100", "200", "All 296")
panel_levels <- c(
  "Mean absolute error of item means (8 cells)",
  "A425 slope on the respondent's A421–A424 mean\n(two human-history cells)",
  "A425 on urban residence (8 cells): spurious in the model",
  "A425 on female (8 cells): a real gradient"
)

size_label <- function(n) {
  recode(as.character(n), "0" = "None", "296" = "All 296")
}

summary <- read_csv(file.path(v2_dir, "exploratory_calibration_summary.csv"), show_col_types = FALSE)
gradients <- read_csv(file.path(v2_dir, "exploratory_calibration_gradients.csv"), show_col_types = FALSE)
slopes <- read_csv(file.path(v2_dir, "exploratory_calibration_history_slope.csv"), show_col_types = FALSE)

points <- bind_rows(
  summary |>
    transmute(panel = panel_levels[1], cell, method, n,
              value = mean_abs_mean_error_median, low = NA_real_, high = NA_real_),
  slopes |>
    transmute(panel = panel_levels[2], cell, method, n,
              value = slope_median, low = slope_ci95_low, high = slope_ci95_high),
  gradients |>
    filter(outcome == "household", predictor == "urban_residence") |>
    transmute(panel = panel_levels[3], cell, method, n, value = model_median,
              low = NA_real_, high = NA_real_),
  gradients |>
    filter(outcome == "household", predictor == "female") |>
    transmute(panel = panel_levels[4], cell, method, n, value = model_median,
              low = NA_real_, high = NA_real_)
) |>
  mutate(
    panel = factor(panel, levels = panel_levels),
    method = factor(method_labels[method], levels = names(method_palette)),
    size = factor(size_label(n), levels = size_levels)
  )

references <- tibble(
  panel = factor(panel_levels[2:4], levels = panel_levels),
  value = c(
    slopes$cgss_weighted_slope[[1]],
    gradients |> filter(outcome == "household", predictor == "urban_residence") |>
      pull(cgss_estimate) |> first(),
    gradients |> filter(outcome == "household", predictor == "female") |>
      pull(cgss_estimate) |> first()
  )
)

dodge <- position_dodge(width = .55)
figure <- ggplot(points, aes(x = size, y = value, color = method, group = method)) +
  geom_hline(data = references, aes(yintercept = value), color = cgss_color,
             linetype = "dashed", linewidth = .45) +
  geom_linerange(aes(ymin = low, ymax = high), position = dodge, linewidth = .5, na.rm = TRUE) +
  geom_point(position = dodge, size = 1.7, alpha = .75) +
  facet_wrap(~panel, ncol = 2, scales = "free_y") +
  scale_color_manual(values = method_palette, limits = names(method_palette)) +
  scale_y_continuous(labels = label_number(accuracy = .01)) +
  labs(
    title = "Recalibrating on a few real answers fixes the margins, not the structure",
    subtitle = paste0(
      "Each point is one cell's median over 1,000 replicates; vertical lines are 95% intervals. ",
      "Dashed blue line: weighted CGSS."
    ),
    x = "Matched respondents used for recalibration",
    y = NULL,
    caption = paste0(
      "Offsets alone move every profile alike: item means approach CGSS while the history slope and the ",
      "urban gradient on A425 stay.\nAdding a temperature removes them only by flattening all profiles, ",
      "which also removes the real gender gap on A425."
    )
  ) +
  theme_smr()

save_figure <- function(plot, name, height) {
  ggsave(
    file.path(figure_dir, name), plot,
    device = ragg::agg_png, width = 7.2, height = height, units = "in", dpi = 300, bg = "white"
  )
}
save_figure(figure, "v2_recalibration.png", 5.6)

# Console checks against the tables.
points |>
  group_by(panel, method, size) |>
  summarise(min = min(value), max = max(value), .groups = "drop") |>
  mutate(across(c(min, max), \(x) round(x, 3))) |>
  print(n = 40)
print(references)
