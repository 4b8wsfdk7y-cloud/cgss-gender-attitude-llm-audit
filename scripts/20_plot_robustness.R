# requires: dplyr, readr, tidyr, ggplot2, scales, ragg
# Results-page figure (docs/assets/) for the sampled-history robustness check.
# Reads only the aggregate tables written by scripts/14_analyze_sampled_history.py;
# no restricted data are needed. Theme and palette follow scripts/10_plot_v2.R.
#
# Usage: Rscript scripts/20_plot_robustness.R [V2_DIR] [FIGURE_DIR]
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

cell_levels <- c(
  "A0-B0-C0", "A0-B0-C1", "A1-B0-C0", "A1-B0-C1",
  "A0-B1-C0", "A0-B1-C1", "A1-B1-C0", "A1-B1-C1"
)
measure_palette <- c(
  "Expected scores, most probable history" = "grey55",
  "Expected scores, sampled history" = "#D55E00",
  "Drawn answers (response level)" = "#009E73"
)
measure_shapes <- c(
  "Expected scores, most probable history" = 1,
  "Expected scores, sampled history" = 16,
  "Drawn answers (response level)" = 17
)
human_color <- "#0072B2"

primary <- read_csv(
  file.path(v2_dir, "robustness_primary_by_cell.csv"), show_col_types = FALSE
)
comparators <- read_csv(
  file.path(v2_dir, "robustness_response_level_comparators.csv"), show_col_types = FALSE
)
stopifnot(nrow(primary) == 8)
human_median <- comparators$value[grepl("median", comparators$source)]
human_upper <- comparators$value[grepl("97.5th", comparators$source)]
cgss_full <- comparators$value[grepl("full sample", comparators$source)]
stopifnot(length(human_median) == 1, length(human_upper) == 1, length(cgss_full) == 1)

points <- primary |>
  transmute(
    cell = factor(cell, levels = cell_levels),
    history = answer_history == 1,
    `Expected scores, most probable history` = modal_expected,
    `Expected scores, sampled history` = if_else(history, sampled_expected, NA_real_),
    `Drawn answers (response level)` = response_level
  ) |>
  pivot_longer(-c(cell, history), names_to = "measure", values_to = "value") |>
  filter(!is.na(value)) |>
  mutate(measure = factor(measure, levels = names(measure_palette)))

p <- ggplot(points, aes(x = value, y = cell, color = measure, shape = measure)) +
  annotate(
    "rect", xmin = 0, xmax = human_upper, ymin = -Inf, ymax = Inf,
    fill = human_color, alpha = .07
  ) +
  geom_vline(xintercept = human_median, linetype = "dashed", color = human_color, linewidth = .4) +
  annotate(
    "text", x = human_upper, y = 8.6,
    label = sprintf("Human samples of 100 per wave:\nmedian %.2f, 97.5th percentile %.2f",
                    human_median, human_upper),
    hjust = -0.03, size = 2.5, color = human_color, lineheight = .95
  ) +
  geom_point(size = 2.1, stroke = .7) +
  scale_color_manual(values = measure_palette) +
  scale_shape_manual(values = measure_shapes) +
  scale_x_continuous(
    limits = c(0, 1), breaks = seq(0, 1, .2),
    labels = label_number(accuracy = .1), expand = expansion(mult = c(0, .02))
  ) +
  scale_y_discrete(expand = expansion(add = c(.6, 1.3))) +
  guides(color = guide_legend(nrow = 1)) +
  labs(
    title = "Answer history still drives coherence when earlier answers are sampled",
    subtitle = paste0(
      "Mean absolute correlation of A425 with the other four items; five seeded paths per ",
      "profile in the four history cells"
    ),
    x = "Mean absolute correlation", y = NULL,
    caption = paste0(
      "A = full five-item block shown; B = the model's earlier answers kept in context; ",
      "C = JSON output. Cells without history have no sampled-history value.\n",
      sprintf("Only drawn answers are on the scale of the human band; full weighted CGSS: %.3f. ",
              cgss_full),
      "Qwen3.5-9B (4-bit MLX), 300 CGSS profiles."
    )
  ) +
  theme_smr() +
  theme(
    panel.grid.major.x = element_line(color = "grey90", linewidth = .3),
    panel.grid.major.y = element_blank(),
    plot.caption = element_text(hjust = 0, lineheight = 1.05),
    plot.title.position = "plot",
    plot.caption.position = "plot"
  )

ggsave(
  file.path(figure_dir, "v2_robustness_by_cell.png"), p,
  device = ragg::agg_png, width = 7.2, height = 4.1, units = "in", dpi = 300, bg = "white"
)

points |>
  arrange(cell, measure) |>
  mutate(value = round(value, 3)) |>
  print(n = Inf)
