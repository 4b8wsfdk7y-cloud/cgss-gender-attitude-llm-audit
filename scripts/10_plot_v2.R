# requires: dplyr, readr, ggplot2, scales, ragg
# README figures for the v2 option-probability audit.
# Reads only the aggregate tables written by scripts/09_analyze_logprob_audit.py
# and scripts/08_export_human_benchmark.R; no restricted data are needed.
# Theme and palette follow paper/reproduce.R.
#
# Usage: Rscript scripts/10_plot_v2.R [V2_DIR] [FIGURE_DIR]
#   defaults: output/v2 output/figures

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
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

history_palette <- c(
  "Earlier answers in context" = "#D55E00",
  "No earlier answers" = "#E69F00"
)
hgb_color <- "#0072B2"

# Cells ordered bottom to top: no-history cells first, so the history split
# reads as two blocks.
cell_levels <- c(
  "A0-B0-C0", "A0-B0-C1", "A1-B0-C0", "A1-B0-C1",
  "A0-B1-C0", "A0-B1-C1", "A1-B1-C0", "A1-B1-C1"
)

add_history <- function(data) {
  data |>
    mutate(
      cell = factor(cell, levels = cell_levels),
      history = factor(
        if_else(grepl("B1", cell), "Earlier answers in context", "No earlier answers"),
        levels = names(history_palette)
      )
    )
}

design_note <- paste0(
  "A = full five-item block shown; B = the model's earlier answers kept in context; ",
  "C = JSON output (0 = off, 1 = on).\n",
  "Qwen3.5-9B (4-bit MLX), 300 CGSS profiles, one fp32 forward pass per profile, item, and cell."
)

primary <- read_csv(
  file.path(v2_dir, "v2_part2_primary_by_cell.csv"), show_col_types = FALSE
)
secondary <- read_csv(
  file.path(v2_dir, "v2_part2_secondary_by_cell.csv"), show_col_types = FALSE
)
part1 <- read_csv(file.path(v2_dir, "v2_part1_summary.csv"), show_col_types = FALSE)

coherence <- primary |>
  filter(cell %in% cell_levels) |>
  add_history()
stopifnot(nrow(coherence) == 8)
hgb_coherence <- primary$a425_mean_abs_correlation[
  primary$cell == "HGB expected scores (exploratory)"
]
stopifnot(length(hgb_coherence) == 1)

calibration <- secondary |>
  filter(cell %in% cell_levels) |>
  add_history()
stopifnot(nrow(calibration) == 8)
hgb_log_score <- part1$mean_over_15_wave_items[part1$metric == "hgb_matched_log_score"]
stopifnot(length(hgb_log_score) == 1)
uniform_log_score <- log(5)

# ---- Figure 1: inter-item coherence by cell ----
p_coherence <- ggplot(
  coherence, aes(x = a425_mean_abs_correlation, y = cell, color = history)
) +
  geom_vline(xintercept = hgb_coherence, linetype = "dashed", color = hgb_color, linewidth = .45) +
  annotate(
    "text", x = hgb_coherence, y = 8.55,
    label = sprintf("HGB expected scores %.2f", hgb_coherence),
    hjust = -0.04, size = 2.6, color = hgb_color
  ) +
  geom_linerange(aes(xmin = ci95_low, xmax = ci95_high), linewidth = .8) +
  geom_point(size = 2.2) +
  geom_text(
    aes(label = sprintf("%.2f", a425_mean_abs_correlation)),
    vjust = -1.05, size = 2.6, show.legend = FALSE
  ) +
  scale_color_manual(values = history_palette) +
  scale_x_continuous(
    limits = c(0, 1), breaks = seq(0, 1, .2),
    labels = label_number(accuracy = .1), expand = expansion(mult = c(0, .02))
  ) +
  scale_y_discrete(expand = expansion(add = c(.6, 1))) +
  labs(
    title = "Answer history drives the excess coherence",
    subtitle = paste0(
      "Mean absolute correlation of A425 with the other four items, from profile expected ",
      "scores; 95% profile-bootstrap intervals"
    ),
    x = "Mean absolute correlation", y = NULL,
    caption = paste0(
      design_note, "\n",
      "The CGSS response-level value (0.07) is computed on individual answers and is not ",
      "comparable to expected-score correlations."
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
  file.path(figure_dir, "v2_coherence_by_cell.png"), p_coherence,
  device = ragg::agg_png, width = 7.2, height = 3.9, units = "in", dpi = 300, bg = "white"
)

# ---- Figure 2: calibration against real respondents ----
p_calibration <- ggplot(
  calibration, aes(x = mean_matched_log_score, y = cell, color = history)
) +
  annotate(
    "rect", xmin = uniform_log_score, xmax = Inf, ymin = -Inf, ymax = Inf,
    fill = "#D55E00", alpha = .05
  ) +
  geom_vline(xintercept = uniform_log_score, linetype = "dashed", color = "grey25", linewidth = .45) +
  geom_vline(xintercept = hgb_log_score, linetype = "dashed", color = hgb_color, linewidth = .45) +
  annotate(
    "text", x = uniform_log_score, y = 8.55,
    label = sprintf("Uniform guess (ln 5 = %.2f)", uniform_log_score),
    hjust = -0.04, size = 2.6, color = "grey25"
  ) +
  annotate(
    "text", x = hgb_log_score, y = 8.55,
    label = sprintf("HGB %.2f", hgb_log_score),
    hjust = 1.06, size = 2.6, color = hgb_color
  ) +
  geom_point(size = 2.2) +
  geom_text(
    aes(label = sprintf("%.2f", mean_matched_log_score)),
    hjust = -0.45, size = 2.6, show.legend = FALSE
  ) +
  scale_color_manual(values = history_palette) +
  scale_x_continuous(
    limits = c(1, 3.4), breaks = seq(1, 3.4, .4),
    labels = label_number(accuracy = .1), expand = expansion(mult = c(0, .02))
  ) +
  scale_y_discrete(expand = expansion(add = c(.6, 1))) +
  labs(
    title = "Every condition is worse than a uniform guess",
    subtitle = paste0(
      "Matched log score: mean negative log probability of each profile's actual CGSS ",
      "answer (lower is better)"
    ),
    x = "Matched log score", y = NULL,
    caption = paste0(
      design_note, "\n",
      "HGB is the exploratory supervised baseline trained on survey outcomes. Shaded region: ",
      "worse than assigning 1/5 to every option."
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
  file.path(figure_dir, "v2_calibration_by_cell.png"), p_calibration,
  device = ragg::agg_png, width = 7.2, height = 3.9, units = "in", dpi = 300, bg = "white"
)

# ---- Console check: values drawn in the figures ----
cat("Coherence by cell (A425 mean |r|, 95% CI):\n")
print(
  coherence |>
    arrange(desc(cell)) |>
    transmute(cell, history, estimate = round(a425_mean_abs_correlation, 3),
              low = round(ci95_low, 3), high = round(ci95_high, 3)),
  n = Inf
)
cat(sprintf("HGB expected-score coherence: %.3f\n", hgb_coherence))
cat("\nMatched log score by cell:\n")
print(
  calibration |>
    arrange(desc(cell)) |>
    transmute(cell, log_score = round(mean_matched_log_score, 3)),
  n = Inf
)
cat(sprintf("Uniform guess: %.3f | HGB: %.3f\n", uniform_log_score, hgb_log_score))
