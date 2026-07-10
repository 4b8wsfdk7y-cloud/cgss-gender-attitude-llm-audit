# requires: dplyr, readr, tidyr, psych

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(psych)
})

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1]]))
} else {
  normalizePath("scripts/05_extended_validation.R")
}
audit_root <- normalizePath(file.path(dirname(script_path), ".."))
output_dir <- file.path(audit_root, "output")
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)

benchmark <- readRDS(file.path(
  audit_root, "data", "source", "dimension_pilot_results.rds"
))
profiles <- read_csv(
  file.path(audit_root, "data", "profiles_pilot.csv"),
  show_col_types = FALSE
) |>
  mutate(wave = as.character(wave))
responses <- read_csv(
  file.path(output_dir, "responses.csv"),
  show_col_types = FALSE
) |>
  mutate(wave = as.character(wave))

items <- paste0("a42", 1:5)
eq_items <- paste0("eq_", items)

add_equality_scores <- function(data) {
  data |>
    mutate(
      eq_a421 = 6 - a421,
      eq_a422 = 6 - a422,
      eq_a423 = 6 - a423,
      eq_a424 = 6 - a424,
      eq_a425 = a425
    )
}

weighted_distribution <- function(x, w) {
  keep <- !is.na(x) & !is.na(w) & w > 0
  mass <- vapply(1:5, function(value) sum(w[keep & x == value]), numeric(1))
  mass / sum(mass)
}

weighted_variance <- function(x, w) {
  keep <- !is.na(x) & !is.na(w) & w > 0
  center <- weighted.mean(x[keep], w[keep])
  weighted.mean((x[keep] - center)^2, w[keep])
}

weighted_correlation <- function(data, variables, weight) {
  output <- diag(length(variables))
  dimnames(output) <- list(variables, variables)
  for (row in seq_along(variables)) {
    for (column in seq_len(row - 1L)) {
      x <- data[[variables[[row]]]]
      y <- data[[variables[[column]]]]
      keep <- !is.na(x) & !is.na(y) & !is.na(weight) & weight > 0
      w <- weight[keep]
      x <- x[keep]
      y <- y[keep]
      x_center <- weighted.mean(x, w)
      y_center <- weighted.mean(y, w)
      covariance <- weighted.mean(
        (x - x_center) * (y - y_center), w
      )
      correlation <- covariance / sqrt(
        weighted.mean((x - x_center)^2, w) *
          weighted.mean((y - y_center)^2, w)
      )
      output[row, column] <- correlation
      output[column, row] <- correlation
    }
  }
  output
}

correlation_metrics <- function(correlation, target) {
  values <- abs(correlation[1:4, 5])
  tibble(
    correlation_rmse = sqrt(mean((correlation - target)^2, na.rm = TRUE)),
    a425_mean_abs_correlation = mean(values, na.rm = TRUE)
  )
}

largest_remainder <- function(shares, total) {
  raw <- shares / sum(shares) * total
  allocation <- floor(raw)
  remaining <- total - sum(allocation)
  if (remaining > 0) {
    add_to <- order(raw - allocation, decreasing = TRUE)[seq_len(remaining)]
    allocation[add_to] <- allocation[add_to] + 1L
  }
  as.integer(allocation)
}

sample_population <- function(data, n_target) {
  framed <- data |>
    mutate(
      education_group = case_when(
        educ_years <= 9 ~ "low",
        educ_years <= 12 ~ "middle",
        TRUE ~ "high"
      ),
      stratum = interaction(
        female, education_group, urban_residence,
        drop = TRUE, sep = "_"
      )
    )
  allocation <- framed |>
    group_by(stratum) |>
    summarise(
      weighted_size = sum(weight),
      available_n = n(),
      .groups = "drop"
    ) |>
    mutate(target_n = largest_remainder(weighted_size, n_target))
  selected <- framed |>
    inner_join(allocation |> select(stratum, target_n), by = "stratum") |>
    group_by(stratum) |>
    group_modify(\(data, key) {
      n_take <- unique(data$target_n)
      data[sample.int(
        nrow(data), n_take, replace = FALSE, prob = data$weight
      ), , drop = FALSE]
    }) |>
    ungroup()
  stopifnot(nrow(selected) == n_target)
  selected
}

human_targets <- lapply(names(benchmark$samples), function(wave) {
  data <- benchmark$samples[[wave]] |>
    add_equality_scores()
  distributions <- lapply(
    eq_items, \(item) weighted_distribution(data[[item]], data$weight)
  )
  variances <- vapply(
    eq_items, \(item) weighted_variance(data[[item]], data$weight), numeric(1)
  )
  weighted_cor <- weighted_correlation(data, eq_items, data$weight)
  unweighted_cor <- cor(data[eq_items], use = "pairwise.complete.obs")
  polychoric_cor <- suppressWarnings(
    psych::polychoric(data[eq_items], correct = 0)$rho
  )
  list(
    wave = wave,
    data = data,
    distributions = distributions,
    variances = variances,
    weighted_cor = weighted_cor,
    unweighted_cor = unweighted_cor,
    polychoric_cor = polychoric_cor
  )
})
names(human_targets) <- names(benchmark$samples)

correlation_sensitivity <- bind_rows(lapply(human_targets, function(target) {
  bind_rows(
    tibble(
      wave = target$wave,
      method = "unweighted_pearson",
      a425_mean_abs_correlation =
        mean(abs(target$unweighted_cor[1:4, 5]))
    ),
    tibble(
      wave = target$wave,
      method = "survey_weighted_pearson",
      a425_mean_abs_correlation =
        mean(abs(target$weighted_cor[1:4, 5]))
    ),
    tibble(
      wave = target$wave,
      method = "unweighted_polychoric",
      a425_mean_abs_correlation =
        mean(abs(target$polychoric_cor[1:4, 5]))
    )
  )
}))

sample_metrics <- function(sample, target) {
  sample <- add_equality_scores(sample)
  item_metrics <- bind_rows(lapply(seq_along(eq_items), function(index) {
    item <- eq_items[[index]]
    observed <- sample[[item]]
    observed <- observed[!is.na(observed)]
    distribution <- tabulate(observed, nbins = 5) / length(observed)
    target_distribution <- target$distributions[[index]]
    tibble(
      absolute_mean_error = abs(
        sum((1:5) * distribution) -
          sum((1:5) * target_distribution)
      ),
      total_variation = .5 * sum(abs(distribution - target_distribution)),
      variance_ratio = mean(
        (observed - mean(observed))^2
      ) / target$variances[[index]]
    )
  }))
  sample_cor <- cor(sample[eq_items], use = "pairwise.complete.obs")
  bind_cols(
    item_metrics |>
      summarise(across(everything(), \(x) mean(x, na.rm = TRUE))),
    correlation_metrics(sample_cor, target$weighted_cor)
  )
}

set.seed(20260706)
human_resampling_repetitions <- 1000L
human_resampling <- bind_rows(lapply(human_targets, function(target) {
  bind_rows(lapply(seq_len(human_resampling_repetitions), function(replication) {
    sample_population(target$data, 100L) |>
      sample_metrics(target) |>
      mutate(
        wave = target$wave,
        replication = replication,
        .before = 1
      )
  }))
}))

human_sampling_envelope <- human_resampling |>
  pivot_longer(
    c(
      absolute_mean_error, total_variation, variance_ratio,
      correlation_rmse, a425_mean_abs_correlation
    ),
    names_to = "metric", values_to = "value"
  ) |>
  group_by(wave, metric) |>
  summarise(
    p025 = quantile(value, .025, na.rm = TRUE),
    median = median(value, na.rm = TRUE),
    p975 = quantile(value, .975, na.rm = TRUE),
    repetitions = n(),
    .groups = "drop"
  )

neutral_joint <- responses |>
  filter(
    success, condition == "neutral_verbal", `repeat` %in% 1:5
  ) |>
  mutate(across(all_of(items), as.integer)) |>
  add_equality_scores()

qwen_envelope_comparison <- bind_rows(lapply(human_targets, function(target) {
  data <- neutral_joint |>
    filter(wave == target$wave)
  item_metrics <- bind_rows(lapply(seq_along(eq_items), function(index) {
    item <- eq_items[[index]]
    observed <- data[[item]]
    distribution <- tabulate(observed, nbins = 5) / length(observed)
    target_distribution <- target$distributions[[index]]
    tibble(
      absolute_mean_error = abs(
        sum((1:5) * distribution) -
          sum((1:5) * target_distribution)
      ),
      total_variation = .5 * sum(abs(distribution - target_distribution)),
      variance_ratio = mean(
        (observed - mean(observed))^2
      ) / target$variances[[index]]
    )
  }))
  correlation_rows <- data |>
    group_by(`repeat`) |>
    group_modify(\(repeat_data, key) {
      correlation_metrics(
        cor(repeat_data[eq_items], use = "pairwise.complete.obs"),
        target$weighted_cor
      )
    }) |>
    ungroup()
  bind_cols(
    item_metrics |>
      summarise(across(everything(), mean, na.rm = TRUE)),
    correlation_rows |>
      summarise(across(
        c(correlation_rmse, a425_mean_abs_correlation),
        mean, na.rm = TRUE
      ))
  ) |>
    mutate(wave = target$wave, .before = 1)
}))

qwen_envelope_comparison <- qwen_envelope_comparison |>
  pivot_longer(
    -wave, names_to = "metric", values_to = "qwen_estimate"
  ) |>
  left_join(human_sampling_envelope, by = c("wave", "metric")) |>
  mutate(
    outside_human_envelope = qwen_estimate < p025 | qwen_estimate > p975
  )

profiles_for_donors <- profiles |>
  mutate(
    education_group = case_when(
      educ_years <= 9 ~ "low",
      educ_years <= 12 ~ "middle",
      TRUE ~ "high"
    )
  )

set.seed(20260706)
joint_donor_rows <- vector("list", nrow(profiles_for_donors) * 5L)
row_index <- 1L
for (profile_index in seq_len(nrow(profiles_for_donors))) {
  profile <- profiles_for_donors[profile_index, ]
  pool <- human_targets[[profile$wave]]$data |>
    mutate(
      education_group = case_when(
        educ_years <= 9 ~ "low",
        educ_years <= 12 ~ "middle",
        TRUE ~ "high"
      )
    ) |>
    filter(
      female == profile$female,
      education_group == profile$education_group,
      urban_residence == profile$urban_residence,
      respondent_id != profile$respondent_id,
      if_all(all_of(eq_items), ~ !is.na(.x))
    )
  if (nrow(pool) < 20) {
    pool <- human_targets[[profile$wave]]$data |>
      filter(
        female == profile$female,
        respondent_id != profile$respondent_id,
        if_all(all_of(eq_items), ~ !is.na(.x))
      )
  }
  donor_indices <- sample.int(
    nrow(pool), 5L, replace = TRUE, prob = pool$weight
  )
  for (repeat_index in 1:5) {
    donor <- pool[donor_indices[[repeat_index]], ]
    joint_donor_rows[[row_index]] <- bind_cols(
      tibble(
        profile_id = profile$profile_id,
        wave = profile$wave,
        `repeat` = repeat_index,
        donor_pool_n = nrow(pool)
      ),
      donor |>
        select(all_of(eq_items))
    )
    row_index <- row_index + 1L
  }
}
joint_donor <- bind_rows(joint_donor_rows)

cat("\nFive-row joint-donor preview:\n")
print(joint_donor |> slice_head(n = 5), n = 5, width = Inf)

matched_human <- profiles |>
  add_equality_scores() |>
  select(profile_id, wave, all_of(eq_items))

joint_donor_profile <- joint_donor |>
  group_by(profile_id, wave) |>
  summarise(
    across(all_of(eq_items), mean),
    .groups = "drop"
  ) |>
  left_join(
    matched_human,
    by = c("profile_id", "wave"),
    suffix = c("_predicted", "_human")
  )

joint_donor_omae <- bind_rows(lapply(eq_items, function(item) {
  joint_donor_profile |>
    transmute(
      wave,
      error = abs(
        .data[[paste0(item, "_predicted")]] -
          .data[[paste0(item, "_human")]]
      )
    ) |>
    filter(!is.na(error)) |>
    summarise(omae = mean(error), .by = wave)
})) |>
  summarise(omae = mean(omae))

joint_donor_distribution_metrics <- bind_rows(lapply(
  human_targets, function(target) {
    donor_wave <- joint_donor |>
      filter(wave == target$wave)
    matched_wave <- matched_human |>
      filter(wave == target$wave)
    item_metrics <- bind_rows(lapply(seq_along(eq_items), function(index) {
      values <- donor_wave[[eq_items[[index]]]]
      distribution <- tabulate(values, nbins = 5) / length(values)
      matched_values <- matched_wave[[eq_items[[index]]]]
      matched_values <- matched_values[!is.na(matched_values)]
      target_distribution <- tabulate(
        matched_values, nbins = 5
      ) / length(matched_values)
      tibble(
        total_variation = .5 * sum(abs(
          distribution - target_distribution
        )),
        variance_ratio = mean(
          (values - mean(values))^2
        ) / mean((matched_values - mean(matched_values))^2)
      )
    }))
    correlations <- donor_wave |>
      group_by(`repeat`) |>
      group_modify(\(repeat_data, key) {
        correlation_metrics(
          cor(repeat_data[eq_items], use = "pairwise.complete.obs"),
          target$weighted_cor
        )
      })
    bind_cols(
      item_metrics |>
        summarise(across(everything(), mean)),
      correlations |>
        summarise(across(
          c(correlation_rmse, a425_mean_abs_correlation), mean
        ))
    ) |>
      mutate(wave = target$wave, .before = 1)
  }
))

joint_donor_baseline <- joint_donor_distribution_metrics |>
  summarise(
    model = "stratified_joint_donor",
    omae = joint_donor_omae$omae,
    across(
      c(
        total_variation, variance_ratio,
        correlation_rmse, a425_mean_abs_correlation
      ),
      mean
    )
  )

write_csv(
  correlation_sensitivity,
  file.path(output_dir, "metrics_correlation_sensitivity.csv")
)
write_csv(
  human_resampling,
  file.path(output_dir, "metrics_human_sampling_replications.csv")
)
write_csv(
  human_sampling_envelope,
  file.path(output_dir, "metrics_human_sampling_envelope.csv")
)
write_csv(
  qwen_envelope_comparison,
  file.path(output_dir, "metrics_qwen_sampling_envelope.csv")
)
write_csv(
  joint_donor_baseline,
  file.path(output_dir, "metrics_joint_donor_baseline.csv")
)

cat("\nCorrelation sensitivity:\n")
print(correlation_sensitivity, n = Inf)
cat("\nHuman sampling envelope:\n")
print(human_sampling_envelope, n = Inf)
cat("\nQwen relative to human sampling envelope:\n")
print(qwen_envelope_comparison, n = Inf)
cat("\nJoint outcome-informed baseline:\n")
print(joint_donor_baseline, n = Inf)
