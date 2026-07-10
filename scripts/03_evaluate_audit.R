# requires: dplyr, readr, tidyr, ggplot2, psych

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
  library(tidyr)
  library(ggplot2)
  library(psych)
})

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1]]))
} else {
  normalizePath("scripts/03_evaluate_audit.R")
}
audit_root <- normalizePath(file.path(dirname(script_path), ".."))
args <- commandArgs(trailingOnly = TRUE)
output_arg <- grep("^--output-dir=", args, value = TRUE)
output_dir <- if (length(output_arg)) {
  normalizePath(sub("^--output-dir=", "", output_arg[[1]]), mustWork = FALSE)
} else {
  file.path(audit_root, "output")
}
figure_dir <- file.path(output_dir, "figures")
dir.create(figure_dir, recursive = TRUE, showWarnings = FALSE)

human_path <- file.path(audit_root, "data", "profiles_pilot.csv")
llm_path <- file.path(output_dir, "responses.csv")
independent_path <- file.path(output_dir, "responses_independent.csv")
benchmark_path <- file.path(
  audit_root, "data", "source", "dimension_pilot_results.rds"
)
if (!all(file.exists(c(human_path, llm_path, benchmark_path)))) {
  stop("Missing profiles, responses, or benchmark RDS.")
}

human_profiles <- read_csv(human_path, show_col_types = FALSE)
llm_raw <- read_csv(llm_path, show_col_types = FALSE) |>
  mutate(wave = as.character(wave))
independent_raw <- if (file.exists(independent_path)) {
  read_csv(independent_path, show_col_types = FALSE) |>
    mutate(wave = as.character(wave))
} else {
  tibble()
}
benchmark <- readRDS(benchmark_path)
items <- paste0("a42", 1:5)

llm_valid <- llm_raw |>
  filter(success) |>
  mutate(across(all_of(items), as.integer)) |>
  filter(if_all(all_of(items), ~ .x %in% 1:5)) |>
  left_join(
    human_profiles |>
      select(
        profile_id, female, age, educ_years, urban_residence,
        education_group, starts_with("a42")
      ) |>
      rename_with(~ paste0(.x, "_human"), all_of(items)),
    by = "profile_id"
  )

condition_counts <- llm_valid |>
  distinct(profile_id, wave, condition) |>
  count(condition, wave, name = "profiles")
complete_conditions <- condition_counts |>
  group_by(condition) |>
  summarise(
    waves = n_distinct(wave),
    min_profiles_per_wave = min(profiles),
    .groups = "drop"
  ) |>
  filter(waves == 3, min_profiles_per_wave >= 60) |>
  pull(condition)

call_quality <- llm_raw |>
  summarise(
    calls = n(),
    successful_calls = sum(success),
    success_rate = mean(success),
    median_seconds = median(elapsed_seconds[success], na.rm = TRUE),
    p90_seconds = quantile(
      elapsed_seconds[success], .9, na.rm = TRUE, names = FALSE
    )
  )

full_human <- bind_rows(lapply(names(benchmark$samples), function(wave) {
  benchmark$samples[[wave]] |>
    mutate(wave = wave)
}))

weighted_distribution <- function(data, value, weight) {
  data |>
    filter(
      !is.na(.data[[value]]),
      !is.na(.data[[weight]]),
      .data[[weight]] > 0
    ) |>
    count(
      wave, response = .data[[value]],
      wt = .data[[weight]], name = "mass"
    ) |>
    group_by(wave) |>
    mutate(probability = mass / sum(mass)) |>
    ungroup() |>
    select(wave, response, probability)
}

human_distributions <- bind_rows(lapply(items, function(item) {
  weighted_distribution(full_human, item, "weight") |>
    mutate(item = item, source = "CGSS")
}))
llm_long <- llm_valid |>
  select(profile_id, wave, condition, `repeat`, all_of(items)) |>
  pivot_longer(all_of(items), names_to = "item", values_to = "response")
llm_distributions <- llm_long |>
  count(wave, condition, item, response, name = "mass") |>
  group_by(wave, condition, item) |>
  mutate(probability = mass / sum(mass)) |>
  ungroup() |>
  mutate(source = "LLM")

distribution_grid <- expand_grid(
  wave = unique(llm_long$wave),
  condition = unique(llm_long$condition),
  item = items,
  response = 1:5
) |>
  left_join(
    human_distributions |>
      select(wave, item, response, human_probability = probability),
    by = c("wave", "item", "response")
  ) |>
  left_join(
    llm_distributions |>
      select(wave, condition, item, response, llm_probability = probability),
    by = c("wave", "condition", "item", "response")
  ) |>
  mutate(across(c(human_probability, llm_probability), ~ replace_na(.x, 0)))

distribution_metrics <- distribution_grid |>
  group_by(wave, condition, item) |>
  summarise(
    human_mean = sum(response * human_probability),
    llm_mean = sum(response * llm_probability),
    mean_error = llm_mean - human_mean,
    total_variation = .5 * sum(abs(llm_probability - human_probability)),
    .groups = "drop"
  )

variance_metrics <- llm_long |>
  group_by(wave, condition, item) |>
  summarise(llm_variance = var(response), .groups = "drop") |>
  left_join(
    bind_rows(lapply(names(benchmark$samples), function(wave) {
      data <- benchmark$samples[[wave]]
      bind_rows(lapply(items, function(item) {
        complete <- !is.na(data[[item]]) &
          !is.na(data$weight) &
          data$weight > 0
        center <- weighted.mean(data[[item]][complete], data$weight[complete])
        variance <- sum(
          data$weight[complete] * (data[[item]][complete] - center)^2
        ) / sum(data$weight[complete])
        tibble(wave = wave, item = item, human_variance = variance)
      }))
    })),
    by = c("wave", "item")
  ) |>
  mutate(variance_ratio = llm_variance / human_variance)

paired_metrics <- bind_rows(lapply(items, function(item) {
  llm_valid |>
    transmute(
      wave, condition, item = item,
      absolute_error = abs(.data[[item]] - .data[[paste0(item, "_human")]]),
      exact = .data[[item]] == .data[[paste0(item, "_human")]]
    ) |>
    group_by(wave, condition, item) |>
    summarise(
      profile_mae = mean(absolute_error),
      exact_match = mean(exact),
      .groups = "drop"
    )
}))

correlation_metrics <- bind_rows(lapply(
  split(llm_valid, interaction(llm_valid$wave, llm_valid$condition)),
  function(data) {
    if (nrow(data) < 20) return(tibble())
    llm_cor <- suppressWarnings(cor(data[items], use = "pairwise.complete.obs"))
    human_data <- benchmark$samples[[as.character(data$wave[[1]])]]
    human_cor <- suppressWarnings(
      cor(human_data[items], use = "pairwise.complete.obs")
    )
    tibble(
      wave = data$wave[[1]],
      condition = data$condition[[1]],
      correlation_rmse = sqrt(mean((llm_cor - human_cor)^2)),
      a425_mean_abs_correlation_llm = mean(abs(llm_cor[1:4, 5])),
      a425_mean_abs_correlation_human = mean(abs(human_cor[1:4, 5]))
    )
  }
))

repeat_stability <- llm_long |>
  group_by(profile_id, wave, condition, item) |>
  summarise(
    distinct_responses = n_distinct(response),
    repeat_sd = sd(response),
    .groups = "drop"
  ) |>
  group_by(wave, condition, item) |>
  summarise(
    share_identical_across_repeats = mean(distinct_responses == 1),
    mean_repeat_sd = mean(repeat_sd, na.rm = TRUE),
    .groups = "drop"
  )

# Follow-up experiment: repeated joint draws and independent item calls.
equality_score <- function(item, response) {
  if_else(item == "a425", response, 6 - response)
}

joint_neutral <- llm_valid |>
  filter(condition == "neutral_verbal", `repeat` <= 5) |>
  select(profile_id, wave, `repeat`, all_of(items)) |>
  distinct(profile_id, wave, `repeat`, .keep_all = TRUE) |>
  pivot_longer(all_of(items), names_to = "item", values_to = "response") |>
  mutate(eq_response = equality_score(item, response))

human_matched <- human_profiles |>
  select(profile_id, wave, all_of(items)) |>
  mutate(wave = as.character(wave)) |>
  pivot_longer(all_of(items), names_to = "item", values_to = "human_response") |>
  mutate(human_eq = equality_score(item, human_response))

empirical_profile_probabilities <- function(data) {
  data |>
    group_by(profile_id, wave, item) |>
    summarise(
      repeats = n(),
      predicted_mean = mean(eq_response),
      p1 = mean(eq_response == 1),
      p2 = mean(eq_response == 2),
      p3 = mean(eq_response == 3),
      p4 = mean(eq_response == 4),
      p5 = mean(eq_response == 5),
      .groups = "drop"
    )
}

profile_probabilities <- empirical_profile_probabilities(joint_neutral) |>
  left_join(
    human_matched |>
      select(profile_id, wave, item, human_eq),
    by = c("profile_id", "wave", "item")
  )

summarise_predictive_metrics <- function(data, analysis_label) {
  data |>
    group_by(wave, item) |>
    group_modify(~ {
      .x <- .x |>
        filter(!is.na(human_eq))
      probability_matrix <- as.matrix(.x[paste0("p", 1:5)])
      predicted_distribution <- colMeans(probability_matrix)
      observed_distribution <- tabulate(
        .x$human_eq, nbins = 5
      ) / nrow(.x)
      predicted_mean <- sum((1:5) * predicted_distribution)
      observed_mean <- mean(.x$human_eq)
      predicted_variance <- sum(
        ((1:5) - predicted_mean)^2 * predicted_distribution
      )
      observed_variance <- mean((.x$human_eq - observed_mean)^2)
      tibble(
        analysis = analysis_label,
        n = nrow(.x),
        repeats = min(.x$repeats),
        ordinal_expected_mae = mean(
          abs(.x$predicted_mean - .x$human_eq)
        ),
        expected_draw_absolute_error = mean(
          rowSums(
            probability_matrix * abs(outer(.x$human_eq, 1:5, "-"))
          )
        ),
        human_mean = observed_mean,
        predicted_mean = predicted_mean,
        mean_error = predicted_mean - observed_mean,
        total_variation = .5 * sum(
          abs(predicted_distribution - observed_distribution)
        ),
        human_variance = observed_variance,
        predicted_variance = predicted_variance,
        variance_ratio = predicted_variance / observed_variance
      )
    }) |>
    ungroup()
}

repeated_predictive_metrics <- summarise_predictive_metrics(
  profile_probabilities, "five_draw_empirical_distribution"
)

first_draw_probabilities <- joint_neutral |>
  filter(`repeat` == 1) |>
  transmute(
    profile_id, wave, item,
    repeats = 1L,
    predicted_mean = eq_response,
    p1 = as.numeric(eq_response == 1),
    p2 = as.numeric(eq_response == 2),
    p3 = as.numeric(eq_response == 3),
    p4 = as.numeric(eq_response == 4),
    p5 = as.numeric(eq_response == 5)
  ) |>
  left_join(
    human_matched |>
      select(profile_id, wave, item, human_eq),
    by = c("profile_id", "wave", "item")
  )
first_draw_metrics <- summarise_predictive_metrics(
  first_draw_probabilities, "first_draw_sensitivity"
)

full_human_variances <- bind_rows(lapply(names(benchmark$samples), function(wave) {
  data <- benchmark$samples[[wave]]
  bind_rows(lapply(items, function(item) {
    response <- if (item == "a425") data[[item]] else 6 - data[[item]]
    keep <- !is.na(response) & !is.na(data$weight) & data$weight > 0
    center <- weighted.mean(response[keep], data$weight[keep])
    tibble(
      wave = as.character(wave),
      item = item,
      human_variance = weighted.mean(
        (response[keep] - center)^2, data$weight[keep]
      )
    )
  }))
}))

variance_decomposition <- joint_neutral |>
  group_by(profile_id, wave, item) |>
  summarise(
    profile_mean = mean(eq_response),
    within_variance = mean((eq_response - profile_mean)^2),
    repeats = n(),
    .groups = "drop"
  ) |>
  group_by(wave, item) |>
  summarise(
    profiles = n(),
    repeats = min(repeats),
    grand_mean = mean(profile_mean),
    between_profile_variance = mean(
      (profile_mean - grand_mean)^2
    ),
    within_profile_variance = mean(within_variance),
    total_predictive_variance =
      between_profile_variance + within_profile_variance,
    within_share = within_profile_variance / total_predictive_variance,
    .groups = "drop"
  ) |>
  left_join(full_human_variances, by = c("wave", "item")) |>
  mutate(
    total_variance_ratio = total_predictive_variance / human_variance,
    between_variance_ratio = between_profile_variance / human_variance
  )

independent_valid <- if (nrow(independent_raw)) {
  independent_raw |>
    filter(success) |>
    transmute(
      profile_id, wave, item = tolower(item),
      response = as.integer(score)
    ) |>
    distinct(profile_id, wave, item, .keep_all = TRUE)
} else {
  tibble(
    profile_id = character(),
    wave = character(),
    item = character(),
    response = integer()
  )
}
independent_item_metrics <- if (nrow(independent_valid)) {
  independent_valid |>
    mutate(eq_response = equality_score(item, response)) |>
    group_by(wave, item) |>
    summarise(
      profiles = n(),
      distinct_responses = n_distinct(eq_response),
      mean = mean(eq_response),
      variance = mean((eq_response - mean(eq_response))^2),
      .groups = "drop"
    ) |>
    left_join(full_human_variances, by = c("wave", "item")) |>
    mutate(variance_ratio = variance / human_variance)
} else {
  tibble(
    wave = character(),
    item = character(),
    profiles = integer(),
    distinct_responses = integer(),
    mean = numeric(),
    variance = numeric(),
    human_variance = numeric(),
    variance_ratio = numeric()
  )
}

correlation_diagnostics <- function(data) {
  matrix <- as.matrix(data[items])
  correlation <- suppressWarnings(cor(matrix))
  a425_correlations <- abs(correlation[1:4, 5])
  tibble(
    correlation_rmse = NA_real_,
    a425_mean_abs_correlation = if (any(is.finite(a425_correlations))) {
      mean(a425_correlations, na.rm = TRUE)
    } else {
      NA_real_
    },
    estimable_a425_pairs = sum(is.finite(a425_correlations))
  )
}

joint_wide <- joint_neutral |>
  select(profile_id, wave, `repeat`, item, response) |>
  pivot_wider(names_from = item, values_from = response)
joint_correlations_by_repeat <- joint_wide |>
  group_by(wave, `repeat`) |>
  group_modify(~ correlation_diagnostics(.x)) |>
  ungroup()

independent_wide <- if (nrow(independent_valid)) {
  independent_valid |>
    pivot_wider(names_from = item, values_from = response)
} else {
  tibble(profile_id = character(), wave = character())
}
independent_correlations <- if (nrow(independent_wide)) {
  independent_wide |>
    group_by(wave) |>
    group_modify(~ correlation_diagnostics(.x)) |>
    ungroup()
} else {
  tibble(
    wave = character(),
    correlation_rmse = numeric(),
    a425_mean_abs_correlation = numeric(),
    estimable_a425_pairs = integer()
  )
}

human_a425_correlations <- correlation_metrics |>
  filter(condition == "neutral_verbal") |>
  distinct(
    wave,
    human_a425 = a425_mean_abs_correlation_human
  )
coherence_comparison <- joint_correlations_by_repeat |>
  group_by(wave) |>
  summarise(
    joint_a425 = mean(a425_mean_abs_correlation),
    joint_repeat_sd = sd(a425_mean_abs_correlation),
    joint_estimable_pairs = min(estimable_a425_pairs),
    repeats = n(),
    .groups = "drop"
  ) |>
  left_join(
    independent_correlations |>
      select(
        wave,
        independent_a425 = a425_mean_abs_correlation,
        independent_estimable_pairs = estimable_a425_pairs
      ),
    by = "wave"
  ) |>
  left_join(human_a425_correlations, by = "wave") |>
  mutate(
    context_difference = joint_a425 - independent_a425,
    independent_human_gap = independent_a425 - human_a425
  )

followup_ready <- nrow(joint_neutral) == 300 * 5 * 5 &&
  nrow(independent_valid) == 300 * 5 &&
  all(profile_probabilities$repeats == 5)

core_bootstrap <- tibble()
coherence_bootstrap_by_wave <- tibble()
if (followup_ready) {
  prepare_wave <- function(wave) {
    profile_order <- human_profiles |>
      filter(as.character(wave) == !!wave) |>
      pull(profile_id)
    joint_wave <- joint_wide |>
      filter(.data$wave == !!wave)
    independent_wave <- independent_wide |>
      filter(.data$wave == !!wave)
    human_wave <- human_matched |>
      filter(.data$wave == !!wave) |>
      select(profile_id, item, human_eq) |>
      pivot_wider(names_from = item, values_from = human_eq)
    available <- Reduce(
      intersect,
      list(
        profile_order,
        unique(joint_wave$profile_id),
        independent_wave$profile_id,
        human_wave$profile_id
      )
    )
    profile_order <- profile_order[profile_order %in% available]
    joint_matrices <- lapply(1:5, function(repeat_index) {
      data <- joint_wave |>
        filter(`repeat` == repeat_index)
      as.matrix(
        data[match(profile_order, data$profile_id), items]
      )
    })
    independent_matrix <- as.matrix(
      independent_wave[
        match(profile_order, independent_wave$profile_id), items
      ]
    )
    human_matrix <- as.matrix(
      human_wave[match(profile_order, human_wave$profile_id), items]
    )
    full_data <- benchmark$samples[[wave]]
    full_human_correlation <- suppressWarnings(
      cor(full_data[items], use = "pairwise.complete.obs")
    )
    list(
      profiles = profile_order,
      joint = joint_matrices,
      independent = independent_matrix,
      human = human_matrix,
      full_human_correlation = full_human_correlation
    )
  }

  wave_data <- setNames(
    lapply(c("2012", "2018", "2021"), prepare_wave),
    c("2012", "2018", "2021")
  )
  stopifnot(all(vapply(wave_data, function(x) {
    length(x$profiles) == 100 &&
      all(vapply(x$joint, nrow, integer(1)) == 100)
  }, logical(1))))

  calculate_wave_metrics <- function(data, index) {
    joint_eq <- lapply(data$joint, function(matrix) {
      transformed <- matrix
      transformed[, 1:4] <- 6 - transformed[, 1:4]
      transformed[index, , drop = FALSE]
    })
    human_eq <- data$human[index, , drop = FALSE]
    item_metrics <- bind_rows(lapply(seq_along(items), function(j) {
      observed <- human_eq[, j]
      complete <- !is.na(observed)
      draws <- unlist(lapply(
        joint_eq,
        function(matrix) matrix[complete, j]
      ))
      profile_means <- rowMeans(do.call(
        cbind, lapply(joint_eq, function(matrix) matrix[, j])
      ))[complete]
      observed <- observed[complete]
      predicted_distribution <- tabulate(draws, nbins = 5) / length(draws)
      observed_distribution <- tabulate(observed, nbins = 5) / length(observed)
      predicted_mean <- mean(draws)
      observed_mean <- mean(observed)
      draw_matrix <- do.call(
        cbind,
        lapply(joint_eq, function(matrix) matrix[complete, j])
      )
      tibble(
        omae = mean(abs(profile_means - observed)),
        expected_draw_absolute_error = mean(abs(draw_matrix - observed)),
        absolute_mean_error = abs(predicted_mean - observed_mean),
        total_variation = .5 * sum(
          abs(predicted_distribution - observed_distribution)
        ),
        variance_ratio =
          mean((draws - predicted_mean)^2) /
          mean((observed - observed_mean)^2)
      )
    }))
    joint_correlations <- lapply(data$joint, function(matrix) {
      suppressWarnings(cor(matrix[index, , drop = FALSE]))
    })
    joint_correlation_rmse <- mean(vapply(
      joint_correlations,
      function(correlation) {
        sqrt(mean(
          (correlation - data$full_human_correlation)^2
        ))
      },
      numeric(1)
    ))
    joint_a425 <- mean(vapply(
      joint_correlations,
      function(correlation) {
        values <- abs(correlation[1:4, 5])
        if (any(is.finite(values))) mean(values, na.rm = TRUE) else NA_real_
      },
      numeric(1)
    ))
    independent_correlation <- suppressWarnings(
      cor(data$independent[index, , drop = FALSE])
    )
    independent_values <- abs(independent_correlation[1:4, 5])
    independent_a425 <- if (any(is.finite(independent_values))) {
      mean(independent_values, na.rm = TRUE)
    } else {
      NA_real_
    }
    tibble(
      omae = mean(item_metrics$omae),
      absolute_mean_error = mean(item_metrics$absolute_mean_error),
      total_variation = mean(item_metrics$total_variation),
      variance_ratio = mean(item_metrics$variance_ratio),
      correlation_rmse = joint_correlation_rmse,
      joint_a425 = joint_a425,
      independent_a425 = independent_a425,
      context_difference = joint_a425 - independent_a425
    )
  }

  point_by_wave <- bind_rows(lapply(names(wave_data), function(wave) {
    calculate_wave_metrics(
      wave_data[[wave]],
      seq_along(wave_data[[wave]]$profiles)
    ) |>
      mutate(wave = wave, .before = 1)
  }))
  point_overall <- point_by_wave |>
    summarise(across(where(is.numeric), ~ mean(.x, na.rm = TRUE)))

  set.seed(20260702)
  bootstrap_repetitions <- 2000
  bootstrap_rows <- vector("list", bootstrap_repetitions)
  bootstrap_wave_rows <- vector("list", bootstrap_repetitions)
  for (bootstrap_index in seq_len(bootstrap_repetitions)) {
    draw <- bind_rows(lapply(names(wave_data), function(wave) {
      n <- length(wave_data[[wave]]$profiles)
      calculate_wave_metrics(
        wave_data[[wave]],
        sample.int(n, n, replace = TRUE)
      ) |>
        mutate(wave = wave, .before = 1)
    }))
    bootstrap_wave_rows[[bootstrap_index]] <- draw |>
      select(wave, joint_a425, independent_a425, context_difference) |>
      mutate(replication = bootstrap_index)
    bootstrap_rows[[bootstrap_index]] <- draw |>
      summarise(across(
        where(is.numeric), ~ mean(.x, na.rm = TRUE)
      )) |>
      mutate(replication = bootstrap_index)
  }
  bootstrap_frame <- bind_rows(bootstrap_rows)
  core_bootstrap <- bind_rows(lapply(
    setdiff(names(point_overall), "replication"),
    function(metric) {
      tibble(
        metric = metric,
        estimate = point_overall[[metric]],
        ci_low = quantile(
          bootstrap_frame[[metric]], .025, names = FALSE, na.rm = TRUE
        ),
        ci_high = quantile(
          bootstrap_frame[[metric]], .975, names = FALSE, na.rm = TRUE
        ),
        bootstrap_repetitions = bootstrap_repetitions,
        profiles = 300
      )
    }
  ))
  bootstrap_wave_frame <- bind_rows(bootstrap_wave_rows)
  coherence_bootstrap_by_wave <- point_by_wave |>
    select(wave, joint_a425, independent_a425, context_difference) |>
    pivot_longer(
      -wave, names_to = "metric", values_to = "estimate"
    ) |>
    left_join(
      bootstrap_wave_frame |>
        pivot_longer(
          c(joint_a425, independent_a425, context_difference),
          names_to = "metric", values_to = "value"
        ) |>
        group_by(wave, metric) |>
        summarise(
          ci_low = quantile(value, .025, na.rm = TRUE),
          ci_high = quantile(value, .975, na.rm = TRUE),
          .groups = "drop"
        ),
      by = c("wave", "metric")
    ) |>
    mutate(
      bootstrap_repetitions = bootstrap_repetitions,
      profiles = 100
    )
}

prompt_sensitivity <- if (all(c("original", "paraphrased") %in% unique(llm_long$condition))) {
  llm_long |>
    group_by(profile_id, wave, condition, item) |>
    summarise(response = mean(response), .groups = "drop") |>
    pivot_wider(names_from = condition, values_from = response) |>
    filter(!is.na(original), !is.na(paraphrased)) |>
    group_by(wave, item) |>
    summarise(
      mean_absolute_prompt_shift = mean(abs(original - paraphrased)),
      prompt_agreement = mean(original == paraphrased),
      .groups = "drop"
    )
} else {
  tibble(
    wave = character(),
    item = character(),
    mean_absolute_prompt_shift = numeric(),
    prompt_agreement = numeric()
  )
}

add_equality_scores <- function(data) {
  data |>
    mutate(
      eq_a421 = 6 - a421,
      eq_a422 = 6 - a422,
      eq_a423 = 6 - a423,
      eq_a424 = 6 - a424,
      eq_a425 = a425,
      eq_public = rowMeans(
        pick(eq_a421, eq_a422, eq_a423, eq_a424),
        na.rm = TRUE
      ),
      eq_household = eq_a425
    )
}

fit_gradients <- function(data, source_label, condition_label, weighted) {
  outcomes <- c(
    public = "eq_public",
    household = "eq_household",
    a421 = "eq_a421", a422 = "eq_a422", a423 = "eq_a423",
    a424 = "eq_a424", a425 = "eq_a425"
  )
  predictors <- c(
    "female", "educ_years", "age_decade", "urban_residence"
  )
  bind_rows(lapply(split(data, data$wave), function(wave_data) {
    wave_data <- wave_data |>
      mutate(age_decade = (age - mean(age, na.rm = TRUE)) / 10)
    bind_rows(lapply(names(outcomes), function(outcome_name) {
      formula <- as.formula(paste(
        outcomes[[outcome_name]], "~", paste(predictors, collapse = " + ")
      ))
      model <- if (weighted) {
        lm(formula, data = wave_data, weights = analysis_weight)
      } else {
        lm(formula, data = wave_data)
      }
      coefficients <- summary(model)$coefficients
      bind_rows(lapply(predictors, function(predictor) {
        tibble(
          source = source_label,
          condition = condition_label,
          wave = as.character(wave_data$wave[[1]]),
          outcome = outcome_name,
          predictor = predictor,
          estimate = coefficients[predictor, "Estimate"],
          std_error = coefficients[predictor, "Std. Error"],
          n = nobs(model)
        )
      }))
    }))
  }))
}

human_gradient_data <- full_human |>
  add_equality_scores() |>
  mutate(analysis_weight = weight)
human_gradients <- fit_gradients(
  human_gradient_data, "CGSS", "benchmark", weighted = TRUE
)

llm_gradient_data <- llm_valid |>
  add_equality_scores() |>
  mutate(analysis_weight = 1)
llm_gradients <- bind_rows(lapply(
  split(llm_gradient_data, llm_gradient_data$condition),
  function(condition_data) {
    if (nrow(condition_data) < 60) return(tibble())
    fit_gradients(
      condition_data,
      "LLM",
      as.character(condition_data$condition[[1]]),
      weighted = FALSE
    )
  }
))
gradient_comparison <- llm_gradients |>
  select(
    condition, wave, outcome, predictor,
    llm_estimate = estimate, llm_std_error = std_error, llm_n = n
  ) |>
  left_join(
    human_gradients |>
      select(
        wave, outcome, predictor,
        human_estimate = estimate,
        human_std_error = std_error,
        human_n = n
      ),
    by = c("wave", "outcome", "predictor")
  ) |>
  mutate(coefficient_error = llm_estimate - human_estimate)

distribution_plot <- distribution_metrics |>
  filter(condition %in% complete_conditions) |>
  mutate(
    item = toupper(item),
    wave = factor(wave, levels = c("2012", "2018", "2021"))
  ) |>
  ggplot(aes(human_mean, llm_mean, color = condition)) +
  geom_abline(slope = 1, intercept = 0, color = "grey55", linewidth = .5) +
  geom_point(size = 2.2, alpha = .85) +
  facet_grid(wave ~ item) +
  coord_equal(xlim = c(1, 5), ylim = c(1, 5)) +
  labs(
    title = "LLM与CGSS题项均值比较",
    x = "CGSS加权均值", y = "LLM均值", color = "提示条件"
  ) +
  theme_minimal(base_size = 11) +
  theme(legend.position = "bottom")

ggsave(
  file.path(figure_dir, "item_mean_comparison.png"),
  distribution_plot, width = 10, height = 5.6, dpi = 220, bg = "white"
)

write_csv(call_quality, file.path(output_dir, "metrics_call_quality.csv"))
write_csv(condition_counts, file.path(output_dir, "metrics_condition_counts.csv"))
write_csv(
  distribution_metrics,
  file.path(output_dir, "metrics_distribution.csv")
)
write_csv(variance_metrics, file.path(output_dir, "metrics_variance.csv"))
write_csv(paired_metrics, file.path(output_dir, "metrics_profile_error.csv"))
write_csv(
  correlation_metrics,
  file.path(output_dir, "metrics_correlations.csv")
)
write_csv(
  repeat_stability,
  file.path(output_dir, "metrics_repeat_stability.csv")
)
write_csv(
  repeated_predictive_metrics,
  file.path(output_dir, "metrics_repeated_predictive.csv")
)
write_csv(
  first_draw_metrics,
  file.path(output_dir, "metrics_first_draw_sensitivity.csv")
)
write_csv(
  variance_decomposition,
  file.path(output_dir, "metrics_variance_decomposition.csv")
)
write_csv(
  joint_correlations_by_repeat,
  file.path(output_dir, "metrics_joint_correlations_by_repeat.csv")
)
write_csv(
  coherence_comparison,
  file.path(output_dir, "metrics_coherence_comparison.csv")
)
write_csv(
  independent_item_metrics,
  file.path(output_dir, "metrics_independent_items.csv")
)
write_csv(
  core_bootstrap,
  file.path(output_dir, "metrics_core_bootstrap.csv")
)
write_csv(
  coherence_bootstrap_by_wave,
  file.path(output_dir, "metrics_coherence_bootstrap_by_wave.csv")
)
write_csv(
  prompt_sensitivity,
  file.path(output_dir, "metrics_prompt_sensitivity.csv")
)
write_csv(
  gradient_comparison,
  file.path(output_dir, "metrics_subgroup_gradients.csv")
)

cat("\nCall quality:\n")
print(call_quality)
cat("\nDistribution metrics:\n")
print(distribution_metrics, n = Inf)
cat("\nCorrelation metrics:\n")
print(correlation_metrics, n = Inf)
cat("\nRepeated-draw predictive metrics:\n")
print(repeated_predictive_metrics, n = Inf)
cat("\nVariance decomposition:\n")
print(variance_decomposition, n = Inf)
cat("\nJoint versus independent coherence:\n")
print(coherence_comparison, n = Inf)
if (followup_ready) {
  cat("\nProfile-cluster bootstrap:\n")
  print(core_bootstrap, n = Inf)
}
cat("\nKey subgroup gradients:\n")
print(
  gradient_comparison |>
    filter(outcome %in% c("public", "household")) |>
    select(
      condition, wave, outcome, predictor,
      human_estimate, llm_estimate, coefficient_error
    ),
  n = Inf
)
cat("\nOutputs written to ", output_dir, "\n", sep = "")
