# requires: R 4.x, lavaan (>= 0.6), jsonlite
# Exploratory downstream statistics (PREREGISTRATION_V2.md, 2026-11-15 entry):
# the analyses common in survey research (Cronbach's alpha, one-factor CFA,
# OLS of A425 on the A421-A424 index) on CGSS and on the Qwen synthetic
# datasets written by scripts/25_export_drawn_answers.py. Profile-level inputs
# are read from outside the repository; only aggregate tables are written.
#
# Usage: Rscript scripts/26_downstream_statistics.R BENCHMARK_RDS SYNTHETIC_CSV OUTPUT_DIR

suppressPackageStartupMessages({
  library(lavaan)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 3) {
  stop("Usage: 26_downstream_statistics.R BENCHMARK_RDS SYNTHETIC_CSV OUTPUT_DIR")
}
output_dir <- args[[3]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
items <- paste0("a42", 1:5)
reps <- 1000
seed <- 20261115

# Higher values are more egalitarian on every item (as in scripts/12).
egal <- function(d) {
  for (i in items[1:4]) d[[i]] <- 6 - as.numeric(d[[i]])
  d$a425 <- as.numeric(d$a425)
  d
}
complete5 <- function(d) Reduce(`&`, lapply(items, function(i) d[[i]] %in% 1:5))

statistics <- function(d, w = NULL) {
  if (is.null(w)) w <- rep(1, nrow(d))
  w <- w / sum(w)
  X <- as.matrix(d[items])
  S <- cov.wt(X, wt = w)$cov
  alpha <- function(S) { k <- ncol(S); k / (k - 1) * (1 - sum(diag(S)) / sum(S)) }
  out <- c(
    alpha5 = alpha(S),
    alpha4 = alpha(S[1:4, 1:4]),
    a425_item_total = cov.wt(cbind(X[, 5], rowSums(X[, 1:4])), wt = w, cor = TRUE)$cor[1, 2],
    a425_mean_abs_r = mean(abs(cov2cor(S)[5, 1:4]))
  )
  d$w <- w * nrow(d)
  d$index <- rowMeans(X[, 1:4])
  d$age_decade <- d$age / 10
  m1 <- lm(a425 ~ index, data = d, weights = w)
  m2 <- lm(a425 ~ index + female + educ_years + age_decade + urban_residence + factor(wave),
           data = d, weights = w)
  out <- c(out,
           b_index = unname(coef(m1)["index"]), r2_index = summary(m1)$r.squared,
           b_index_ctrl = unname(coef(m2)["index"]), b_female_ctrl = unname(coef(m2)["female"]),
           b_urban_ctrl = unname(coef(m2)["urban_residence"]))
  fit <- tryCatch({
    f <- cfa("g =~ a421 + a422 + a423 + a424 + a425", data = d, std.lv = TRUE, estimator = "ML",
             sampling.weights = if (length(unique(d$w)) > 1) "w" else NULL, warn = FALSE)
    fm <- fitMeasures(f, c("cfi", "rmsea", "srmr"))
    ss <- standardizedSolution(f)
    loading <- ss$op == "=~"
    c(cfi = unname(fm["cfi"]), rmsea = unname(fm["rmsea"]), srmr = unname(fm["srmr"]),
      a425_loading = ss$est.std[loading & ss$rhs == "a425"],
      min_other_loading = min(ss$est.std[loading & ss$rhs != "a425"]),
      cfa_converged = as.numeric(lavInspect(f, "converged")))
  }, error = function(e) c(cfi = NA, rmsea = NA, srmr = NA, a425_loading = NA,
                           min_other_loading = NA, cfa_converged = 0))
  c(out, fit)
}
row <- function(population, cell, path, n, s) {
  data.frame(population = population, cell = cell, path = path, n = n, t(s), check.names = FALSE)
}

# ---- CGSS ----
benchmark <- readRDS(args[[1]])
human <- do.call(rbind, lapply(names(benchmark$samples), function(wave) {
  d <- benchmark$samples[[wave]]
  keep <- complete5(d) & !is.na(d$weight) & d$weight > 0 & !is.na(d$female) &
    !is.na(d$educ_years) & !is.na(d$age) & !is.na(d$urban_residence)
  d <- d[keep, c(items, "weight", "female", "educ_years", "age", "urban_residence")]
  d$wave <- wave
  d
}))
# Waves count equally (as in scripts/12).
human$w <- ave(human$weight, human$wave, FUN = function(x) x / sum(x))
human <- egal(human)
rows <- list(row("H-full", "", NA, nrow(human), statistics(human, human$w)))

# Sampling range at the synthetic sample size: 100 per wave, drawn with
# probability proportional to the survey weight.
set.seed(seed)
by_wave <- split(seq_len(nrow(human)), human$wave)
h300 <- t(replicate(reps, {
  idx <- unlist(lapply(by_wave, function(i) sample(i, 100, replace = TRUE, prob = human$weight[i])))
  statistics(human[idx, ])
}))
for (q in c(0.025, 0.5, 0.975)) {
  label <- c("0.025" = "H-300 p2.5", "0.5" = "H-300 median", "0.975" = "H-300 p97.5")[[as.character(q)]]
  rows[[length(rows) + 1]] <- row(label, "", NA, 300, apply(h300, 2, quantile, q, na.rm = TRUE))
}

# ---- matched respondents and synthetic datasets ----
s <- read.csv(args[[2]], stringsAsFactors = FALSE)
s$wave <- as.character(s$wave)
s <- egal(s[complete5(s), ])
matched <- s[s$source == "H-matched", ]
rows[[length(rows) + 1]] <- row("H-matched", "", NA, nrow(matched), statistics(matched))
cells <- sort(unique(s$cell[s$source == "S-drawn"]))
paths <- sort(unique(s$path[s$source == "S-drawn"]))
for (cell in cells) {
  for (k in paths) {
    d <- s[s$source == "S-drawn" & s$cell == cell & s$path == k, ]
    rows[[length(rows) + 1]] <- row("S-drawn", cell, k, nrow(d), statistics(d))
  }
  d <- s[s$source == "S-modal" & s$cell == cell, ]
  rows[[length(rows) + 1]] <- row("S-modal", cell, NA, nrow(d), statistics(d))
}
by_dataset <- do.call(rbind, rows)
write.csv(by_dataset, file.path(output_dir, "exploratory_downstream_by_dataset.csv"), row.names = FALSE)

# ---- summary: drawn datasets averaged over paths, against the sampling range ----
measures <- setdiff(names(by_dataset), c("population", "cell", "path", "n", "cfa_converged"))
low <- unlist(by_dataset[by_dataset$population == "H-300 p2.5", measures])
high <- unlist(by_dataset[by_dataset$population == "H-300 p97.5", measures])
summary_rows <- list()
for (cell in cells) {
  d <- by_dataset[by_dataset$population == "S-drawn" & by_dataset$cell == cell, measures]
  for (m in measures) {
    v <- d[[m]]
    mean_v <- mean(v, na.rm = TRUE)
    summary_rows[[length(summary_rows) + 1]] <- data.frame(
      cell = cell, measure = m, mean_over_paths = mean_v,
      min_over_paths = min(v, na.rm = TRUE), max_over_paths = max(v, na.rm = TRUE),
      cgss_full = by_dataset[by_dataset$population == "H-full", m],
      cgss_300_p2.5 = low[[m]], cgss_300_p97.5 = high[[m]],
      outside_cgss_300_range = mean_v < low[[m]] | mean_v > high[[m]])
  }
}
write.csv(do.call(rbind, summary_rows), file.path(output_dir, "exploratory_downstream_summary.csv"),
          row.names = FALSE)

# Decisions a researcher might take, by dataset.
decision <- function(d) data.frame(
  population = d$population, cell = d$cell, path = d$path,
  drop_a425_raises_alpha = d$alpha4 > d$alpha5,
  keep_a425_item_total_0.30 = d$a425_item_total >= 0.30,
  keep_a425_loading_0.40 = d$a425_loading >= 0.40,
  alpha5_at_least_0.70 = d$alpha5 >= 0.70)
h_decisions <- colMeans(cbind(drop_a425_raises_alpha = h300[, "alpha4"] > h300[, "alpha5"],
                              keep_a425_item_total_0.30 = h300[, "a425_item_total"] >= 0.30,
                              keep_a425_loading_0.40 = h300[, "a425_loading"] >= 0.40,
                              alpha5_at_least_0.70 = h300[, "alpha5"] >= 0.70), na.rm = TRUE)
decisions <- decision(by_dataset[by_dataset$population %in% c("H-full", "H-matched", "S-drawn", "S-modal"), ])
write.csv(decisions, file.path(output_dir, "exploratory_downstream_decisions.csv"), row.names = FALSE)

meta <- list(
  plan = "PREREGISTRATION_V2.md, 2026-11-15 entry (exploratory; recorded after the run)",
  benchmark_rds_md5 = unname(tools::md5sum(args[[1]])),
  synthetic_csv_md5 = unname(tools::md5sum(args[[2]])),
  human_n = nrow(human), matched_n = nrow(matched),
  synthetic_n_per_dataset = 300, drawn_paths = length(paths), cells = cells,
  sampling_range = list(replicates = reps, seed = seed, per_wave = 100,
                        draw = "with replacement, probability proportional to the survey weight",
                        share_of_samples = as.list(h_decisions)),
  coding = "higher = more egalitarian; A421-A424 reversed as 6 - x",
  cfa = "one factor, ML, items treated as continuous, std.lv; survey weights for H-full only",
  regression = "OLS; controls female, educ_years, age/10, urban_residence, wave dummies",
  cfa_failures = by_dataset[by_dataset$cfa_converged == 0, c("population", "cell", "path")],
  lavaan = as.character(packageVersion("lavaan")), r = R.version.string)
write_json(meta, file.path(output_dir, "exploratory_downstream_metadata.json"), auto_unbox = TRUE,
           pretty = TRUE, digits = NA)
cat("H-full n =", nrow(human), "| H-matched n =", nrow(matched), "| datasets =", nrow(by_dataset), "\n")
