# requires: base R
# Weighted CGSS aggregates used as the v2 human benchmark.
# Definitions follow scripts/03_evaluate_audit.R: per-item complete cases with a
# positive weight, weighted population variance, pairwise weighted correlations.
# Writes aggregates only.
#
# Usage: Rscript scripts/08_export_human_benchmark.R BENCHMARK_RDS OUTPUT_DIR

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 2) stop("Usage: 08_export_human_benchmark.R BENCHMARK_RDS OUTPUT_DIR")
benchmark <- readRDS(args[[1]])
output_dir <- args[[2]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
items <- paste0("a42", 1:5)

marginals <- list()
moments <- list()
correlations <- list()
for (wave in names(benchmark$samples)) {
  data <- benchmark$samples[[wave]]
  for (item in items) {
    x <- data[[item]]
    w <- data$weight
    keep <- !is.na(x) & x %in% 1:5 & !is.na(w) & w > 0
    x <- x[keep]
    w <- w[keep] / sum(w[keep])
    center <- sum(w * x)
    moments[[length(moments) + 1]] <- data.frame(
      wave = wave, item = item, n = sum(keep),
      mean = center, variance = sum(w * (x - center)^2)
    )
    marginals[[length(marginals) + 1]] <- data.frame(
      wave = wave, item = item, option = 1:5,
      probability = vapply(1:5, function(k) sum(w[x == k]), numeric(1))
    )
  }
  for (i in seq_along(items)) {
    for (j in seq_along(items)) {
      x <- data[[items[[i]]]]
      y <- data[[items[[j]]]]
      w <- data$weight
      keep <- x %in% 1:5 & y %in% 1:5 & !is.na(w) & w > 0
      x <- x[keep]
      y <- y[keep]
      w <- w[keep] / sum(w[keep])
      xc <- x - sum(w * x)
      yc <- y - sum(w * y)
      correlations[[length(correlations) + 1]] <- data.frame(
        wave = wave, item_x = items[[i]], item_y = items[[j]], n = sum(keep),
        correlation = sum(w * xc * yc) / sqrt(sum(w * xc^2) * sum(w * yc^2))
      )
    }
  }
}

write.csv(do.call(rbind, marginals), file.path(output_dir, "human_marginals.csv"), row.names = FALSE)
write.csv(do.call(rbind, moments), file.path(output_dir, "human_moments.csv"), row.names = FALSE)
write.csv(do.call(rbind, correlations), file.path(output_dir, "human_correlations.csv"), row.names = FALSE)
