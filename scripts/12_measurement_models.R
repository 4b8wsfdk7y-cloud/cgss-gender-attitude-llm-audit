# requires: R 4.x, lavaan (>= 0.6), semTools, mirt, jsonlite
# Part 3 of PREREGISTRATION_V2.md: measurement models for CGSS and three
# synthetic populations (S1: modal answers in A1-B1-C1; S2: modal answers in
# A0-B0-C0; S3: v1 sampled joint responses, five replicates of 300).
# Profile-level inputs are read from outside the repository; only aggregate
# tables are written. replicate = 0 marks populations without replicates.
#
# Usage: Rscript scripts/12_measurement_models.R BENCHMARK_RDS V2_LEDGER V1_RESPONSES OUTPUT_DIR

suppressPackageStartupMessages({
  library(lavaan)
  library(semTools)
  library(mirt)
  library(jsonlite)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 4) {
  stop("Usage: 12_measurement_models.R BENCHMARK_RDS V2_LEDGER V1_RESPONSES OUTPUT_DIR")
}
output_dir <- args[[4]]
dir.create(output_dir, recursive = TRUE, showWarnings = FALSE)
items <- paste0("a42", 1:5)
threshold <- 10

# Higher values are more egalitarian on every item.
recode <- function(d) {
  for (item in items[1:4]) d[[item]] <- 6L - as.integer(d[[item]])
  d$a425 <- as.integer(d$a425)
  d
}

# ---- populations ----
benchmark <- readRDS(args[[1]])
human <- do.call(rbind, lapply(names(benchmark$samples), function(wave) {
  d <- benchmark$samples[[wave]]
  keep <- Reduce(`&`, lapply(items, function(i) d[[i]] %in% 1:5)) & !is.na(d$weight) & d$weight > 0
  d <- d[keep, c(items, "weight")]
  d$wave <- wave
  d
}))
n_waves <- length(unique(human$wave))
human$w <- ave(human$weight, human$wave, FUN = function(w) w / sum(w)) * nrow(human) / n_waves
human <- recode(human)

ledger <- stream_in(file(args[[2]]), verbose = FALSE, simplifyVector = FALSE)
modal_population <- function(cell) {
  rows <- Filter(function(r) r$cell == cell, ledger)
  long <- data.frame(
    profile_id = vapply(rows, `[[`, "", "profile_id"),
    item = tolower(vapply(rows, `[[`, "", "item")),
    answer = vapply(rows, function(r) which.max(unlist(r$probabilities[as.character(1:5)])), 1L)
  )
  wide <- reshape(long, idvar = "profile_id", timevar = "item", direction = "wide")
  names(wide) <- sub("^answer\\.", "", names(wide))
  stopifnot(nrow(wide) == 300, all(items %in% names(wide)), !anyNA(wide[items]))
  recode(wide[c("profile_id", items)])
}
s1 <- modal_population("A1-B1-C1")
s2 <- modal_population("A0-B0-C0")

v1 <- stream_in(file(args[[3]]), verbose = FALSE, simplifyVector = FALSE)
s3 <- lapply(1:5, function(k) {
  rows <- Filter(function(r) r[["repeat"]] == k, v1)
  d <- data.frame(profile_id = vapply(rows, `[[`, "", "profile_id"))
  for (item in items) d[[item]] <- vapply(rows, function(r) as.integer(r$scores[[toupper(item)]]), 1L)
  stopifnot(nrow(d) == 300, !anyDuplicated(d$profile_id))
  recode(d)
})
rm(ledger, v1)

# ---- sparse categories (same coding for every population in an analysis) ----
merge_categories <- function(pops, analysis) {
  log <- list()
  for (item in items) {
    groups <- as.list(1:5)  # each element: original values in one category
    repeat {
      code <- function(x) vapply(x, function(v) which(vapply(groups, function(g) v %in% g, TRUE)), 1L)
      counts <- sapply(pops, function(d) tabulate(code(d[[item]]), nbins = length(groups)))
      counts <- matrix(counts, nrow = length(groups))
      low <- apply(counts, 1, min)
      if (length(groups) == 1 || all(low >= threshold)) break
      k <- which.min(low)
      centre <- vapply(groups, mean, 0)
      if (centre[[k]] < 3) j <- k + 1
      else if (centre[[k]] > 3) j <- k - 1
      else {
        neighbours <- c(k - 1, k + 1)
        neighbours <- neighbours[neighbours >= 1 & neighbours <= length(groups)]
        j <- neighbours[which.max(rowSums(counts)[neighbours])]
      }
      log[[length(log) + 1]] <- data.frame(
        analysis = analysis, item = item,
        merged = paste(groups[[k]], collapse = "+"), into = paste(groups[[j]], collapse = "+"),
        min_count = low[[k]]
      )
      groups[[j]] <- sort(c(groups[[j]], groups[[k]]))
      groups[[k]] <- NULL
    }
    for (p in seq_along(pops)) {
      pops[[p]][[item]] <- vapply(pops[[p]][[item]], function(v) which(vapply(groups, function(g) v %in% g, TRUE)), 1L)
    }
    if (length(groups) == 1) {
      log[[length(log) + 1]] <- data.frame(analysis = analysis, item = item,
                                           merged = "all", into = "dropped", min_count = NA)
      for (p in seq_along(pops)) pops[[p]][[item]] <- NULL
    }
  }
  list(pops = pops, log = do.call(rbind, log))
}

present <- function(d, set) intersect(set, names(d))
syntax <- function(set) paste("F =~", paste(set, collapse = " + "))

# A model that fails or does not converge is recorded as not estimable.
fit_one <- function(d, set, weighted) {
  fit <- tryCatch(
    suppressWarnings(cfa(syntax(set), data = d, ordered = set, estimator = "WLSMV",
                         parameterization = "delta", std.lv = TRUE,
                         sampling.weights = if (weighted) "w" else NULL)),
    error = function(e) NULL)
  if (!is.null(fit) && !lavInspect(fit, "converged")) fit <- NULL
  fit
}
measures <- c("chisq.scaled", "df.scaled", "cfi.scaled", "tli.scaled", "rmsea.scaled", "srmr")

fit_row <- function(fit, population, model, replicate, n) {
  m <- if (is.null(fit)) setNames(as.list(rep(NA_real_, length(measures))), measures)
       else as.list(fitMeasures(fit, measures))
  proper <- if (is.null(fit)) NA else {
    std <- standardizedSolution(fit)
    lavInspect(fit, "post.check") && all(abs(std$est.std[std$op == "=~"]) <= 1)
  }
  data.frame(population = population, replicate = replicate, model = model, n = n,
             estimable = !is.null(fit), proper = proper, m)
}

loading_rows <- function(fit, population, replicate) {
  if (is.null(fit)) return(NULL)
  s <- standardizedSolution(fit)
  s <- s[s$op == "=~", ]
  data.frame(population = population, replicate = replicate, item = s$rhs,
             loading = s$est.std, se = s$se)
}

# ---- single-population models ----
categories <- list(); fits <- list(); loadings <- list(); grm <- list(); polychoric <- list()
single <- list(list(name = "human", data = human, weighted = TRUE, replicate = 0),
               list(name = "S1", data = s1, weighted = FALSE, replicate = 0),
               list(name = "S2", data = s2, weighted = FALSE, replicate = 0))
for (k in 1:5) single[[length(single) + 1]] <- list(name = "S3", data = s3[[k]], weighted = FALSE, replicate = k)

for (pop in single) {
  label <- if (pop$replicate == 0) pop$name else paste0(pop$name, "_r", pop$replicate)
  merged <- merge_categories(list(pop$data), paste("single", label))
  d <- merged$pops[[1]]
  categories[[length(categories) + 1]] <- merged$log
  set5 <- present(d, items)
  set4 <- present(d, items[1:4])
  f1 <- fit_one(d, set5, pop$weighted)
  f2 <- fit_one(d, set4, pop$weighted)
  fits[[length(fits) + 1]] <- fit_row(f1, pop$name, "M1", pop$replicate, nrow(d))
  fits[[length(fits) + 1]] <- fit_row(f2, pop$name, "M2", pop$replicate, nrow(d))
  loadings[[length(loadings) + 1]] <- loading_rows(f1, pop$name, pop$replicate)

  # Descriptive (added after estimation, see the pre-registration): polychoric correlations.
  pc <- tryCatch(suppressWarnings(lavCor(d[set5], ordered = set5,
                                         sampling.weights = if (pop$weighted) "w" else NULL)),
                 error = function(e) NULL)
  if (!is.null(pc)) {
    pairs <- which(upper.tri(pc), arr.ind = TRUE)
    polychoric[[length(polychoric) + 1]] <- data.frame(
      population = pop$name, replicate = pop$replicate,
      item_x = rownames(pc)[pairs[, 1]], item_y = colnames(pc)[pairs[, 2]], correlation = pc[pairs])
  }

  g <- tryCatch(suppressWarnings(mirt(d[set5], 1, itemtype = "graded", SE = TRUE, verbose = FALSE,
                                      survey.weights = if (pop$weighted) d$w else NULL)),
                error = function(e) NULL)
  if (!is.null(g) && extract.mirt(g, "converged")) {
    cf <- coef(g, printSE = TRUE)
    grm[[length(grm) + 1]] <- do.call(rbind, lapply(set5, function(item) {
      data.frame(population = pop$name, replicate = pop$replicate, item = item,
                 discrimination = cf[[item]]["par", "a1"], se = cf[[item]]["SE", "a1"])
    }))
  } else {
    grm[[length(grm) + 1]] <- data.frame(population = pop$name, replicate = pop$replicate,
                                         item = set5, discrimination = NA, se = NA)
  }
}
categories <- do.call(rbind, categories)
fits <- do.call(rbind, fits)
loadings <- do.call(rbind, loadings)
grm <- do.call(rbind, grm)

# ---- primary estimand: standardized A425 loading in M1 ----
h <- loadings[loadings$population == "human" & loadings$item == "a425", ]
a425 <- do.call(rbind, lapply(split(loadings[loadings$population != "human" & loadings$item == "a425", ],
                                  ~ population + replicate, drop = TRUE), function(s) {
  diff <- s$loading - h$loading
  half <- 1.96 * sqrt(s$se^2 + h$se^2)
  data.frame(population = s$population, replicate = s$replicate, synthetic = s$loading,
             human = h$loading, difference = diff, lower = diff - half, upper = diff + half)
}))
missing <- fits[fits$model == "M1" & !fits$estimable & fits$population != "human", c("population", "replicate")]
if (nrow(missing)) {
  a425 <- rbind(a425, data.frame(missing, synthetic = NA, human = h$loading, difference = NA,
                                 lower = NA, upper = NA))
}
a425 <- a425[order(a425$population, a425$replicate), ]

grm_a425 <- do.call(rbind, lapply(split(grm, ~ population + replicate, drop = TRUE), function(s) {
  others <- s$discrimination[s$item %in% items[1:4]]
  a <- s$discrimination[s$item == "a425"]
  data.frame(population = s$population[[1]], replicate = s$replicate[[1]],
             a425 = if (length(a)) a else NA, mean_a421_a424 = mean(others),
             ratio = if (length(a)) a / mean(others) else NA)
}))
grm_a425 <- grm_a425[order(grm_a425$population, grm_a425$replicate), ]

# ---- invariance: human versus S1 and versus each S3 replicate ----
invariance <- list()
compare <- list(list(name = "S1", data = s1, replicate = 0))
for (k in 1:5) compare[[length(compare) + 1]] <- list(name = "S3", data = s3[[k]], replicate = k)
for (cmp in compare) {
  label <- if (cmp$replicate == 0) cmp$name else paste0(cmp$name, "_r", cmp$replicate)
  hd <- human[c(items, "w")]; hd$population <- "human"
  sd <- cmp$data[items]; sd$w <- 1; sd$population <- "synthetic"
  merged <- merge_categories(list(hd, sd), paste("invariance human vs", label))
  categories <- rbind(categories, merged$log)
  d <- rbind(merged$pops[[1]], merged$pops[[2]])
  set5 <- present(d, items)
  d$w[d$population == "human"] <- d$w[d$population == "human"] / mean(d$w[d$population == "human"])
  steps <- list(configural = "", thresholds = "thresholds", loadings = c("thresholds", "loadings"))
  models <- lapply(steps, function(eq) {
    s <- measEq.syntax(configural.model = syntax(set5), data = d, ordered = set5,
                       parameterization = "theta", ID.fac = "std.lv", ID.cat = "Wu.Estabrook.2016",
                       group = "population", group.equal = if (identical(eq, "")) "" else eq)
    fit <- tryCatch(suppressWarnings(cfa(as.character(s), data = d, group = "population",
                                         ordered = set5, estimator = "WLSMV",
                                         parameterization = "theta", sampling.weights = "w")),
                    error = function(e) NULL)
    if (!is.null(fit) && !lavInspect(fit, "converged")) fit <- NULL
    fit
  })
  inv_measures <- c("chisq.scaled", "df.scaled", "cfi.scaled", "rmsea.scaled")
  for (i in seq_along(models)) {
    m <- if (is.null(models[[i]])) setNames(as.list(rep(NA_real_, 4)), inv_measures)
         else as.list(fitMeasures(models[[i]], inv_measures))
    row <- data.frame(comparison = paste("human vs", cmp$name), replicate = cmp$replicate,
                      step = names(models)[[i]], estimable = !is.null(models[[i]]),
                      m, delta_chisq = NA, delta_df = NA, p = NA, delta_cfi = NA, delta_rmsea = NA)
    if (i > 1 && !is.null(models[[i]]) && !is.null(models[[i - 1]])) {
      test <- lavTestLRT(models[[i - 1]], models[[i]])
      prev <- fitMeasures(models[[i - 1]], c("cfi.scaled", "rmsea.scaled"))
      row$delta_chisq <- test[2, "Chisq diff"]
      row$delta_df <- test[2, "Df diff"]
      row$p <- test[2, "Pr(>Chisq)"]
      row$delta_cfi <- m[["cfi.scaled"]] - prev[["cfi.scaled"]]
      row$delta_rmsea <- m[["rmsea.scaled"]] - prev[["rmsea.scaled"]]
    }
    invariance[[length(invariance) + 1]] <- row
  }
}
invariance <- do.call(rbind, invariance)
invariance$supported <- with(invariance, ifelse(is.na(p), NA, !(p < .05 & delta_cfi < -0.01)))

# ---- S3: median and range over replicates ----
summarise_s3 <- function(tab, value_cols, by) {
  s <- tab[tab$population == "S3", ]
  out <- do.call(rbind, lapply(split(s, s[by], drop = TRUE), function(g) {
    row <- g[1, by, drop = FALSE]
    for (v in value_cols) {
      row[[paste0(v, "_median")]] <- median(g[[v]])
      row[[paste0(v, "_min")]] <- min(g[[v]])
      row[[paste0(v, "_max")]] <- max(g[[v]])
    }
    row
  }))
  rownames(out) <- NULL
  out
}

if (is.null(categories)) {
  categories <- data.frame(analysis = character(), item = character(), merged = character(),
                           into = character(), min_count = integer())
}
write.csv(categories, file.path(output_dir, "part3_category_merges.csv"), row.names = FALSE)
write.csv(fits, file.path(output_dir, "part3_fit.csv"), row.names = FALSE)
write.csv(do.call(rbind, polychoric), file.path(output_dir, "part3_polychoric.csv"), row.names = FALSE)
write.csv(loadings, file.path(output_dir, "part3_loadings.csv"), row.names = FALSE)
write.csv(a425, file.path(output_dir, "part3_a425_loading.csv"), row.names = FALSE)
write.csv(invariance, file.path(output_dir, "part3_invariance.csv"), row.names = FALSE)
write.csv(grm, file.path(output_dir, "part3_grm.csv"), row.names = FALSE)
write.csv(grm_a425, file.path(output_dir, "part3_grm_a425.csv"), row.names = FALSE)
write.csv(summarise_s3(loadings, c("loading"), c("population", "item")),
          file.path(output_dir, "part3_s3_loadings_summary.csv"), row.names = FALSE)
write_json(list(
  benchmark_rds_md5 = unname(tools::md5sum(args[[1]])),
  v2_ledger_md5 = unname(tools::md5sum(args[[2]])),
  v1_responses_md5 = unname(tools::md5sum(args[[3]])),
  human_n = nrow(human), synthetic_n = 300, category_threshold = threshold,
  lavaan = as.character(packageVersion("lavaan")), semTools = as.character(packageVersion("semTools")),
  mirt = as.character(packageVersion("mirt")), r = R.version.string
), file.path(output_dir, "part3_metadata.json"), auto_unbox = TRUE, pretty = TRUE)
cat("Part 3 tables written to", output_dir, "\n")
