# requires: dplyr, readr
#
# Trust replication (PREREGISTRATION_V2.md, 2026-11-17, point 8): exports the
# eligible CGSS 2012 respondents written by scripts/27_build_trust_profiles.R in
# the column layout of ml/scripts/01_export_data.R, so that the survey-trained
# HGB can be fitted with the features of ml/config.json. Trust items keep the
# 1-4 coding of script 27 (higher = more trust) under the names t_n16xx.
# With BATTERY = issp (PREREGISTRATION_V2.md, 2026-11-18) it does the same for
# the output of scripts/31_build_issp_profiles.R: items m101-m202, coded 1-5
# (higher = more agreement), under the names t_m1xx and t_m2xx.
#
# Usage: Rscript scripts/28_export_trust_ml_input.R DIR [BATTERY]
# DIR is the output directory of script 27 (BATTERY = trust, the default) or
# 31 (issp); BATTERY_ml_input.csv is written there. The file is
# respondent-level and must stay outside the repository. Only aggregate counts
# are printed.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
if (!length(args) %in% 1:2) stop("Usage: 28_export_trust_ml_input.R DIR [BATTERY]")
trust_dir <- normalizePath(args[[1]], mustWork = TRUE)
battery <- if (length(args) == 2) args[[2]] else "trust"
if (!battery %in% c("trust", "issp")) stop("BATTERY must be trust or issp")
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
repo_root <- normalizePath(file.path(dirname(sub("^--file=", "", script_arg[[1]])), ".."))
if (startsWith(trust_dir, repo_root)) stop("Inputs and outputs must stay outside the repository")

bench <- readRDS(file.path(trust_dir, paste0(battery, "_benchmark.rds")))
selected <- bench$selected
n_options <- if (is.null(bench$n_options)) 4L else bench$n_options
profiles <- read_csv(file.path(trust_dir, paste0("profiles_", battery, "_pilot.csv")), show_col_types = FALSE,
                     col_types = cols(.default = col_character())) |>
  transmute(profile_id, row_id = as.integer(row_id))

human <- bench$benchmark |>
  transmute(
    wave = as.character(wave),
    survey_year = as.integer(wave),
    row_id = as.integer(row_id),
    respondent_id = as.character(respondent_id),
    prov = as.character(as.integer(prov)),
    female = as.integer(female),
    age = as.numeric(age),
    educ_years = as.numeric(educ_years),
    urban_hukou = as.integer(urban_hukou),
    urban_residence = as.integer(urban_residence),
    partnership = as.integer(partnership),
    econ_status = as.numeric(econ_status),
    marital_group = as.character(marital_group),
    analysis_weight = as.numeric(weight),
    across(all_of(selected), as.integer, .names = "t_{.col}")
  ) |>
  left_join(profiles, by = "row_id") |>
  mutate(is_llm_profile = as.integer(!is.na(profile_id))) |>
  arrange(row_id)

if (any(human$analysis_weight <= 0)) stop("Analysis weights must be positive.")
if (!all(unlist(human[paste0("t_", selected)]) %in% seq_len(n_options))) {
  stop(sprintf("Items must be complete 1-%d.", n_options))
}
if (sum(human$is_llm_profile) != nrow(profiles)) stop("Not every profile matched a respondent.")

write_csv(human, file.path(trust_dir, paste0(battery, "_ml_input.csv")))
cat(sprintf("respondents: %d; matched profiles: %d; items: %d\n",
            nrow(human), sum(human$is_llm_profile), length(selected)))
