# requires: dplyr, readr
#
# Trust replication (PREREGISTRATION_V2.md, 2026-11-17, point 8): exports the
# eligible CGSS 2012 respondents written by scripts/27_build_trust_profiles.R in
# the column layout of ml/scripts/01_export_data.R, so that the survey-trained
# HGB can be fitted with the features of ml/config.json. Trust items keep the
# 1-4 coding of script 27 (higher = more trust) under the names t_n16xx.
#
# Usage: Rscript scripts/28_export_trust_ml_input.R TRUST_DIR
# TRUST_DIR is the output directory of script 27; trust_ml_input.csv is written
# there. The file is respondent-level and must stay outside the repository.
# Only aggregate counts are printed.

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: 28_export_trust_ml_input.R TRUST_DIR")
trust_dir <- normalizePath(args[[1]], mustWork = TRUE)
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
repo_root <- normalizePath(file.path(dirname(sub("^--file=", "", script_arg[[1]])), ".."))
if (startsWith(trust_dir, repo_root)) stop("Inputs and outputs must stay outside the repository")

bench <- readRDS(file.path(trust_dir, "trust_benchmark.rds"))
selected <- bench$selected
profiles <- read_csv(file.path(trust_dir, "profiles_trust_pilot.csv"), show_col_types = FALSE,
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
if (!all(unlist(human[paste0("t_", selected)]) %in% 1:4)) stop("Trust items must be complete 1-4.")
if (sum(human$is_llm_profile) != nrow(profiles)) stop("Not every profile matched a respondent.")

write_csv(human, file.path(trust_dir, "trust_ml_input.csv"))
cat(sprintf("respondents: %d; matched profiles: %d; items: %d\n",
            nrow(human), sum(human$is_llm_profile), length(selected)))
