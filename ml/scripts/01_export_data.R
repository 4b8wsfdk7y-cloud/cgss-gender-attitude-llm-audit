# requires: dplyr, readr

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1]]))
} else {
  normalizePath("ml/scripts/01_export_data.R")
}
audit_root <- normalizePath(file.path(dirname(script_path), "..", ".."))
ml_root <- file.path(audit_root, "ml")
data_dir <- file.path(ml_root, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

benchmark_path <- file.path(
  audit_root, "data", "source", "dimension_pilot_results.rds"
)
profiles_path <- file.path(audit_root, "data", "profiles_pilot.csv")
if (!all(file.exists(c(benchmark_path, profiles_path)))) {
  stop("Missing benchmark RDS or LLM profile file.")
}

benchmark <- readRDS(benchmark_path)
profiles <- read_csv(profiles_path, show_col_types = FALSE) |>
  select(profile_id, wave, row_id)

human <- bind_rows(lapply(names(benchmark$samples), function(year) {
  benchmark$samples[[year]] |>
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
      analysis_weight = as.numeric(weight),
      raw_a421 = as.integer(a421),
      raw_a422 = as.integer(a422),
      raw_a423 = as.integer(a423),
      raw_a424 = as.integer(a424),
      raw_a425 = as.integer(a425),
      eq_a421 = as.integer(6 - a421),
      eq_a422 = as.integer(6 - a422),
      eq_a423 = as.integer(6 - a423),
      eq_a424 = as.integer(6 - a424),
      eq_a425 = as.integer(a425)
    )
})) |>
  left_join(
    profiles |> mutate(wave = as.character(wave)),
    by = c("wave", "row_id")
  ) |>
  mutate(is_llm_profile = as.integer(!is.na(profile_id))) |>
  arrange(survey_year, row_id)

if (any(human$analysis_weight <= 0, na.rm = TRUE)) {
  stop("Analysis weights must be positive.")
}
if (!all(
  unlist(human[paste0("eq_a42", 1:5)]) %in% c(1:5, NA_integer_)
)) {
  stop("Equality-oriented outcomes contain values outside 1..5.")
}

cat("\nSeven-row ML input preview (all outcomes: higher = more egalitarian):\n")
print(
  human |>
    select(
      wave, row_id, female, age, educ_years, urban_residence,
      starts_with("eq_a42"), profile_id
    ) |>
    slice_head(n = 7),
  n = 7, width = Inf
)

write_csv(human, file.path(data_dir, "human_ml_input.csv"), na = "")
cat(
  "\nWrote ", nrow(human), " respondents; matched LLM profiles: ",
  sum(human$is_llm_profile), "\n", sep = ""
)
