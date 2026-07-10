# requires: haven, dplyr, tibble
#
# Rebuild the restricted benchmark object from authorized CGSS .dta files.
# Set CGSS_RAW_DIR to the directory containing CGSS2012.dta,
# CGSS2018.dta, and CGSS2021.dta. The output remains restricted because it
# contains respondent-level survey records.

suppressPackageStartupMessages({
  library(haven)
  library(dplyr)
  library(tibble)
})

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1]]))
} else {
  normalizePath("scripts/00_build_authorized_benchmark.R")
}
audit_root <- normalizePath(file.path(dirname(script_path), ".."))

raw_dir <- Sys.getenv(
  "CGSS_RAW_DIR",
  unset = file.path(audit_root, "data", "raw")
)
out_dir <- file.path(audit_root, "data", "source")
out_file <- file.path(out_dir, "dimension_pilot_results.rds")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)

required_files <- c("CGSS2012.dta", "CGSS2018.dta", "CGSS2021.dta")
missing_files <- required_files[!file.exists(file.path(raw_dir, required_files))]
if (length(missing_files)) {
  stop(
    "Missing authorized CGSS files in ", raw_dir, ": ",
    paste(missing_files, collapse = ", "),
    "\nSet CGSS_RAW_DIR to the correct directory."
  )
}

educ_map <- c(
  `1` = 0, `2` = 3, `3` = 6, `4` = 9,
  `5` = 12, `6` = 12, `7` = 12, `8` = 12,
  `9` = 15, `10` = 15, `11` = 16, `12` = 16,
  `13` = 19, `14` = NA_real_
)

wave_specs <- list(
  `2012` = list(
    file = "CGSS2012.dta", names = "lower", birth = "a3a",
    marital = "a69", gender = paste0("a42", 1:5),
    internet = "a285", reading = "a3004", culture = "a3005",
    online = "a3012", weight = "weight", residence = "isurban",
    urban_residence_codes = 2, rural_residence_codes = 1,
    urban_hukou_codes = c(2, 3, 5), rural_hukou_codes = c(1, 4),
    missing_codes = c(-3, -2, -1)
  ),
  `2018` = list(
    file = "CGSS2018.dta", names = "lower", birth = "a31",
    marital = "a69", gender = paste0("a42", 1:5),
    internet = "a285", reading = "a304", culture = "a305",
    online = "a3012", weight = "weight", residence = "isurban",
    urban_residence_codes = 1, rural_residence_codes = 2,
    urban_hukou_codes = c(2, 4), rural_hukou_codes = c(1, 3),
    missing_codes = c(98, 99)
  ),
  `2021` = list(
    file = "CGSS2021.dta", names = "upper", birth = "A3_1",
    marital = "A69", gender = paste0("A42_", 1:5),
    internet = "A28_5", reading = "A30_4", culture = "A30_5",
    online = "A30_12", weight = "weight", residence = "isurban",
    urban_residence_codes = 1, rural_residence_codes = 2,
    urban_hukou_codes = c(2, 4), rural_hukou_codes = c(1, 3),
    missing_codes = c(98, 99)
  )
)

clean_numeric <- function(x, missing_codes, valid = NULL) {
  value <- suppressWarnings(as.numeric(x))
  value[is.na(value) | value %in% missing_codes] <- NA_real_
  if (!is.null(valid)) {
    value[!value %in% valid] <- NA_real_
  }
  value
}

binary_from_codes <- function(x, one_codes, zero_codes) {
  case_when(
    x %in% one_codes ~ 1,
    x %in% zero_codes ~ 0,
    TRUE ~ NA_real_
  )
}

row_mean_min <- function(data, min_valid) {
  valid_n <- rowSums(!is.na(data))
  value <- rowMeans(data, na.rm = TRUE)
  value[valid_n < min_valid] <- NA_real_
  value
}

build_wave <- function(year, spec) {
  raw <- read_dta(file.path(raw_dir, spec$file))
  var <- function(lower, upper = toupper(lower)) {
    if (spec$names == "upper") upper else lower
  }
  clean <- function(name, valid = NULL) {
    if (!name %in% names(raw)) {
      stop("Variable ", name, " is absent from ", spec$file)
    }
    clean_numeric(raw[[name]], spec$missing_codes, valid)
  }

  gender_items <- lapply(spec$gender, clean, valid = 1:5)
  media_items <- tibble(
    internet = clean(spec$internet, 1:5),
    reading = 6 - clean(spec$reading, 1:5),
    culture = 6 - clean(spec$culture, 1:5),
    online = 6 - clean(spec$online, 1:5)
  )

  tibble(
    wave = as.character(year),
    row_id = seq_len(nrow(raw)),
    prov = clean(var("s41", "s41")),
    female = binary_from_codes(clean(var("a2", "A2")), 2, 1),
    birth_year = clean(spec$birth),
    educ_raw = clean(var("a7a", "A7a")),
    hukou_raw = clean(var("a18", "A18")),
    residence_raw = clean(spec$residence),
    marital_raw = clean(spec$marital, 1:7),
    econ_status = clean(var("a64", "A64"), 1:5),
    a421 = gender_items[[1]],
    a422 = gender_items[[2]],
    a423 = gender_items[[3]],
    a424 = gender_items[[4]],
    a425 = gender_items[[5]],
    internet_raw = media_items$internet,
    reading_raw = media_items$reading,
    culture_raw = media_items$culture,
    online_raw = media_items$online,
    media_raw = row_mean_min(media_items, 2),
    weight = clean(spec$weight)
  ) |>
    mutate(
      public_role_raw = row_mean_min(
        tibble(6 - a421, 6 - a422, 6 - a423, 6 - a424),
        min_valid = 3
      ),
      household_raw = a425,
      age = as.numeric(year) - birth_year,
      age = if_else(age >= 18 & age <= 100, age, NA_real_),
      age2 = age^2,
      educ_years = unname(educ_map[as.character(educ_raw)]),
      urban_hukou = binary_from_codes(
        hukou_raw, spec$urban_hukou_codes, spec$rural_hukou_codes
      ),
      urban_residence = binary_from_codes(
        residence_raw,
        spec$urban_residence_codes,
        spec$rural_residence_codes
      ),
      partnership = case_when(
        marital_raw %in% c(2, 3, 4) ~ 1,
        marital_raw %in% c(1, 5, 6, 7) ~ 0,
        TRUE ~ NA_real_
      ),
      marital_group = case_when(
        marital_raw == 1 ~ "never_married",
        marital_raw %in% c(2, 3, 4) ~ "partnered",
        marital_raw %in% c(5, 6, 7) ~ "formerly_partnered",
        TRUE ~ NA_character_
      )
    ) |>
    filter(
      complete.cases(
        prov, female, age, age2, educ_years, econ_status,
        urban_hukou, urban_residence, public_role_raw, household_raw,
        media_raw, weight
      ),
      weight > 0
    ) |>
    mutate(
      respondent_id = paste(wave, row_id, sep = "_"),
      public_role_z = as.numeric(scale(public_role_raw)),
      household_z = as.numeric(scale(household_raw)),
      media_culture_z = as.numeric(scale(media_raw)),
      media_internet_z = as.numeric(scale(
        row_mean_min(tibble(internet_raw, online_raw), 1)
      )),
      media_reading_z = as.numeric(scale(
        row_mean_min(tibble(reading_raw, culture_raw), 1)
      ))
    )
}

samples <- lapply(names(wave_specs), function(year) {
  build_wave(year, wave_specs[[year]])
})
names(samples) <- names(wave_specs)

preview <- bind_rows(samples, .id = "source_wave") |>
  select(
    wave, row_id, prov, female, age, educ_years,
    a421, a422, a423, a424, a425, weight
  ) |>
  slice_head(n = 10)

cat("\n10-row benchmark preview:\n")
print(preview, n = 10, width = Inf)

saveRDS(list(preview = preview, samples = samples), out_file)
cat(
  "\nRestricted benchmark written to:\n", out_file,
  "\nRows by wave:\n"
)
print(vapply(samples, nrow, integer(1)))
