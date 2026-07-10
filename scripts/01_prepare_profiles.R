# requires: dplyr, readr, jsonlite

suppressPackageStartupMessages({
  library(dplyr)
  library(readr)
})

set.seed(20260702)

script_arg <- grep("^--file=", commandArgs(), value = TRUE)
script_path <- if (length(script_arg)) {
  normalizePath(sub("^--file=", "", script_arg[[1]]))
} else {
  normalizePath("scripts/01_prepare_profiles.R")
}
audit_root <- normalizePath(file.path(dirname(script_path), ".."))
data_dir <- file.path(audit_root, "data")
dir.create(data_dir, recursive = TRUE, showWarnings = FALSE)

config <- jsonlite::fromJSON(file.path(audit_root, "config.json"))
n_per_wave <- config$profiles_per_wave
source_rds <- file.path(
  audit_root, "data", "source", "dimension_pilot_results.rds"
)
if (!file.exists(source_rds)) {
  stop("Missing benchmark RDS: ", source_rds)
}

benchmark <- readRDS(source_rds)
if (!all(c("2012", "2018", "2021") %in% names(benchmark$samples))) {
  stop("Benchmark RDS does not contain all three waves.")
}

province_labels <- c(
  `1` = "上海", `2` = "云南", `3` = "内蒙古", `4` = "北京",
  `5` = "吉林", `6` = "四川", `7` = "天津", `8` = "宁夏",
  `9` = "安徽", `10` = "山东", `11` = "山西", `12` = "广东",
  `13` = "广西", `14` = "新疆", `15` = "江苏", `16` = "江西",
  `17` = "河北", `18` = "河南", `19` = "浙江", `20` = "海南",
  `21` = "湖北", `22` = "湖南", `23` = "甘肃", `24` = "福建",
  `25` = "西藏", `26` = "贵州", `27` = "辽宁", `28` = "重庆",
  `29` = "陕西", `30` = "青海", `31` = "黑龙江"
)
economic_labels <- c(
  `1` = "远低于当地平均水平",
  `2` = "低于当地平均水平",
  `3` = "当地平均水平",
  `4` = "高于当地平均水平",
  `5` = "远高于当地平均水平"
)
marital_labels <- c(
  never_married = "未婚",
  partnered = "已婚或同居",
  formerly_partnered = "离婚、分居或丧偶"
)

education_text <- function(years) {
  case_when(
    years == 0 ~ "未接受正规学校教育",
    years <= 6 ~ paste0("约", years, "年教育（小学及以下）"),
    years <= 9 ~ paste0("约", years, "年教育（初中程度）"),
    years <= 12 ~ paste0("约", years, "年教育（高中或中专程度）"),
    years <= 15 ~ paste0("约", years, "年教育（大专程度）"),
    years <= 16 ~ paste0("约", years, "年教育（大学本科程度）"),
    TRUE ~ paste0("约", years, "年教育（研究生程度）")
  )
}

largest_remainder <- function(shares, total) {
  raw <- shares / sum(shares) * total
  base <- floor(raw)
  remaining <- total - sum(base)
  if (remaining > 0) {
    add_to <- order(raw - base, decreasing = TRUE)[seq_len(remaining)]
    base[add_to] <- base[add_to] + 1L
  }
  as.integer(base)
}

sample_wave <- function(data, year, n_target) {
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

  while (any(allocation$target_n > allocation$available_n)) {
    overflow <- sum(pmax(allocation$target_n - allocation$available_n, 0))
    allocation$target_n <- pmin(
      allocation$target_n, allocation$available_n
    )
    eligible <- which(allocation$target_n < allocation$available_n)
    if (!length(eligible) || overflow == 0) break
    extra <- largest_remainder(
      allocation$weighted_size[eligible], overflow
    )
    allocation$target_n[eligible] <- pmin(
      allocation$target_n[eligible] + extra,
      allocation$available_n[eligible]
    )
  }

  selected <- framed |>
    inner_join(allocation |> select(stratum, target_n), by = "stratum") |>
    group_by(stratum) |>
    group_modify(\(d, key) {
      n_take <- unique(d$target_n)
      d[sample.int(nrow(d), n_take, prob = d$weight), , drop = FALSE]
    }) |>
    ungroup()

  if (nrow(selected) != n_target) {
    stop("Sampling produced ", nrow(selected), " rows for ", year)
  }
  selected
}

profiles <- bind_rows(lapply(names(benchmark$samples), function(year) {
  sample_wave(benchmark$samples[[year]], year, n_per_wave)
})) |>
  mutate(
    profile_id = sprintf("CGSS%s_%03d", wave, ave(
      row_id, wave, FUN = seq_along
    )),
    gender_label = if_else(female == 1, "女性", "男性"),
    province_label = unname(province_labels[as.character(as.integer(prov))]),
    province_label = if_else(
      is.na(province_label), paste0("省份代码", prov), province_label
    ),
    education_label = education_text(educ_years),
    hukou_label = if_else(urban_hukou == 1, "城镇户口", "农村户口"),
    residence_label = if_else(
      urban_residence == 1, "城市或城镇居住", "农村居住"
    ),
    marital_label = unname(marital_labels[marital_group]),
    economic_label = unname(
      economic_labels[as.character(as.integer(econ_status))]
    ),
    persona = paste0(
      "调查年份：", wave, "年；所在地区：", province_label,
      "；性别：", gender_label, "；年龄：", round(age), "岁",
      "；教育：", education_label,
      "；户口：", hukou_label,
      "；居住地：", residence_label,
      "；婚姻状态：", marital_label,
      "；家庭经济状况：", economic_label, "。"
    )
  ) |>
  arrange(wave, profile_id)

human_output <- profiles |>
  select(
    profile_id, respondent_id, wave, row_id, prov, female, age,
    educ_years, urban_hukou, urban_residence, partnership,
    marital_group, econ_status, weight, education_group,
    a421, a422, a423, a424, a425, persona
  )
model_output <- human_output |>
  select(
    profile_id, wave, female, age, educ_years, urban_hukou,
    urban_residence, partnership, econ_status, education_group, persona
  )

cat("\n10-row profile preview (human responses withheld from model file):\n")
print(
  model_output |>
    select(
      profile_id, wave, female, age, educ_years,
      urban_residence, persona
    ) |>
    slice_head(n = 10),
  n = 10, width = Inf
)

write_csv(human_output, file.path(data_dir, "profiles_pilot.csv"))
write_csv(model_output, file.path(data_dir, "profiles_llm_input.csv"))

cat("\nProfiles by wave and stratum:\n")
print(
  model_output |>
    count(wave, female, education_group, urban_residence),
  n = Inf
)
cat("\nWrote ", nrow(model_output), " profiles to ", data_dir, "\n", sep = "")
