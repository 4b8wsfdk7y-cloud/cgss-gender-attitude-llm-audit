# requires: haven, dplyr, readr
#
# Trust replication (PREREGISTRATION_V2.md, 2026-11-17): builds the CGSS 2012
# trust benchmark and samples 300 profiles with the stratified scheme of
# scripts/01_prepare_profiles.R. Demographic coding follows
# scripts/00_build_authorized_benchmark.R for 2012. Trust items N1601-N1616 are
# recoded so that higher values mean more trust (5 - CGSS code).
#
# Usage: Rscript scripts/27_build_trust_profiles.R OUTPUT_DIR
# Set CGSS_RAW_DIR to the directory containing CGSS2012.dta. All outputs are
# respondent- or profile-level and must stay outside the repository. Only
# aggregate counts are printed.

suppressPackageStartupMessages({
  library(haven)
  library(dplyr)
  library(readr)
})

args <- commandArgs(trailingOnly = TRUE)
if (length(args) != 1) stop("Usage: 27_build_trust_profiles.R OUTPUT_DIR")
out_dir <- normalizePath(args[[1]], mustWork = FALSE)
script_arg <- grep("^--file=", commandArgs(), value = TRUE)
repo_root <- normalizePath(file.path(dirname(sub("^--file=", "", script_arg[[1]])), ".."))
if (startsWith(out_dir, repo_root)) stop("Outputs must be written outside the repository")
dir.create(out_dir, recursive = TRUE, showWarnings = FALSE)
raw_dir <- Sys.getenv("CGSS_RAW_DIR", unset = file.path(repo_root, "data", "raw"))

set.seed(20261117)
n_profiles <- 300
all_items <- sprintf("n16%02d", 1:16)
trust_labels <- c("亲戚", "朋友", "邻居", "同事", "陌生人", "医生", "银行职员", "公司高管",
                  "记者", "非盈利/非政府组织的领导", "老师", "地方政府官员", "中央政府官员",
                  "警察", "军官", "法官")
selected <- c("n1601", "n1603", "n1605", "n1606", "n1611", "n1612", "n1613", "n1614")
missing_codes <- c(-3, -2, -1)
educ_map <- c(`1` = 0, `2` = 3, `3` = 6, `4` = 9, `5` = 12, `6` = 12, `7` = 12, `8` = 12,
              `9` = 15, `10` = 15, `11` = 16, `12` = 16, `13` = 19, `14` = NA_real_)

clean <- function(x, valid = NULL) {
  v <- suppressWarnings(as.numeric(x))
  v[is.na(v) | v %in% missing_codes] <- NA_real_
  if (!is.null(valid)) v[!v %in% valid] <- NA_real_
  v
}
binary <- function(x, one, zero) case_when(x %in% one ~ 1, x %in% zero ~ 0, TRUE ~ NA_real_)

raw <- read_dta(file.path(raw_dir, "CGSS2012.dta"),
                col_select = c("s41", "a2", "a3a", "a7a", "a18", "isurban", "a69", "a64",
                               "weight", all_of(all_items)))
trust <- as.data.frame(lapply(raw[all_items], function(x) 5 - clean(x, 1:4)))

benchmark <- tibble(
  wave = "2012", row_id = seq_len(nrow(raw)),
  prov = clean(raw$s41), female = binary(clean(raw$a2), 2, 1),
  birth_year = clean(raw$a3a), educ_raw = clean(raw$a7a), hukou_raw = clean(raw$a18),
  residence_raw = clean(raw$isurban), marital_raw = clean(raw$a69, 1:7),
  econ_status = clean(raw$a64, 1:5), weight = clean(raw$weight)
) |>
  bind_cols(trust) |>
  mutate(
    age = 2012 - birth_year,
    age = if_else(age >= 18 & age <= 100, age, NA_real_),
    educ_years = unname(educ_map[as.character(educ_raw)]),
    urban_hukou = binary(hukou_raw, c(2, 3, 5), c(1, 4)),
    urban_residence = binary(residence_raw, 2, 1),
    partnership = case_when(marital_raw %in% 2:4 ~ 1, marital_raw %in% c(1, 5, 6, 7) ~ 0),
    marital_group = case_when(marital_raw == 1 ~ "never_married",
                              marital_raw %in% 2:4 ~ "partnered",
                              marital_raw %in% 5:7 ~ "formerly_partnered")
  ) |>
  filter(complete.cases(prov, female, age, educ_years, econ_status, urban_hukou,
                        urban_residence, marital_group, weight),
         weight > 0,
         if_all(all_of(selected), ~ !is.na(.x))) |>
  mutate(respondent_id = paste(wave, row_id, sep = "_"),
         education_group = case_when(educ_years <= 9 ~ "low", educ_years <= 12 ~ "middle",
                                     TRUE ~ "high"))

largest_remainder <- function(shares, total) {
  raw_n <- shares / sum(shares) * total
  base <- floor(raw_n)
  extra <- total - sum(base)
  if (extra > 0) {
    add <- order(raw_n - base, decreasing = TRUE)[seq_len(extra)]
    base[add] <- base[add] + 1L
  }
  as.integer(base)
}
framed <- benchmark |>
  mutate(stratum = interaction(female, education_group, urban_residence, drop = TRUE, sep = "_"))
allocation <- framed |>
  group_by(stratum) |>
  summarise(weighted_size = sum(weight), available_n = n(), .groups = "drop") |>
  mutate(target_n = largest_remainder(weighted_size, n_profiles))
stopifnot(all(allocation$target_n <= allocation$available_n))
sample_rows <- framed |>
  inner_join(select(allocation, stratum, target_n), by = "stratum") |>
  group_by(stratum) |>
  group_modify(\(d, key) d[sample.int(nrow(d), unique(d$target_n), prob = d$weight), , drop = FALSE]) |>
  ungroup()
stopifnot(nrow(sample_rows) == n_profiles)

province_labels <- c(`1` = "上海", `2` = "云南", `3` = "内蒙古", `4` = "北京", `5` = "吉林",
  `6` = "四川", `7` = "天津", `8` = "宁夏", `9` = "安徽", `10` = "山东", `11` = "山西",
  `12` = "广东", `13` = "广西", `14` = "新疆", `15` = "江苏", `16` = "江西", `17` = "河北",
  `18` = "河南", `19` = "浙江", `20` = "海南", `21` = "湖北", `22` = "湖南", `23` = "甘肃",
  `24` = "福建", `25` = "西藏", `26` = "贵州", `27` = "辽宁", `28` = "重庆", `29` = "陕西",
  `30` = "青海", `31` = "黑龙江")
economic_labels <- c(`1` = "远低于当地平均水平", `2` = "低于当地平均水平", `3` = "当地平均水平",
                     `4` = "高于当地平均水平", `5` = "远高于当地平均水平")
marital_labels <- c(never_married = "未婚", partnered = "已婚或同居",
                    formerly_partnered = "离婚、分居或丧偶")
education_text <- function(years) case_when(
  years == 0 ~ "未接受正规学校教育",
  years <= 6 ~ paste0("约", years, "年教育（小学及以下）"),
  years <= 9 ~ paste0("约", years, "年教育（初中程度）"),
  years <= 12 ~ paste0("约", years, "年教育（高中或中专程度）"),
  years <= 15 ~ paste0("约", years, "年教育（大专程度）"),
  years <= 16 ~ paste0("约", years, "年教育（大学本科程度）"),
  TRUE ~ paste0("约", years, "年教育（研究生程度）"))

profiles <- sample_rows |>
  arrange(stratum, row_id) |>
  mutate(
    profile_id = sprintf("TRUST2012_%03d", row_number()),
    province_label = unname(province_labels[as.character(as.integer(prov))]),
    province_label = if_else(is.na(province_label), paste0("省份代码", prov), province_label),
    persona = paste0(
      "调查年份：", wave, "年；所在地区：", province_label,
      "；性别：", if_else(female == 1, "女性", "男性"), "；年龄：", round(age), "岁",
      "；教育：", education_text(educ_years),
      "；户口：", if_else(urban_hukou == 1, "城镇户口", "农村户口"),
      "；居住地：", if_else(urban_residence == 1, "城市或城镇居住", "农村居住"),
      "；婚姻状态：", unname(marital_labels[marital_group]),
      "；家庭经济状况：", unname(economic_labels[as.character(as.integer(econ_status))]), "。")
  )

model_cols <- c("profile_id", "wave", "female", "age", "educ_years", "urban_hukou",
                "urban_residence", "partnership", "econ_status", "education_group", "persona")
write_csv(select(profiles, all_of(model_cols)), file.path(out_dir, "profiles_trust_llm_input.csv"))
write_csv(select(profiles, profile_id, respondent_id, row_id, prov, marital_group, weight,
                 all_of(model_cols[-1]), all_of(all_items)),
          file.path(out_dir, "profiles_trust_pilot.csv"))
saveRDS(list(benchmark = benchmark, items = all_items, labels = trust_labels,
             selected = selected, seed = 20261117,
             raw_md5 = unname(tools::md5sum(file.path(raw_dir, "CGSS2012.dta")))),
        file.path(out_dir, "trust_benchmark.rds"))

cat("Eligible respondents (complete demographics and the 8 selected items):", nrow(benchmark), "\n")
cat("Profiles written:", nrow(profiles), "\n")
print(count(profiles, female, education_group, urban_residence), n = Inf)
