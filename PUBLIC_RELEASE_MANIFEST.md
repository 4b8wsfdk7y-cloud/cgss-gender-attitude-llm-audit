# Public Release Manifest

The public replication archive is code-only with aggregate outputs. CGSS
microdata are licensed and are not redistributed.

## Include

- `README.md`, `config.json`, and `FOLLOWUP_SPECIFICATION.md`
- `scripts/`, `prompts/`, and `ml/scripts/`
- `ml/requirements.txt`, `ml/README.md`, and aggregate files in `ml/output/`
- aggregate `output/metrics_*.csv`, except the replication-level file listed
  below
- `paper_smr/` source files, bibliography, figures, and final PDFs
- `environment/R-session-info.txt`

## Exclude

| Path | Reason |
|---|---|
| `data/raw/` | Licensed CGSS source files |
| `data/source/dimension_pilot_results.rds` | Respondent-level derived survey data |
| `data/profiles_pilot.csv` | Sampled respondent records and held-out answers |
| `ml/data/human_ml_input.csv` | Respondent-level ML analysis data |
| `output/responses*.csv` / `output/responses*.jsonl` | Profile-level model responses linked to sampled survey profiles |
| `output/metrics_human_sampling_replications.csv` | Replication-level simulation output; regenerated locally |

`data/profiles_llm_input.csv` contains synthetic persona text derived from
survey profiles. It is not required for a conservative public archive and
should be excluded unless the CGSS data-use terms have been checked
specifically for this derivative.

## Authorized rebuild

Place the three licensed files in a private directory and run:

```bash
export CGSS_RAW_DIR="/absolute/path/to/authorized/cgss/files"
Rscript scripts/00_build_authorized_benchmark.R
Rscript scripts/01_prepare_profiles.R
Rscript scripts/03_evaluate_audit.R
Rscript scripts/05_extended_validation.R
Rscript ml/scripts/01_export_data.R
ml/.venv/bin/python ml/scripts/02_train_evaluate.py
Rscript paper_smr/reproduce.R
```

The first command creates `data/source/dimension_pilot_results.rds`. All
respondent-level outputs must remain outside the public archive.
