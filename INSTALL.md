# Installation and Step 0 Unified Pipeline CLI

## Install

```bash
python -m pip install -e .
```

After installation:

```bash
ptp1bqsar --help
```

## Step 0: Unified Pipeline CLI

Step 0 introduces a single orchestration command (`ptp1bqsar`) that manages Steps 1–15 through scripts in `scripts/`.

### Config template

Use `configs/ptp1b.yaml` as the starting point.

### Commands

Validate setup:

```bash
ptp1bqsar check --config configs/ptp1b.yaml
```

Run one step:

```bash
ptp1bqsar step 12 --config configs/ptp1b.yaml --input_path data/screening/raw/AAAA.smi
```

Run a range:

```bash
ptp1bqsar run --config configs/ptp1b.yaml --steps "1-4,8,10-15"
```

Run full (Step 0 documented, Steps 1–15 executed):

```bash
ptp1bqsar run --config configs/ptp1b.yaml --steps "0-15"
```

Build manuscript pack only:

```bash
ptp1bqsar manuscript --config configs/ptp1b.yaml --paper_id ptp1b_causal_qsar_v1
```

## Pipeline run artifacts

Each run creates `outputs/pipeline_runs/<pipeline_run_id>/` with:

- `pipeline_config_resolved.yaml`
- `pipeline_steps_executed.json`
- `pipeline_log.txt`
- `pipeline_errors.json` (if any errors)
- `provenance.json`
- `environment.txt`


## Step 1: Extract target bioactivity from ChEMBL 36 SQLite (production-grade)

### Inputs

- `data/raw/chembl/chembl_36.db`
- Target ChEMBL ID, e.g. `CHEMBL335`

### Output structure

`data/interim/extracts/<TARGET_CHEMBL_ID>/` will contain:

```text
data/interim/extracts/<TARGET>/
├─ <TARGET>_raw.csv
├─ <TARGET>_qsar_ready.csv
├─ extraction_config.json
├─ provenance.json
├─ summary_tables/
│  ├─ counts_by_standard_type.csv
│  ├─ counts_by_units.csv
│  ├─ counts_by_relation.csv
│  ├─ counts_by_confidence.csv
│  └─ missingness_report.csv
└─ figures/
   ├─ fig_standard_type_distribution.svg
   ├─ fig_units_distribution.svg
   ├─ fig_confidence_distribution.svg
   └─ fig_value_distribution_log.svg
```

### Extraction command

```bash
python scripts/extract_chembl36_sqlite.py \
  --db data/raw/chembl/chembl_36.db \
  --target CHEMBLXXXX \
  --outdir data/interim/extracts/CHEMBLXXXX
```

### Reporting command

```bash
python scripts/extract_report.py \
  --input_dir data/interim/extracts/CHEMBLXXXX \
  --outdir data/interim/extracts/CHEMBLXXXX \
  --svg \
  --font "Times New Roman" \
  --bold_text \
  --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

All report figures are written as SVG and style is globally enforced through Matplotlib rcParams (`savefig.format=svg`, `svg.fonttype=none`) with Times New Roman, bold text, and the fixed 5-color palette (`#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`).
