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

## Step 2 — QSAR post-processing (production-grade: tables + provenance + SVG figures)

### Objective

Convert extracted ChEMBL bioactivity into a reproducible QSAR-ready dataset with row-level pIC50 conversion, compound-level aggregation, binary labels, RDKit/Lipinski properties, manuscript-ready SVG figures, and full provenance.

### Inputs

From Step 1:

- `data/interim/extracts/<TARGET>/<TARGET>_qsar_ready.csv`

### Output structure

```text
data/processed/qsar/<TARGET>/
├─ data/
│  ├─ row_level_with_pIC50.csv
│  ├─ compound_level_pIC50.csv
│  ├─ compound_level_with_properties.csv
│  └─ summary.csv
├─ figures/
│  ├─ fig_class_balance.svg
│  ├─ fig_spider_properties_active_vs_inactive.svg
│  ├─ fig_bubble_mw_vs_logp.svg
│  ├─ fig_pIC50_distribution.svg
│  ├─ fig_endpoint_units_relations.svg
│  └─ fig_missingness_properties.svg
└─ provenance/
   ├─ run_config.json
   ├─ provenance.json
   └─ environment.txt
```

### Run Step 2

1) Create folders:

```bash
mkdir -p data/processed/qsar/CHEMBLXXXX/{data,figures,provenance}
```

2) Generate data tables:

```bash
python scripts/qsar_postprocess.py \
  --input data/interim/extracts/CHEMBLXXXX/CHEMBLXXXX_qsar_ready.csv \
  --outdir data/processed/qsar/CHEMBLXXXX/data \
  --endpoint IC50 \
  --threshold 6.0 \
  --aggregate best \
  --prefer_pchembl \
  --svg
```

3) Generate figures + provenance:

```bash
python scripts/qsar_postprocess_report.py \
  --input_dir data/processed/qsar/CHEMBLXXXX/data \
  --outdir data/processed/qsar/CHEMBLXXXX \
  --svg \
  --font "Times New Roman" \
  --bold_text \
  --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

All Step 2 figures are emitted as SVG (editable text, not paths) with `svg.fonttype=none`, Times New Roman, bold text (titles/labels/ticks/legend), and a fixed colorblind-friendly Nature palette (`#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`).

## Step 3 — Assemble multi-environment dataset + validate domain shift + latent environment discovery

### Step 3 output structure

```text
data/processed/environments/<TARGET>/
├─ data/
│  ├─ multienv_row_level.parquet
│  ├─ multienv_compound_level.parquet
│  ├─ env_definitions.json
│  ├─ env_vector_schema.json
│  ├─ env_counts.csv
│  ├─ series_assignments.csv
│  ├─ learned_env_assignments.csv
│  ├─ learned_env_feature_matrix.parquet
│  └─ learned_env_scaler.json
├─ reports/
│  ├─ shift_metrics.csv
│  ├─ env_predictability.csv
│  ├─ scaffold_overlap.csv
│  ├─ label_shift.csv
│  ├─ missingness_by_env.csv
│  ├─ alignment_metrics.csv
│  ├─ cluster_profiles.csv
│  ├─ cluster_purity.csv
│  └─ clustering_stability.csv
├─ figures/
│  ├─ fig_env_counts.svg
│  ├─ fig_label_distribution_by_env.svg
│  ├─ fig_active_rate_by_env.svg
│  ├─ fig_scaffold_overlap_heatmap.svg
│  ├─ fig_shift_metrics.svg
│  ├─ fig_env_predictability.svg
│  ├─ fig_cluster_sizes.svg
│  ├─ fig_cluster_profiles.svg
│  ├─ fig_alignment_ari_nmi.svg
│  └─ fig_manual_vs_learned_contingency.svg
└─ provenance/
   ├─ run_config.json
   ├─ provenance.json
   └─ environment.txt
```

### 3.1 Create folders

```bash
mkdir -p data/processed/environments/CHEMBLXXXX/{data,reports,figures,provenance}
```

### 3.2 Assemble explicit environments

```bash
python scripts/assemble_environments.py \
  --target CHEMBLXXXX \
  --row_level_csv data/processed/qsar/CHEMBLXXXX/data/row_level_with_pIC50.csv \
  --compound_level_csv data/processed/qsar/CHEMBLXXXX/data/compound_level_with_properties.csv \
  --raw_extract_csv data/interim/extracts/CHEMBLXXXX/CHEMBLXXXX_raw.csv \
  --outdir data/processed/environments/CHEMBLXXXX/data \
  --env_keys assay_type species readout publication chemistry_regime series \
  --bbb_rules configs/bbb_rules.yaml \
  --series_rules configs/series_rules.yaml
```

### 3.3 Validate environment shift and leakage risk

```bash
python scripts/env_validation_report.py \
  --input_dir data/processed/environments/CHEMBLXXXX/data \
  --outdir data/processed/environments/CHEMBLXXXX \
  --svg \
  --font "Times New Roman" \
  --bold_text \
  --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### 3.4 Latent environment discovery (unsupervised domains)

Use learned environments to verify domain structure, test alignment with manual environments, and detect hidden assay/publication regimes.

```bash
python scripts/latent_env_discovery.py \
  --input_compound_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --outdir data/processed/environments/CHEMBLXXXX \
  --features MW LogP TPSA HBD HBA RotB Rings \
  --method kmeans \
  --k_min 3 --k_max 12 \
  --select_by silhouette \
  --random_seed 42 \
  --svg \
  --font "Times New Roman" \
  --bold_text \
  --palette nature5 \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

All Step 3 figures are SVG only with editable text (`svg.fonttype=none`), Times New Roman font, bold text for titles/labels/ticks/legends, configurable font sizes, and the fixed Nature palette: `#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`.

## Step 4 — Nature-level strict split suite + leakage audits + shift quantification

### Purpose

Step 4 builds benchmark-grade OOD split manifests (random, scaffold, time, environment holdouts, combo OOD, matched-shift, hard-boundary, and neighbor-similarity) with hard leakage/integrity checks, similarity leakage auditing, split-level shift quantification, reproducible provenance, and manuscript-ready SVG figures.

### Output structure

```text
data/processed/splits/<TARGET>/
├─ splits/
│  ├─ random/
│  ├─ scaffold_bm/
│  ├─ time_publication/
│  ├─ env_holdout_assay/
│  ├─ env_holdout_pubfam/
│  ├─ combo_scaffold_env/
│  ├─ combo_time_env/
│  ├─ scaffold_matched_props/
│  ├─ hard_boundary/
│  └─ neighbor_similarity/
├─ reports/
│  ├─ split_summary.csv
│  ├─ label_shift.csv
│  ├─ covariate_shift.csv
│  ├─ group_integrity_checks.csv
│  ├─ scaffold_overlap.csv
│  ├─ env_overlap.csv
│  ├─ similarity_leakage.csv
│  ├─ matching_quality.csv
│  └─ time_coverage.csv
├─ figures/
│  ├─ fig_split_sizes.svg
│  ├─ fig_label_shift_by_split.svg
│  ├─ fig_covariate_shift_props.svg
│  ├─ fig_scaffold_overlap_by_split.svg
│  ├─ fig_env_overlap_by_split.svg
│  ├─ fig_similarity_leakage.svg
│  ├─ fig_time_split_timeline.svg
│  └─ fig_matching_quality.svg
└─ provenance/
   ├─ run_config.json
   ├─ provenance.json
   └─ environment.txt
```

### Run split generation

```bash
python scripts/make_splits.py \
  --target CHEMBLXXXX \
  --input_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --outdir data/processed/splits/CHEMBLXXXX/splits \
  --seed 42 \
  --train_frac 0.8 --val_frac 0.1 --test_frac 0.1 \
  --enable random scaffold_bm time_publication env_holdout_assay env_holdout_pubfam \
           combo_scaffold_env combo_time_env scaffold_matched_props hard_boundary neighbor_similarity \
  --time_key publication_year \
  --assay_holdout_value cell-based \
  --similarity_radius 2 --similarity_nbits 2048 \
  --neighbor_threshold 0.65 \
  --hard_delta 0.3 \
  --match_props MW LogP TPSA HBD HBA RotB Rings
```

### Run split reporting + SVG figures

```bash
python scripts/splits_report.py \
  --input_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --splits_dir data/processed/splits/CHEMBLXXXX/splits \
  --outdir data/processed/splits/CHEMBLXXXX \
  --font "Times New Roman" \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

### 4.4 BBB-aware stratification

#### Outputs

```text
data/processed/bbb/<TARGET>/
├─ data/
│  ├─ bbb_annotations.parquet
│  ├─ cns_mpo_components.csv
│  ├─ cns_bins.csv
│  └─ pgp_predictions.csv (optional)
├─ reports/
│  ├─ bbb_summary.csv
│  ├─ bbb_shift_by_split.csv
│  └─ cns_vs_non_cns_overlap.csv
├─ figures/
│  ├─ fig_cns_mpo_distribution.svg
│  ├─ fig_cns_like_rate_by_split.svg
│  ├─ fig_potency_vs_cns_mpo.svg
│  ├─ fig_pareto_frontier.svg
│  └─ fig_pgp_risk_distribution.svg (optional)
└─ provenance/
   ├─ run_config.json
   ├─ provenance.json
   └─ environment.txt
```

#### Run command

```bash
python scripts/bbb_stratify.py \
  --target CHEMBLXXXX \
  --input_parquet data/processed/environments/CHEMBLXXXX/data/multienv_compound_level.parquet \
  --splits_dir data/processed/splits/CHEMBLXXXX/splits \
  --outdir data/processed/bbb/CHEMBLXXXX \
  --compute_cns_mpo \
  --cns_mpo_threshold 4.0 \
  --cns_bins 0 2 4 6 \
  --pgp_model_path "" \
  --font "Times New Roman" \
  --font_title 16 --font_label 14 --font_tick 12 --font_legend 12
```

All Step 4 and Step 4.4 figures are SVG-only with editable text (`svg.fonttype=none`), Times New Roman, bold titles/labels/ticks/legend, configurable font sizes, and the fixed Nature 5-color palette (`#E69F00`, `#009E73`, `#0072B2`, `#D55E00`, `#CC79A7`).
