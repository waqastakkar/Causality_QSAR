# Human Focus International PTP1B Causal QSAR + Screening Pipeline

**Tagline:** Reproducible, causality-aware QSAR modeling and virtual screening for PTP1B lead discovery.

This repository implements an end-to-end pipeline for **causal QSAR development**, robust model evaluation, and **large-scale screening** around the PTP1B target (e.g., CHEMBL335). It addresses a common scientific challenge in medicinal chemistry: models that perform well in-distribution but fail under assay/domain shifts. By combining multi-environment data assembly, invariance-driven diagnostics, counterfactual analysis, interpretability, and manuscript-ready reporting, the project supports both practical screening and publication-grade reproducibility.

Compared with many conventional QSAR workflows, this pipeline emphasizes:

- explicit environment-aware splitting and validation,
- provenance-first execution (run manifests, hashes, environment capture),
- integrated manuscript pack generation,
- one unified CLI (`ptp1bqsar`) for steps 0–15.

## Table of Contents

- [Human Focus International PTP1B Causal QSAR + Screening Pipeline](#human-focus-international-ptp1b-causal-qsar--screening-pipeline)
  - [Table of Contents](#table-of-contents)
  - [Features](#features)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Usage](#usage)
    - [Unified CLI](#unified-cli)
    - [Check configuration](#check-configuration)
    - [Run entire pipeline (Steps 0–15)](#run-entire-pipeline-steps-015)
    - [Run specific steps](#run-specific-steps)
    - [Manuscript builder](#manuscript-builder)
    - [Step catalog (0–15)](#step-catalog-015)
  - [Output folders and file explanations](#output-folders-and-file-explanations)
  - [Reproducibility](#reproducibility)
  - [Example end-to-end workflow](#example-end-to-end-workflow)
  - [Troubleshooting](#troubleshooting)
  - [Contributing](#contributing)
  - [Citation](#citation)
  - [License](#license)
  - [Contact](#contact)

## Features

- Unified orchestration CLI for checking, single-step runs, range runs, and manuscript assembly.
- Stepwise pipeline for extraction, processing, environment construction, training, evaluation, and screening.
- Causal-QSAR orientation: environment shift diagnostics and robustness analysis.
- Counterfactual generation for molecule-level what-if analysis.
- Interpretability stage for model explanation assets.
- Virtual screening + post-screening analysis + feature matching.
- Publication-oriented report/manuscript artifact generation.
- Reproducibility artifacts per run: resolved config, executed steps, logs, errors, provenance, environment snapshots.

## Prerequisites

| Requirement | Recommendation |
|---|---|
| Python | 3.10+ |
| Core stack | RDKit, PyTorch 2.x, NumPy/Pandas/SciPy/Scikit-learn |
| OS | Linux/macOS (Linux recommended for GPU workflows) |
| Hardware | 16+ GB RAM recommended; NVIDIA GPU optional for acceleration |

Environment setup options:

1. **Recommended:** Conda environment via `environment.yml` in this repository.
2. **Supplemental:** pip packages from `requirements.txt` for pip-only workflows.

For full platform notes and command-level setup, see [`INSTALL.md`](INSTALL.md).

## Installation

1. Create and activate Conda environment (recommended):

```bash
conda env create -f environment.yml
conda activate ptp1b-causal-qsar
```

2. Install the project package in editable mode:

```bash
python -m pip install -e .
```

3. Confirm CLI availability:

```bash
ptp1bqsar --help
```

For additional installation details and step-specific notes, see [`INSTALL.md`](INSTALL.md).

## Configuration

Use `configs/ptp1b.yaml` as your baseline.

Example template:

```yaml
paper_id: ptp1b_causal_qsar_v1
target: CHEMBL335

paths:
  chembl_sqlite: data/raw/chembl_36.db
  data_root: data
  outputs_root: outputs

style:
  svg_only: true
  font: Times New Roman
  bold_text: true
  palette: nature5
  font_title: 16
  font_label: 14
  font_tick: 12
  font_legend: 12

training:
  task: regression
  label_col: pIC50
  seeds: [1, 2, 3, 4, 5]
  split_default: scaffold_bm

robustness:
  ensemble_size: 5
  conformal_coverage: 0.90
  ad_threshold: 0.35

screening:
  input_format: smi
  smi_layout: smiles_id
  header: auto
  smiles_col_name: smiles
  id_col_name: zinc_id
  cns_mpo_threshold: 4.0
  topk: 500
```

### Main configuration keys

| Key | Purpose |
|---|---|
| `paper_id` | Identifier used for manuscript/release artifacts |
| `target` | Target entity (e.g., CHEMBL335 for PTP1B) |
| `paths.*` | Input/output root paths, including ChEMBL SQLite location |
| `style.*` | Consistent figure typography and palette parameters |
| `training.*` | Task type, labels, seeds, and default split strategy |
| `robustness.*` | Ensemble/conformal/applicability domain controls |
| `screening.*` | Input schema and prioritization settings for virtual screening |

## Usage

### Unified CLI

```bash
ptp1bqsar --help
```

Shows commands for `check`, `step`, `run`, and `manuscript`.

### Check configuration

```bash
ptp1bqsar check --config configs/ptp1b.yaml
```

Validates schema, key paths, script availability, and RDKit import readiness.

### Run entire pipeline (Steps 0–15)

```bash
ptp1bqsar run --config configs/ptp1b.yaml --steps 0-15
```

Runs the complete workflow end to end. Step 0 is orchestration/checkpoint context; executable pipeline scripts map to Steps 1–15.

### Run specific steps

Step 1 (extract target data from ChEMBL SQLite):

```bash
ptp1bqsar step 1 --config configs/ptp1b.yaml
```

Step 5 (model training benchmark):

```bash
ptp1bqsar step 5 --config configs/ptp1b.yaml
```

Step 12 (screening):

```bash
ptp1bqsar step 12 --config configs/ptp1b.yaml
```

You can also run ranges and mixed selections:

```bash
ptp1bqsar run --config configs/ptp1b.yaml --steps "1-4,8,10-15"
```

### Manuscript builder

```bash
ptp1bqsar manuscript --config configs/ptp1b.yaml --paper_id ptp1b_causal_qsar_v1
```

Runs manuscript packaging logic (Step 15) with a specified paper identifier.

### Step catalog (0–15)

| Step | Name | Purpose |
|---|---|---|
| 0 | unified orchestration | Common entrypoint and run bookkeeping |
| 1 | `extract_chembl36_sqlite` | Extract target-linked bioactivity records |
| 2 | `qsar_postprocess` | Convert to QSAR-ready tables and summary outputs |
| 3 | `assemble_environments` | Build multi-environment datasets for shift-aware analysis |
| 4 | `generate_splits` | Create train/validation/test splits |
| 5 | `run_benchmark` | Train baseline/benchmark models |
| 6 | `reserved_step6` | Reserved no-op placeholder |
| 7 | `generate_counterfactuals` | Generate structural counterfactual proposals |
| 8 | `evaluate_model` | Evaluate predictive performance |
| 9 | `evaluate_cross_endpoint` | Cross-endpoint generalization analysis |
| 10 | `interpret_model` | Interpretability artifacts |
| 11 | `evaluate_robustness` | Robustness/conformal/AD diagnostics |
| 12 | `screen_library` | Virtual screening of external libraries |
| 13 | `analyze_screening` | Post-screening ranking and report analytics |
| 14 | `match_screening_features` | Feature-level match and enrichment analysis |
| 15 | `build_manuscript_pack` | Build publication-ready artifact bundle |

## Output folders and file explanations

Primary runtime outputs are written under:

```text
outputs/
└── pipeline_runs/
    └── <pipeline_run_id>/
        ├── pipeline_config_resolved.yaml
        ├── pipeline_steps_executed.json
        ├── pipeline_log.txt
        ├── pipeline_errors.json          # only when errors occur
        ├── provenance.json
        └── environment.txt
```

Step-specific data products are additionally written into pipeline-defined subfolders (e.g., processed QSAR tables, environment reports, screening outputs, and manuscript package files).

### What each step writes (high level)

- Steps 1–3: extraction/processing tables, environment datasets, and associated reports/figures.
- Steps 4–5: split definitions, model checkpoints, training metrics.
- Steps 7–11: counterfactual sets, evaluation summaries, interpretability and robustness outputs.
- Steps 12–14: screened candidate rankings, screening analytics, and feature matching artifacts.
- Step 15: manuscript-aligned package, checklist, and reproducibility metadata.

## Reproducibility

This project is designed around reproducible execution:

- **Resolved configuration capture** in `pipeline_config_resolved.yaml`.
- **Step manifest** in `pipeline_steps_executed.json`.
- **Runtime logs and failures** in `pipeline_log.txt` and `pipeline_errors.json`.
- **Provenance record** in `provenance.json` with run-level metadata.
- **Environment snapshot** in `environment.txt`.

For manuscript workflows, include and review:

- `manuscript_checklist.md` (completeness and reporting checklist).
- **Reproducibility Fingerprint** (recommended composite of config hash, code commit, dependency environment, and input manifest checksum).

Best practice:

1. Pin config and seed values.
2. Record commit SHA with each run.
3. Archive input file hashes and output manifest.
4. Store run directories immutably for published results.

## Example end-to-end workflow

1. Create/activate environment and install package.
2. Prepare `configs/ptp1b.yaml` and ensure `paths.chembl_sqlite` points to your local ChEMBL SQLite.
3. Validate setup:

```bash
ptp1bqsar check --config configs/ptp1b.yaml --steps 0-15
```

4. Execute all steps:

```bash
ptp1bqsar run --config configs/ptp1b.yaml --steps 0-15
```

5. Build manuscript package with a custom identifier:

```bash
ptp1bqsar manuscript --config configs/ptp1b.yaml --paper_id hfintl_ptp1b_v1
```

6. Review artifacts under `outputs/pipeline_runs/<pipeline_run_id>/` and step-specific output directories.

## Troubleshooting

| Symptom | Likely cause | Suggested fix |
|---|---|---|
| `RDKit import failed` during `check` | RDKit not installed in active env | Recreate env from `environment.yml`; ensure `conda-forge` channel is available |
| Missing script error for a step | Incomplete checkout or path issue | Confirm repository integrity and run from repo root |
| `Required path does not exist` warnings | Dataset paths not prepared | Create folders/data files referenced by config |
| CUDA unavailable | GPU runtime package mismatch | Use `pytorch-cuda` version compatible with your driver per `environment.yml` comments |
| Manuscript build missing files | Upstream steps incomplete | Re-run required upstream steps or full `0-15` range |

Additional setup and pipeline details are documented in [`INSTALL.md`](INSTALL.md).

## Contributing

Recommended contribution workflow:

- Branch naming: `feature/<topic>`, `fix/<topic>`, `docs/<topic>`, `refactor/<topic>`.
- Issue labels (suggested): `bug`, `enhancement`, `docs`, `question`, `reproducibility`, `good first issue`.
- Pull request process:
  1. Open an issue or discussion for major changes.
  2. Create a focused branch.
  3. Add/update tests or validation scripts where applicable.
  4. Run key checks locally.
  5. Submit PR with motivation, methodology, and output summary.

## Citation

If you use this pipeline in research or screening campaigns, please cite it as:

```text
Human Focus International. PTP1B Causal QSAR + Screening Pipeline.
GitHub repository, versioned release.
```

You may also include target and commit-specific provenance for exact computational reproducibility.

## License

MIT License (recommended for permissive scientific collaboration).

If your project policy differs, replace this section with the repository’s canonical license text/file.

## Contact

- Email: `research@humanfocus.international`
- GitHub: `@human-focus-international`
