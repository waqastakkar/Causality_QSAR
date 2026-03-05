# Manual pipeline step catalog

- Source of truth for execution order: `scripts/manual/run_all.sh`.
- Logs are written to `outputs/stepX/stepXX_*.log`.
- Each step performs preflight checks and fails with exit code 2 if prerequisites are missing.

## Ordered steps

1. `step01_extract.sh`
2. `step02_postprocess.sh`
3. `step03_assemble_environments.sh`
4. `step04_generate_splits.sh`
5. `step05_benchmark.sh`
6. `step06_train_causal.sh`
7. `step07_counterfactuals.sh`
8. `step08_evaluate_runs.sh`
9. `step08a_prepare_external_inhibition.sh`
10. `step09_cross_endpoint.sh`
11. `step10_interpret.sh`
12. `step11_robustness.sh`
13. `step12_screen_library.sh`
14. `step13_analyze_screening.sh`
15. `step14_match_features.sh`
16. `step15_manuscript.sh`

## Split execution modes

- `training.splits_to_run` unset: run only `training.split_default`.
- `training.splits_to_run=all`: run every split directory under `outputs/step4/`.
- `training.splits_to_run=<comma,list>`: run only those split names.
- Step05/06 publish latest pointers at `outputs/step6/<target>/latest_run.json` and `outputs/step6/<target>/<split>/latest_run.json`.
- Step09/10 resolve those pointers unless `run_dir` is explicitly provided.

## Smoke mode configuration

- Smoke runs must use `configs/ptp1b_smoke.yaml` or pass `smoke=true` override with `configs/ptp1b.yaml`.
- Baseline behavior remains unchanged in `configs/ptp1b.yaml` (`smoke: false`).
- Typical smoke flow:
  1. `python scripts/smoke/make_tiny_step3_parquet.py`
  2. `bash scripts/manual/step04_generate_splits.sh configs/ptp1b_smoke.yaml`
  3. `bash scripts/manual/step05_benchmark.sh configs/ptp1b_smoke.yaml training.splits_to_run=scaffold_bm`
  4. `bash scripts/manual/step06_train_causal.sh configs/ptp1b_smoke.yaml training.splits_to_run=scaffold_bm`
  5. `bash scripts/manual/step08_evaluate_runs.sh configs/ptp1b_smoke.yaml`
