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
