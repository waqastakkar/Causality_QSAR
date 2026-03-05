# Pipeline changelog

## 2026-03-05

- Manual pipeline now uses explicit run pointers (`outputs/step5/run_pointer.json`, `outputs/step6/run_pointer.json`) instead of implicit checkpoint discovery in downstream steps.
- Added `scripts/manual/step08a_prepare_external_inhibition.sh` to materialize canonical external inhibition data at:
  `data/external/processed/ptp1b_inhibition_chembl335/data/inhibition_external_final.parquet`.
- Step09 now requires and consumes the canonical external inhibition parquet path above.
- Added preflight checks (missing files/columns) in manual scripts with clear error messages and exit code `2`.
- Step12 manual wrapper now resolves run pointer + screening inputs explicitly and can generate a tiny demo library for smoke tests.
- Added `scripts/pipeline_doctor.py` for end-to-end contract verification.
- Training now writes `artifacts/feature_schema.json` so screening featurization is schema-safe.
