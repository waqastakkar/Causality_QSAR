# AGENTS.md

## Repository expectations

- Treat scripts/manual/*.sh as the source of truth for execution order and I/O locations.
- For every step, enforce a stable I/O contract:
  - required input files
  - required columns for tabular data
  - guaranteed output files
- Prefer explicit pointers to the selected run directory (write a JSON pointer file) over “search for the first checkpoint”.
- Keep external processed datasets in:
  data/external/processed/<target_namespace>/data/
- Every step must log to outputs/stepX/stepXX_*.log and fail with a clear message when prerequisites are missing.
- Run a smoke-test mode (small subset + 1 epoch) to validate wiring after changes.
- Update docs/CHANGELOG_PIPELINE.md for any behavior change.
