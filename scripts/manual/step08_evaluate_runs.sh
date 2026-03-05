#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"
manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"
manual_style_flags "$CONFIG"
readarray -t CFG < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
out_root = Path(cfg.get('paths', {}).get('outputs_root', 'outputs')).resolve(); training = cfg.get('training', {}) if isinstance(cfg.get('training'), dict) else {}
print(str(out_root)); print(str(cfg['target'])); print(str(training.get('task', 'regression'))); print(str(training.get('label_col', 'pIC50'))); print(str(training.get('env_col', 'env_id_manual')))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; TASK="${CFG[2]}"; LABEL_COL="${CFG[3]}"; ENV_COL="${CFG[4]}"
STEP_OUT="$OUTPUTS_ROOT/step8"; LOG_FILE="$STEP_OUT/step08_evaluate_runs.log"; mkdir -p "$STEP_OUT"

EXPLICIT_RUN_DIR="$(manual_get_override run_dir "${EXTRA_ARGS[@]}")"
EXPLICIT_RUNS_ROOT="$(manual_get_override runs_root "${EXTRA_ARGS[@]}")"
if [[ -n "$EXPLICIT_RUNS_ROOT" ]]; then
  RUNS_ROOT="$EXPLICIT_RUNS_ROOT"
elif [[ -n "$EXPLICIT_RUN_DIR" ]]; then
  RUNS_ROOT="$EXPLICIT_RUN_DIR"
else
  RUNS_ROOT=""
  PTR="$OUTPUTS_ROOT/step6/$TARGET/latest_run.json"
  [[ -f "$PTR" ]] || PTR="$OUTPUTS_ROOT/step5/$TARGET/latest_run.json"
  if [[ -f "$PTR" ]]; then
    RUNS_ROOT="$(manual_read_run_pointer "$PYTHON_BIN" "$PTR")"
  fi
  [[ -n "$RUNS_ROOT" ]] || RUNS_ROOT="$OUTPUTS_ROOT/step6/$TARGET"
  [[ -d "$RUNS_ROOT" ]] || RUNS_ROOT="$OUTPUTS_ROOT/step5/$TARGET"
fi

manual_require_dir "$RUNS_ROOT" "missing runs_root (step06 or step05 outputs)"
manual_require_file "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet"
manual_require_dir "$OUTPUTS_ROOT/step4"
CMD=("$PYTHON_BIN" "scripts/evaluate_runs.py" "--target" "$TARGET" "--runs_root" "$RUNS_ROOT" "--splits_dir" "$OUTPUTS_ROOT/step4" "--dataset_parquet" "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet" "--outdir" "$STEP_OUT" "--task" "$TASK" "--label_col" "$LABEL_COL" "--env_col" "$ENV_COL")
BBB_PARQUET="$OUTPUTS_ROOT/step3/data/bbb_annotations.parquet"; [[ -f "$BBB_PARQUET" ]] || BBB_PARQUET="$OUTPUTS_ROOT/step3/bbb_annotations.parquet"
if [[ -f "$BBB_PARQUET" ]]; then CMD+=("--bbb_parquet" "$BBB_PARQUET"); fi
CMD+=("${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"
