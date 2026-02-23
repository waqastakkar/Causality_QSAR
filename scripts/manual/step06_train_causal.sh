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
out_root = Path(cfg.get('paths', {}).get('outputs_root', 'outputs')).resolve()
training = cfg.get('training', {}) if isinstance(cfg.get('training'), dict) else {}
seeds = training.get('seeds') or []
print(str(out_root)); print(str(cfg['target'])); print(str(training.get('split_default', 'scaffold_bm')))
print(str(training.get('task', 'regression'))); print(str(training.get('label_col', 'pIC50'))); print(str(training.get('env_col', 'env_id_manual')))
print(str(training.get('epochs', 300))); print(str(training.get('early_stopping_patience', 30))); print(str(seeds[0]) if seeds else '')
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; SPLIT_NAME="${CFG[2]}"; TASK="${CFG[3]}"; LABEL_COL="${CFG[4]}"; ENV_COL="${CFG[5]}"; EPOCHS="${CFG[6]}"; PATIENCE="${CFG[7]}"; SEED1="${CFG[8]}"
STEP_OUT="$OUTPUTS_ROOT/step6"
LOG_FILE="$STEP_OUT/step06_train_causal.log"
mkdir -p "$STEP_OUT"
CMD=("$PYTHON_BIN" "scripts/train_causal_qsar.py" "--target" "$TARGET" "--dataset_parquet" "$OUTPUTS_ROOT/step3/multienv_compound_level.parquet" "--splits_dir" "$OUTPUTS_ROOT/step4" "--split_name" "$SPLIT_NAME" "--outdir" "$STEP_OUT" "--task" "$TASK" "--label_col" "$LABEL_COL" "--env_col" "$ENV_COL" "--epochs" "$EPOCHS" "--early_stopping_patience" "$PATIENCE")
if [[ -n "$SEED1" ]]; then CMD+=("--seed" "$SEED1"); fi
if [[ -f "$OUTPUTS_ROOT/step3/bbb_annotations.parquet" ]]; then CMD+=("--bbb_parquet" "$OUTPUTS_ROOT/step3/bbb_annotations.parquet"); fi
CMD+=("${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"
