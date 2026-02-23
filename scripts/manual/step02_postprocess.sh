#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"
manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"
readarray -t CFG < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
out_root = Path(cfg.get('paths', {}).get('outputs_root', 'outputs')).resolve()
target = str(cfg['target'])
print(str(out_root))
print(target)
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"
STEP_OUT="$OUTPUTS_ROOT/step2"
INPUT_CSV="$OUTPUTS_ROOT/step1/${TARGET}_qsar_ready.csv"
LOG_FILE="$STEP_OUT/step02_postprocess.log"
mkdir -p "$STEP_OUT"
CMD=("$PYTHON_BIN" "scripts/qsar_postprocess.py" "--config" "$CONFIG" "--input" "$INPUT_CSV" "--outdir" "$STEP_OUT")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"
