#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"; manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"; manual_style_flags "$CONFIG"
OUTPUTS_ROOT="$($PYTHON_BIN - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
print(Path(cfg.get('paths',{}).get('outputs_root','outputs')).resolve())
PY
)"
STEP_OUT="$OUTPUTS_ROOT/step11"; LOG_FILE="$STEP_OUT/step11_robustness.log"; mkdir -p "$STEP_OUT"
CMD=("$PYTHON_BIN" "scripts/evaluate_robustness.py" "--config" "$CONFIG" "${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD; manual_run_with_log "$LOG_FILE" "${CMD[@]}"
