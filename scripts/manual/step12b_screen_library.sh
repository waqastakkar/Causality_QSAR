#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/_helpers.sh"; manual_parse_common "$@"
PYTHON_BIN="$(manual_python_for_config "$CONFIG")"; manual_style_flags "$CONFIG"
readarray -t CFG < <("$PYTHON_BIN" - "$CONFIG" <<'PY'
import sys, yaml
from pathlib import Path
cfg=yaml.safe_load(Path(sys.argv[1]).read_text(encoding='utf-8')) or {}
out=Path(cfg.get('paths',{}).get('outputs_root','outputs')).resolve()
s=cfg.get('screening',{}) if isinstance(cfg.get('screening'),dict) else {}
print(out)
print(cfg['target'])
print(s.get('cns_mpo_threshold',4.0))
print(s.get('topk',500))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; CNS_MPO="${CFG[2]}"; TOPK="${CFG[3]}"
STEP_OUT="$OUTPUTS_ROOT/step12"; SCREEN_ROOT="$STEP_OUT/screening"; LOG_FILE="$STEP_OUT/step12b_screen_library.log"; mkdir -p "$STEP_OUT"
RUN_DIR="$(manual_get_override run_dir "${EXTRA_ARGS[@]}")"
[[ -n "$RUN_DIR" ]] || RUN_DIR="$(manual_read_run_pointer "$PYTHON_BIN" "$OUTPUTS_ROOT/step6/run_pointer.json")"
[[ -n "$RUN_DIR" ]] || RUN_DIR="$(manual_read_run_pointer "$PYTHON_BIN" "$OUTPUTS_ROOT/step5/run_pointer.json")"
[[ -n "$RUN_DIR" ]] || manual_fail_preflight "missing run pointer for screening"
manual_require_file "$RUN_DIR/artifacts/feature_schema.json" "required for featurization"
PREP_DIR="$(manual_read_run_pointer "$PYTHON_BIN" "$STEP_OUT/latest_prepare.json")"
[[ -n "$PREP_DIR" ]] || manual_fail_preflight "missing prepared-library pointer: $STEP_OUT/latest_prepare.json (run step12a first)"
PREP_DEDUP="$PREP_DIR/processed/library_dedup.parquet"
manual_require_file "$PREP_DEDUP" "run step12a first"
SCREEN_ID="$(date -u +%Y%m%dT%H%M%SZ)"
CMD=("$PYTHON_BIN" "scripts/screen_library.py" "--target" "$TARGET" "--screen_id" "$SCREEN_ID" "--run_dir" "$RUN_DIR" "--prepared_library_path" "$PREP_DEDUP" "--outdir" "$SCREEN_ROOT" "--cns_mpo_threshold" "$CNS_MPO" "--topk" "$TOPK" "${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD; manual_run_with_log "$LOG_FILE" "${CMD[@]}"
SCREEN_DIR="$SCREEN_ROOT/$TARGET/$SCREEN_ID"
[[ -d "$SCREEN_DIR" ]] || manual_fail_preflight "missing screening output directory: $SCREEN_DIR"
manual_require_file "$SCREEN_DIR/predictions/scored_with_uncertainty.parquet" "step12b output missing"
manual_write_run_pointer "$PYTHON_BIN" "$STEP_OUT/latest_screen.json" "$SCREEN_DIR" "step12b_screen_library"
# back-compat pointer for downstream steps expecting step12 root pointer
manual_write_run_pointer "$PYTHON_BIN" "$STEP_OUT/run_pointer.json" "$SCREEN_DIR" "step12b_screen_library"
