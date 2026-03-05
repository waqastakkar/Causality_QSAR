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
print(str(out_root)); print(str(cfg['target']))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"
STEP_OUT="$OUTPUTS_ROOT/step9"; LOG_FILE="$STEP_OUT/step09_cross_endpoint.log"; mkdir -p "$STEP_OUT"

RUN_DIR="$(manual_get_override run_dir "${EXTRA_ARGS[@]}")"
if [[ -z "$RUN_DIR" ]]; then
  mapfile -t SPLITS < <(manual_resolve_splits_to_run "$PYTHON_BIN" "$CONFIG" "$OUTPUTS_ROOT/step4" "${EXTRA_ARGS[@]}")
  FIRST_SPLIT="${SPLITS[0]:-}"
  if [[ -n "$FIRST_SPLIT" ]]; then
    PTR="$OUTPUTS_ROOT/step6/$TARGET/$FIRST_SPLIT/latest_run.json"
    [[ -f "$PTR" ]] || PTR="$OUTPUTS_ROOT/step5/$TARGET/$FIRST_SPLIT/latest_run.json"
    [[ -f "$PTR" ]] && RUN_DIR="$(manual_read_run_pointer "$PYTHON_BIN" "$PTR")"
  fi
fi
if [[ -z "$RUN_DIR" ]]; then
  PTR="$OUTPUTS_ROOT/step6/$TARGET/latest_run.json"
  [[ -f "$PTR" ]] || PTR="$OUTPUTS_ROOT/step5/$TARGET/latest_run.json"
  [[ -f "$PTR" ]] && RUN_DIR="$(manual_read_run_pointer "$PYTHON_BIN" "$PTR")"
fi
[[ -n "$RUN_DIR" ]] || manual_fail_preflight "missing run pointer for cross-endpoint (step6/step5 latest_run.json)"

EXTERNAL_PARQUET="data/external/processed/ptp1b_inhibition_chembl335/data/inhibition_external_final.parquet"
manual_require_file "$EXTERNAL_PARQUET" "run step08a_prepare_external_inhibition.sh first"
manual_require_columns "$PYTHON_BIN" "$EXTERNAL_PARQUET" "smiles_canonical,y_inhib_active,standard_relation_norm"
manual_require_file "$RUN_DIR/checkpoints/best.pt"
CMD=("$PYTHON_BIN" "scripts/evaluate_cross_endpoint.py" "--target" "$TARGET" "--run_dir" "$RUN_DIR" "--external_parquet" "$EXTERNAL_PARQUET" "--outdir" "$STEP_OUT")
BBB_PARQUET="$OUTPUTS_ROOT/step3/data/bbb_annotations.parquet"; [[ -f "$BBB_PARQUET" ]] || BBB_PARQUET="$OUTPUTS_ROOT/step3/bbb_annotations.parquet"
if [[ -f "$BBB_PARQUET" ]]; then CMD+=("--bbb_parquet" "$BBB_PARQUET"); fi
CMD+=("${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"
