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
STEP_OUT="$OUTPUTS_ROOT/step8a"
LOG_FILE="$STEP_OUT/step08a_prepare_external_inhibition.log"
mkdir -p "$STEP_OUT"
OUTDIR="data/external/processed/ptp1b_inhibition_chembl335"
RAW_INHIB_CSV="data/external/raw/ptp1b_inhibition_chembl335.csv"
IC50_PARQUET="$OUTPUTS_ROOT/step3/multienv_compound_level.parquet"
manual_require_file "$RAW_INHIB_CSV" "expected raw inhibition export"
manual_require_file "$IC50_PARQUET" "run step03 first"
manual_require_dir "$OUTPUTS_ROOT/step4" "run step04 first"
mapfile -t SPLITS < <(manual_resolve_splits_to_run "$PYTHON_BIN" "$CONFIG" "$OUTPUTS_ROOT/step4" "${EXTRA_ARGS[@]}")
SPLIT_NAME="${SPLITS[0]:-}"
[[ -n "$SPLIT_NAME" ]] || manual_fail_preflight "no split resolved for external inhibition prep"
manual_require_dir "$OUTPUTS_ROOT/step4/$SPLIT_NAME" "run step04 first"
CMD=("$PYTHON_BIN" "scripts/prepare_inhibition_external.py" "--target" "$TARGET" "--input_csv" "$RAW_INHIB_CSV" "--ic50_parquet" "$IC50_PARQUET" "--splits_dir" "$OUTPUTS_ROOT/step4" "--split_name" "$SPLIT_NAME" "--outdir" "$OUTDIR")
CMD+=("${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"
manual_require_file "$OUTDIR/data/inhibition_external_final.parquet" "external prep output contract"
manual_require_columns "$PYTHON_BIN" "$OUTDIR/data/inhibition_external_final.parquet" "smiles_canonical,y_inhib_active"
