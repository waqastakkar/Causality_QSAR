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
RUN_DIR="$($PYTHON_BIN - <<PY
from pathlib import Path
for root in [Path('$OUTPUTS_ROOT/step6/$TARGET'), Path('$OUTPUTS_ROOT/step5/$TARGET')]:
    cands = sorted(root.glob('**/checkpoints/best.pt')) if root.exists() else []
    if cands:
        print(cands[0].parent.parent)
        raise SystemExit(0)
print('')
PY
)"
EXTERNAL_PARQUET="$($PYTHON_BIN - <<PY
from pathlib import Path
target = '$TARGET'.lower()
cands = [
    Path('data/external/processed') / f"{target}_inhibition" / 'data' / 'inhibition_external_final.parquet',
    Path('data/external/processed') / 'ptp1b_inhibition_chembl335' / 'data' / 'inhibition_external_final.parquet',
]
for p in cands:
    if p.exists():
        print(p)
        break
PY
)"
if [[ -z "$RUN_DIR" || -z "$EXTERNAL_PARQUET" ]]; then
  reason=()
  [[ -z "$RUN_DIR" ]] && reason+=("missing trained run checkpoint under outputs/step5")
  [[ -z "$EXTERNAL_PARQUET" ]] && reason+=("missing external inhibition parquet")
  printf 'skipped step 9: %s\n' "$(IFS=', '; echo "${reason[*]}")" > "$STEP_OUT/step9_noop.txt"
  echo "Step 9 skipped: $(IFS=', '; echo "${reason[*]}")" | tee "$LOG_FILE"
  exit 0
fi
CMD=("$PYTHON_BIN" "scripts/evaluate_cross_endpoint.py" "--target" "$TARGET" "--run_dir" "$RUN_DIR" "--external_parquet" "$EXTERNAL_PARQUET" "--outdir" "$STEP_OUT")
if [[ -f "$OUTPUTS_ROOT/step3/bbb_annotations.parquet" ]]; then CMD+=("--bbb_parquet" "$OUTPUTS_ROOT/step3/bbb_annotations.parquet"); fi
CMD+=("${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"
