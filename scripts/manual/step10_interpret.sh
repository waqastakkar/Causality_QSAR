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
out_root = Path(cfg.get('paths', {}).get('outputs_root', 'outputs')).resolve(); print(str(out_root)); print(str(cfg['target']))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"
STEP_OUT="$OUTPUTS_ROOT/step10"; LOG_FILE="$STEP_OUT/step10_interpret.log"; mkdir -p "$STEP_OUT"
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
DATASET_PARQUET="$OUTPUTS_ROOT/step3/multienv_compound_level.parquet"
if [[ -z "$RUN_DIR" || ! -f "$DATASET_PARQUET" ]]; then
  reason=()
  [[ -z "$RUN_DIR" ]] && reason+=("missing trained run checkpoint under outputs/step5")
  [[ ! -f "$DATASET_PARQUET" ]] && reason+=("missing multienv dataset parquet at outputs/step3")
  printf 'skipped step 10: %s\n' "$(IFS=', '; echo "${reason[*]}")" > "$STEP_OUT/step10_noop.txt"
  echo "Step 10 skipped: $(IFS=', '; echo "${reason[*]}")" | tee "$LOG_FILE"
  exit 0
fi
CMD=("$PYTHON_BIN" "scripts/interpret_model.py" "--target" "$TARGET" "--run_dir" "$RUN_DIR" "--dataset_parquet" "$DATASET_PARQUET" "--outdir" "$STEP_OUT")
BBB_PARQUET="$OUTPUTS_ROOT/step3/data/bbb_annotations.parquet"; [[ -f "$BBB_PARQUET" ]] || BBB_PARQUET="$OUTPUTS_ROOT/step3/bbb_annotations.parquet"
if [[ -f "$BBB_PARQUET" ]]; then CMD+=("--bbb_parquet" "$BBB_PARQUET"); fi
if [[ -f "$OUTPUTS_ROOT/step7/candidates/ranked_topk.parquet" ]]; then CMD+=("--counterfactuals_parquet" "$OUTPUTS_ROOT/step7/candidates/ranked_topk.parquet"); fi
CMD+=("${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"
