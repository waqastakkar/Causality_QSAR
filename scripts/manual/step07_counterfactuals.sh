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
target = str(cfg['target'])
screening = cfg.get('screening', {}) if isinstance(cfg.get('screening'), dict) else {}
runs6 = out_root / 'step6' / target
runs5 = out_root / 'step5' / target
for root in [runs6, runs5]:
    cands = sorted(root.glob('**/checkpoints/best.pt')) if root.exists() else []
    if cands:
        run_dir = cands[0].parent.parent
        break
else:
    run_dir = runs6
print(str(out_root)); print(target); print(str(run_dir));
print('' if screening.get('cns_mpo_threshold') is None else str(screening.get('cns_mpo_threshold')))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; RUN_DIR="${CFG[2]}"; CNS_MPO="${CFG[3]}"
STEP_OUT="$OUTPUTS_ROOT/step7"
LOG_FILE="$STEP_OUT/step07_counterfactuals.log"
mkdir -p "$STEP_OUT/rules" "$STEP_OUT/candidates"
RULES_PARQUET="$STEP_OUT/rules/mmp_rules.parquet"
DATASET_PARQUET="$OUTPUTS_ROOT/step3/multienv_compound_level.parquet"

if [[ ! -f "$RULES_PARQUET" ]]; then
  BUILD_CMD=("$PYTHON_BIN" "scripts/build_mmp_rules.py" "--target" "$TARGET" "--input_parquet" "$DATASET_PARQUET" "--outdir" "$STEP_OUT")
  BUILD_CMD+=("${STYLE_FLAGS[@]}")
  manual_run_with_log "$LOG_FILE" "${BUILD_CMD[@]}"
fi

CMD=("$PYTHON_BIN" "scripts/generate_counterfactuals.py" "--target" "$TARGET" "--run_dir" "$RUN_DIR" "--dataset_parquet" "$DATASET_PARQUET" "--mmp_rules_parquet" "$RULES_PARQUET" "--outdir" "$STEP_OUT")
if [[ -f "$OUTPUTS_ROOT/step3/bbb_annotations.parquet" ]]; then CMD+=("--bbb_parquet" "$OUTPUTS_ROOT/step3/bbb_annotations.parquet"); fi
if [[ -n "$CNS_MPO" ]]; then CMD+=("--cns_mpo_threshold" "$CNS_MPO"); fi
CMD+=("${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"

for output in generated_counterfactuals.parquet filtered_counterfactuals.parquet ranked_topk.parquet; do
  if [[ ! -f "$STEP_OUT/candidates/$output" ]]; then
    echo "[step7] ERROR: missing expected output file: $STEP_OUT/candidates/$output" >&2
    exit 1
  fi
done
