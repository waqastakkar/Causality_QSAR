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
print(str(out_root)); print(target);
print('' if screening.get('cns_mpo_threshold') is None else str(screening.get('cns_mpo_threshold')))
PY
)
OUTPUTS_ROOT="${CFG[0]}"; TARGET="${CFG[1]}"; CNS_MPO="${CFG[2]}"
RUN_DIR="$(manual_read_run_pointer "$PYTHON_BIN" "$OUTPUTS_ROOT/step6/run_pointer.json")"
[[ -n "$RUN_DIR" ]] || RUN_DIR="$(manual_read_run_pointer "$PYTHON_BIN" "$OUTPUTS_ROOT/step5/run_pointer.json")"
[[ -n "$RUN_DIR" ]] || manual_fail_preflight "missing run pointer (expected outputs/step6/run_pointer.json or outputs/step5/run_pointer.json)"
STEP_OUT="$OUTPUTS_ROOT/step7"
LOG_FILE="$STEP_OUT/step07_counterfactuals.log"
mkdir -p "$STEP_OUT/rules" "$STEP_OUT/candidates"
RULES_PARQUET="$STEP_OUT/rules/mmp_rules.parquet"
DATASET_PARQUET="$OUTPUTS_ROOT/step3/multienv_compound_level.parquet"
manual_require_file "$DATASET_PARQUET" "run step03 first"
manual_require_file "$RUN_DIR/checkpoints/best.pt" "run training first"

if [[ ! -f "$RULES_PARQUET" ]]; then
  BUILD_CMD=("$PYTHON_BIN" "scripts/build_mmp_rules.py" "--target" "$TARGET" "--input_parquet" "$DATASET_PARQUET" "--outdir" "$STEP_OUT")
  BUILD_CMD+=("${STYLE_FLAGS[@]}")
  manual_run_with_log "$LOG_FILE" "${BUILD_CMD[@]}"
fi

CMD=("$PYTHON_BIN" "scripts/generate_counterfactuals.py" "--target" "$TARGET" "--run_dir" "$RUN_DIR" "--dataset_parquet" "$DATASET_PARQUET" "--mmp_rules_parquet" "$RULES_PARQUET" "--outdir" "$STEP_OUT")
BBB_PARQUET="$OUTPUTS_ROOT/step3/data/bbb_annotations.parquet"; [[ -f "$BBB_PARQUET" ]] || BBB_PARQUET="$OUTPUTS_ROOT/step3/bbb_annotations.parquet"
if [[ -f "$BBB_PARQUET" ]]; then CMD+=("--bbb_parquet" "$BBB_PARQUET"); fi
if [[ -n "$CNS_MPO" ]]; then CMD+=("--cns_mpo_threshold" "$CNS_MPO"); fi
CMD+=("${STYLE_FLAGS[@]}")
manual_append_overrides EXTRA_ARGS CMD
manual_run_with_log "$LOG_FILE" "${CMD[@]}"

for output in generated_counterfactuals.parquet filtered_counterfactuals.parquet ranked_topk.parquet; do
  [[ -f "$STEP_OUT/candidates/$output" ]] || manual_fail_preflight "missing expected output file: $STEP_OUT/candidates/$output"
done
