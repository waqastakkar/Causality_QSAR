#!/usr/bin/env bash

manual_parse_common() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <config.yaml> [overrides...]" >&2
    return 1
  fi
  CONFIG="$1"
  shift
  EXTRA_ARGS=("$@")
  if [[ ! -f "$CONFIG" ]]; then
    echo "Config not found: $CONFIG" >&2
    return 1
  fi
}

manual_python_for_config() {
  local config_path="$1"
  local bootstrap_py="${PIPELINE_PYTHON:-python}"
  "$bootstrap_py" - "$config_path" <<'PY'
import os
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML is required to parse config: {exc}")

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}

runtime = cfg.get("runtime", {}) if isinstance(cfg.get("runtime"), dict) else {}
runtime_python = runtime.get("python")
if runtime_python:
    print(runtime_python)
else:
    print(os.environ.get("PIPELINE_PYTHON") or "python")
PY
}

manual_style_flags() {
  local config_path="$1"
  local bootstrap_py="${PIPELINE_PYTHON:-python}"
  mapfile -t STYLE_FLAGS < <("$bootstrap_py" - "$config_path" <<'PY'
import sys
from pathlib import Path

try:
    import yaml
except Exception as exc:
    raise SystemExit(f"PyYAML is required to parse config: {exc}")

STYLE_KEYS = [
    "svg_only",
    "font",
    "bold_text",
    "palette",
    "font_title",
    "font_label",
    "font_tick",
    "font_legend",
]
STYLE_ARG_ALIASES = {"svg_only": "svg"}

config_path = Path(sys.argv[1])
with config_path.open("r", encoding="utf-8") as fh:
    cfg = yaml.safe_load(fh) or {}

style = cfg.get("style", {}) if isinstance(cfg.get("style"), dict) else {}
for key in STYLE_KEYS:
    if key not in style or style[key] is None:
        continue
    cli_key = STYLE_ARG_ALIASES.get(key, key)
    value = style[key]
    if isinstance(value, bool):
        if value:
            print(f"--{cli_key}")
        continue
    print(f"--{cli_key}")
    print(str(value))
PY
)
}

manual_append_overrides() {
  local -n _extra_ref=$1
  local -n _cmd_ref=$2
  local arg
  for arg in "${_extra_ref[@]}"; do
    if [[ "$arg" == *=* ]]; then
      _cmd_ref+=("--${arg%%=*}" "${arg#*=}")
    else
      _cmd_ref+=("$arg")
    fi
  done
}

manual_run_with_log() {
  local log_file="$1"
  shift
  mkdir -p "$(dirname "$log_file")"
  echo "[$(basename "$0")] Running command:"
  printf '  %q' "$@"
  printf '\n'
  "$@" 2>&1 | tee -a "$log_file"
}
