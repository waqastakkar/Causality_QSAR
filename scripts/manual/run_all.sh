#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <config.yaml> [step-range] [overrides...]" >&2
  exit 1
fi
CONFIG="$1"; shift
STEP_RANGE="${STEPS:-${1:-1-15}}"
if [[ $# -gt 0 && "$1" != *=* && "$1" != --* ]]; then
  shift
fi
EXTRA_ARGS=("$@")
shopt -s nullglob

in_range() {
  local step="$1" token a b
  IFS=',' read -ra tokens <<< "$STEP_RANGE"
  for token in "${tokens[@]}"; do
    token="${token// /}"
    [[ -z "$token" ]] && continue
    if [[ "$token" == *-* ]]; then
      a="${token%-*}"; b="${token#*-}"
      if (( step >= a && step <= b )); then return 0; fi
    else
      if (( step == token )); then return 0; fi
    fi
  done
  return 1
}

for step in $(seq -w 1 15); do
  n=$((10#$step))
  if in_range "$n"; then
    script=$(printf 'scripts/manual/step%02d_' "$n")
    match=( ${script}*.sh )
    if [[ ${#match[@]} -ne 1 ]]; then
      echo "Could not uniquely resolve step script for step $n" >&2
      exit 1
    fi
    bash "${match[0]}" "$CONFIG" "${EXTRA_ARGS[@]}"
  fi
done
