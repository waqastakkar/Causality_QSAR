from __future__ import annotations

import subprocess
import traceback
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from ptp1b_causal_qsar.config import config_sha256
from ptp1b_causal_qsar.steps_registry import STEPS_REGISTRY
from ptp1b_causal_qsar.utils.logging import dump_json
from ptp1b_causal_qsar.utils.provenance import collect_provenance, write_environment_txt


@dataclass
class RunnerResult:
    run_dir: Path
    steps: list[dict[str, Any]]
    errors: list[dict[str, Any]]


def create_pipeline_run_dir(outputs_root: str | Path) -> Path:
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = f"{ts}"
    run_dir = Path(outputs_root) / "pipeline_runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def execute_steps(
    *,
    config: dict[str, Any],
    steps: list[int],
    run_dir: Path,
    logger: Any,
    continue_on_error: bool = False,
    dry_run: bool = False,
    overrides: dict[str, Any] | None = None,
) -> RunnerResult:
    overrides = overrides or {}
    step_records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []

    resolved_config_path = run_dir / "pipeline_config_resolved.yaml"
    resolved_config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    for step in steps:
        if step == 0:
            logger.info("Step 0 selected (CLI), no script execution required.")
            continue

        meta = STEPS_REGISTRY[step]
        command = meta["build_command"](config, overrides)
        start = datetime.now(timezone.utc)
        logger.info("Running step %s (%s): %s", step, meta["name"], " ".join(command))

        result_code = 0
        if not dry_run:
            proc = subprocess.run(command, capture_output=True, text=True)
            logger.info(proc.stdout)
            if proc.stderr:
                logger.error(proc.stderr)
            result_code = proc.returncode

        end = datetime.now(timezone.utc)
        record = {
            "step_number": step,
            "name": meta["name"],
            "cmd": command,
            "start_time": start.isoformat(),
            "end_time": end.isoformat(),
            "return_code": result_code,
            "output_paths": [meta.get("default_output_path")],
        }
        step_records.append(record)

        if result_code != 0:
            err = {
                "step_number": step,
                "name": meta["name"],
                "command": command,
                "traceback": traceback.format_exc(),
            }
            errors.append(err)
            if not continue_on_error:
                break

    dump_json(run_dir / "pipeline_steps_executed.json", step_records)
    if errors:
        dump_json(run_dir / "pipeline_errors.json", errors)

    prov = collect_provenance(
        config_sha=config_sha256(config),
        config=config,
        executed_commands=step_records,
    )
    dump_json(run_dir / "provenance.json", prov)
    write_environment_txt(run_dir / "environment.txt")

    return RunnerResult(run_dir=run_dir, steps=step_records, errors=errors)
