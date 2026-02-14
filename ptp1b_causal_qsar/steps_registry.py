from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

StepBuilder = Callable[[dict[str, Any], dict[str, Any]], list[str]]

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


def _style_flags(config: dict[str, Any]) -> list[str]:
    style = config.get("style", {})
    flags: list[str] = []
    for key in STYLE_KEYS:
        if key in style and style[key] is not None:
            flags.extend([f"--{key}", str(style[key])])
    return flags


def _default_builder(script_name: str, include_style: bool = False) -> StepBuilder:
    def _build(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
        cmd = ["python", str(Path("scripts") / script_name)]
        cmd.extend(["--config", config["_config_path"]])
        if include_style:
            cmd.extend(_style_flags(config))
        for key, value in overrides.items():
            cmd.extend([f"--{key}", str(value)])
        return cmd

    return _build


def _step1_extract_builder(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    cmd = ["python", str(Path("scripts") / "extract_chembl36_sqlite.py")]
    cmd.extend(["--config", config["_config_path"]])
    cmd.extend(["--db", str(config["paths"]["chembl_sqlite"])])
    cmd.extend(["--target", str(config["target"])])
    cmd.extend(["--outdir", str(Path(config["paths"]["outputs_root"]) / "step1")])
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _step2_postprocess_builder(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    out_root = Path(config["paths"]["outputs_root"])
    step1_out = out_root / "step1" / f"{config['target']}_qsar_ready.csv"
    cmd = ["python", str(Path("scripts") / "qsar_postprocess.py")]
    cmd.extend(["--config", config["_config_path"]])
    cmd.extend(["--input", str(step1_out)])
    cmd.extend(["--outdir", str(out_root / "step2")])
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


def _step3_assemble_environments_builder(config: dict[str, Any], overrides: dict[str, Any]) -> list[str]:
    out_root = Path(config["paths"]["outputs_root"])
    step2_out = out_root / "step2"
    row_level_csv = step2_out / "row_level_with_pIC50.csv"
    compound_level_csv = step2_out / "compound_level_with_properties.csv"
    raw_extract_csv = out_root / "step1" / f"{config['target']}_qsar_ready.csv"

    env_cfg = config.get("environments", {})
    bbb_rules = env_cfg.get("bbb_rules", str(Path("configs") / "bbb_rules.yaml"))
    series_rules = env_cfg.get("series_rules")
    env_keys = env_cfg.get("env_keys")

    cmd = ["python", str(Path("scripts") / "assemble_environments.py")]
    cmd.extend(["--target", str(config["target"])])
    cmd.extend(["--row_level_csv", str(row_level_csv)])
    cmd.extend(["--compound_level_csv", str(compound_level_csv)])
    cmd.extend(["--raw_extract_csv", str(raw_extract_csv)])
    cmd.extend(["--outdir", str(out_root / "step3")])
    cmd.extend(["--bbb_rules", str(bbb_rules)])
    if series_rules:
        cmd.extend(["--series_rules", str(series_rules)])
    if env_keys:
        cmd.extend(["--env_keys", *[str(k) for k in env_keys]])
    for key, value in overrides.items():
        cmd.extend([f"--{key}", str(value)])
    return cmd


STEPS_REGISTRY: dict[int, dict[str, Any]] = {
    1: {
        "name": "extract_chembl36_sqlite",
        "script": "scripts/extract_chembl36_sqlite.py",
        "required_inputs": ["paths.chembl_sqlite"],
        "default_output_path": "{paths.outputs_root}/step1",
        "build_command": _step1_extract_builder,
        "depends_on": [],
    },
    2: {
        "name": "qsar_postprocess",
        "script": "scripts/qsar_postprocess.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step2",
        "build_command": _step2_postprocess_builder,
        "depends_on": [1],
    },
    3: {
        "name": "assemble_environments",
        "script": "scripts/assemble_environments.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step3",
        "build_command": _step3_assemble_environments_builder,
        "depends_on": [2],
    },
    4: {
        "name": "generate_splits",
        "script": "scripts/generate_splits.py",
        "required_inputs": ["paths.outputs_root", "training.split_default"],
        "default_output_path": "{paths.outputs_root}/step4",
        "build_command": _default_builder("generate_splits.py"),
        "depends_on": [3],
    },
    5: {
        "name": "run_benchmark",
        "script": "scripts/run_benchmark.py",
        "required_inputs": ["training.task", "training.label_col"],
        "default_output_path": "{paths.outputs_root}/step5",
        "build_command": _default_builder("run_benchmark.py"),
        "depends_on": [4],
    },
    6: {
        "name": "reserved_step6",
        "script": None,
        "required_inputs": [],
        "default_output_path": "{paths.outputs_root}/step6",
        "build_command": lambda config, overrides: ["python", "-c", "print('Step 6 reserved/no-op')"],
        "depends_on": [5],
    },
    7: {
        "name": "generate_counterfactuals",
        "script": "scripts/generate_counterfactuals.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step7",
        "build_command": _default_builder("generate_counterfactuals.py"),
        "depends_on": [6],
    },
    8: {
        "name": "evaluate_model",
        "script": "scripts/evaluate_model.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step8",
        "build_command": _default_builder("evaluate_model.py"),
        "depends_on": [7],
    },
    9: {
        "name": "evaluate_cross_endpoint",
        "script": "scripts/evaluate_cross_endpoint.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step9",
        "build_command": _default_builder("evaluate_cross_endpoint.py"),
        "depends_on": [8],
    },
    10: {
        "name": "interpret_model",
        "script": "scripts/interpret_model.py",
        "required_inputs": ["paths.outputs_root", "style.font"],
        "default_output_path": "{paths.outputs_root}/step10",
        "build_command": _default_builder("interpret_model.py", include_style=True),
        "depends_on": [9],
    },
    11: {
        "name": "evaluate_robustness",
        "script": "scripts/evaluate_robustness.py",
        "required_inputs": ["robustness.ensemble_size"],
        "default_output_path": "{paths.outputs_root}/step11",
        "build_command": _default_builder("evaluate_robustness.py", include_style=True),
        "depends_on": [10],
    },
    12: {
        "name": "screen_library",
        "script": "scripts/screen_library.py",
        "required_inputs": ["screening.input_format", "screening.smiles_col_name"],
        "default_output_path": "{paths.outputs_root}/step12",
        "build_command": _default_builder("screen_library.py", include_style=True),
        "depends_on": [11],
    },
    13: {
        "name": "analyze_screening",
        "script": "scripts/analyze_screening.py",
        "required_inputs": ["paths.outputs_root", "screening.topk"],
        "default_output_path": "{paths.outputs_root}/step13",
        "build_command": _default_builder("analyze_screening.py", include_style=True),
        "depends_on": [12],
    },
    14: {
        "name": "match_screening_features",
        "script": "scripts/match_screening_features.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step14",
        "build_command": _default_builder("match_screening_features.py", include_style=True),
        "depends_on": [13],
    },
    15: {
        "name": "build_manuscript_pack",
        "script": "scripts/build_manuscript_pack.py",
        "required_inputs": ["paper_id", "paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step15",
        "build_command": _default_builder("build_manuscript_pack.py", include_style=True),
        "depends_on": [14],
    },
}


def parse_step_range(spec: str, valid_steps: set[int] | None = None) -> list[int]:
    parsed: set[int] = set()
    for chunk in [part.strip() for part in spec.split(",") if part.strip()]:
        if "-" in chunk:
            start_s, end_s = chunk.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"Invalid range {chunk}: start>end")
            parsed.update(range(start, end + 1))
        else:
            parsed.add(int(chunk))
    steps = sorted(parsed)
    if valid_steps is not None:
        bad = [step for step in steps if step != 0 and step not in valid_steps]
        if bad:
            raise ValueError(f"Unregistered steps requested: {bad}")
    return steps


def get_nested(config: dict[str, Any], dotted_key: str) -> Any:
    cursor: Any = config
    for part in dotted_key.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            return None
        cursor = cursor[part]
    return cursor
