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


STEPS_REGISTRY: dict[int, dict[str, Any]] = {
    1: {
        "name": "extract_chembl36_sqlite",
        "script": "scripts/extract_chembl36_sqlite.py",
        "required_inputs": ["paths.chembl_sqlite"],
        "default_output_path": "{paths.outputs_root}/step1",
        "build_command": _default_builder("extract_chembl36_sqlite.py"),
    },
    2: {
        "name": "qsar_postprocess",
        "script": "scripts/qsar_postprocess.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step2",
        "build_command": _default_builder("qsar_postprocess.py"),
    },
    3: {
        "name": "assemble_environments",
        "script": "scripts/assemble_environments.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step3",
        "build_command": _default_builder("assemble_environments.py"),
    },
    4: {
        "name": "generate_splits",
        "script": "scripts/generate_splits.py",
        "required_inputs": ["paths.outputs_root", "training.split_default"],
        "default_output_path": "{paths.outputs_root}/step4",
        "build_command": _default_builder("generate_splits.py"),
    },
    5: {
        "name": "run_benchmark",
        "script": "scripts/run_benchmark.py",
        "required_inputs": ["training.task", "training.label_col"],
        "default_output_path": "{paths.outputs_root}/step5",
        "build_command": _default_builder("run_benchmark.py"),
    },
    6: {
        "name": "reserved_step6",
        "script": None,
        "required_inputs": [],
        "default_output_path": "{paths.outputs_root}/step6",
        "build_command": lambda config, overrides: ["python", "-c", "print('Step 6 reserved/no-op')"],
    },
    7: {
        "name": "generate_counterfactuals",
        "script": "scripts/generate_counterfactuals.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step7",
        "build_command": _default_builder("generate_counterfactuals.py"),
    },
    8: {
        "name": "evaluate_model",
        "script": "scripts/evaluate_model.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step8",
        "build_command": _default_builder("evaluate_model.py"),
    },
    9: {
        "name": "evaluate_cross_endpoint",
        "script": "scripts/evaluate_cross_endpoint.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step9",
        "build_command": _default_builder("evaluate_cross_endpoint.py"),
    },
    10: {
        "name": "interpret_model",
        "script": "scripts/interpret_model.py",
        "required_inputs": ["paths.outputs_root", "style.font"],
        "default_output_path": "{paths.outputs_root}/step10",
        "build_command": _default_builder("interpret_model.py", include_style=True),
    },
    11: {
        "name": "evaluate_robustness",
        "script": "scripts/evaluate_robustness.py",
        "required_inputs": ["robustness.ensemble_size"],
        "default_output_path": "{paths.outputs_root}/step11",
        "build_command": _default_builder("evaluate_robustness.py", include_style=True),
    },
    12: {
        "name": "screen_library",
        "script": "scripts/screen_library.py",
        "required_inputs": ["screening.input_format", "screening.smiles_col_name"],
        "default_output_path": "{paths.outputs_root}/step12",
        "build_command": _default_builder("screen_library.py", include_style=True),
    },
    13: {
        "name": "analyze_screening",
        "script": "scripts/analyze_screening.py",
        "required_inputs": ["paths.outputs_root", "screening.topk"],
        "default_output_path": "{paths.outputs_root}/step13",
        "build_command": _default_builder("analyze_screening.py", include_style=True),
    },
    14: {
        "name": "match_screening_features",
        "script": "scripts/match_screening_features.py",
        "required_inputs": ["paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step14",
        "build_command": _default_builder("match_screening_features.py", include_style=True),
    },
    15: {
        "name": "build_manuscript_pack",
        "script": "scripts/build_manuscript_pack.py",
        "required_inputs": ["paper_id", "paths.outputs_root"],
        "default_output_path": "{paths.outputs_root}/step15",
        "build_command": _default_builder("build_manuscript_pack.py", include_style=True),
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
