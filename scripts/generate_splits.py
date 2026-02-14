#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate step-4 data splits from pipeline config")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split_name", default=None, help="Optional split to generate (defaults to all make_splits defaults)")
    return parser.parse_args()


def _resolve_outputs_root(config_path: Path, cfg: dict) -> Path:
    outputs_root = Path(cfg.get("paths", {}).get("outputs_root", "outputs"))
    if not outputs_root.is_absolute():
        outputs_root = (config_path.parent / outputs_root).resolve()
    outputs_root.mkdir(parents=True, exist_ok=True)
    return outputs_root


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle) or {}

    outputs_root = _resolve_outputs_root(config_path, cfg)
    input_parquet = outputs_root / "step3" / "multienv_compound_level.parquet"
    outdir = outputs_root / "step4"

    if not input_parquet.exists():
        raise FileNotFoundError(
            f"Required input parquet not found: {input_parquet}. Run step 3 before step 4."
        )

    target = cfg.get("target")
    if not target:
        raise ValueError("Missing required `target` key in pipeline config.")

    cmd = [
        "python",
        str(Path("scripts") / "make_splits.py"),
        "--target",
        str(target),
        "--input_parquet",
        str(input_parquet),
        "--outdir",
        str(outdir),
        "--seed",
        str(args.seed),
    ]

    if args.split_name:
        cmd.extend(["--enable", args.split_name])

    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
