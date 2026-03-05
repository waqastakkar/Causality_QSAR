#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import yaml


def cols_ok(path: Path, required: list[str]) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing file: {path}"
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    return (not missing, "ok" if not missing else f"missing columns: {missing}")


def pointer_ok(path: Path) -> tuple[bool, str, str | None]:
    if not path.exists():
        return False, f"missing file: {path}", None
    data = json.loads(path.read_text(encoding="utf-8"))
    run_dir = data.get("run_dir")
    if not run_dir:
        return False, f"pointer missing run_dir: {path}", None
    return True, "ok", str(run_dir)


def step2_contract_ok(row_path: Path, max_value_nm: float) -> tuple[bool, str]:
    if not row_path.exists():
        return False, f"missing file: {row_path}"
    df = pd.read_csv(row_path)
    required = ["standard_type", "standard_relation", "standard_value", "pIC50"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        return False, f"missing Step2 columns: {missing}"

    if sorted(df["standard_type"].astype(str).str.upper().unique().tolist()) != ["IC50"]:
        return False, "Step2 primary dataset contains non-IC50 rows"
    if sorted(df["standard_relation"].astype(str).unique().tolist()) != ["="]:
        return False, "Step2 primary dataset contains non '=' relations"
    if (pd.to_numeric(df["standard_value"], errors="coerce") > max_value_nm).any():
        return False, f"Step2 primary dataset contains standard_value > {max_value_nm} nM"
    pic = pd.to_numeric(df["pIC50"], errors="coerce")
    if pic.isna().any():
        return False, "Step2 primary dataset contains NaN pIC50"
    if ((pic < 0) | (pic > 14)).any():
        return False, "Step2 primary dataset has pIC50 outside expected [0, 14]"
    return True, "ok"


def print_check(name: str, ok: bool, detail: str) -> int:
    print(f"{name}\t{'PASS' if ok else 'FAIL'}\t{detail}")
    return 0 if ok else 1


def main() -> int:
    if len(sys.argv) != 2:
        print("Usage: python scripts/pipeline_doctor.py <config.yaml>")
        return 2

    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
    out = Path(cfg.get("paths", {}).get("outputs_root", "outputs")).resolve()
    target = cfg["target"]
    split_default = cfg.get("training", {}).get("split_default", "scaffold_bm")
    split_manifest = out / "step4" / "splits_manifest.json"
    external_parquet = Path("data/external/processed/ptp1b_inhibition_chembl335/data/inhibition_external_final.parquet")
    max_value_nm = float(cfg.get("postprocess", {}).get("max_value_nM", 1e9))

    failed = 0
    print("check\tstatus\tdetail")
    failed += print_check("step01_output", *cols_ok(out / "step1" / f"{target}_qsar_ready.csv", ["canonical_smiles", "standard_type", "standard_value"]))
    failed += print_check("step02_output", *cols_ok(out / "step2" / "compound_level_with_properties.csv", ["canonical_smiles", "pIC50"]))
    failed += print_check("step02_primary_contract", *step2_contract_ok(out / "step2" / "row_level_primary.csv", max_value_nm))
    failed += print_check("step03_output", *cols_ok(out / "step3" / "multienv_compound_level.parquet", ["molecule_id", "smiles", "pIC50"]))

    if split_manifest.exists():
        sm = json.loads(split_manifest.read_text(encoding="utf-8"))
        splits = sm.get("split_names", [])
        failed += print_check("step04_manifest", bool(splits), f"splits={splits}")
    else:
        splits = [split_default] if (out / "step4" / split_default).exists() else []
        failed += print_check("step04_manifest", False, f"missing file: {split_manifest}")

    for split in (splits[:3] if splits else [split_default]):
        failed += print_check(f"step04_{split}_train_ids", *cols_ok(out / "step4" / split / "train_ids.csv", ["molecule_id"]))

    target_latest = out / "step6" / target / "latest_run.json"
    ok, detail, run_dir = pointer_ok(target_latest)
    failed += print_check("step06_target_pointer", ok, detail)
    if run_dir:
        run_path = Path(run_dir)
        failed += print_check("run_checkpoint", (run_path / "checkpoints" / "best.pt").exists(), str(run_path / "checkpoints" / "best.pt"))
        failed += print_check("run_feature_schema", (run_path / "artifacts" / "feature_schema.json").exists(), str(run_path / "artifacts" / "feature_schema.json"))

    failed += print_check("step08a_external", *cols_ok(external_parquet, ["smiles_canonical", "y_inhib_active"]))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
