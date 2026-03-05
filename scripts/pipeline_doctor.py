#!/usr/bin/env python
from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    import pandas as pd
except Exception:
    pd = None
import yaml


def cols_ok(path: Path, required: list[str]) -> tuple[bool, str]:
    if not path.exists():
        return False, f"missing file: {path}"
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = [c for c in required if c not in df.columns]
    return (not missing, "ok" if not missing else f"missing columns: {missing}")


def main() -> int:
    if pd is None:
        print("pipeline_doctor requires pandas in the selected runtime")
        return 2
    if len(sys.argv) != 2:
        print("Usage: python scripts/pipeline_doctor.py <config.yaml>")
        return 2
    cfg = yaml.safe_load(Path(sys.argv[1]).read_text(encoding="utf-8")) or {}
    out = Path(cfg.get("paths", {}).get("outputs_root", "outputs")).resolve()
    split = cfg.get("training", {}).get("split_default", "scaffold_bm")
    checks = [
        ("step01_output", out / "step1" / f"{cfg['target']}_qsar_ready.csv", ["canonical_smiles", "pIC50"]),
        ("step02_compound", out / "step2" / "compound_level_with_properties.csv", ["canonical_smiles"]),
        ("step03_multienv", out / "step3" / "multienv_compound_level.parquet", ["molecule_id", "smiles", "pIC50", "env_id_manual"]),
        ("step04_train_ids", out / "step4" / split / "train_ids.csv", ["molecule_id"]),
        ("step06_pointer", out / "step6" / "run_pointer.json", []),
        ("step08a_external", Path("data/external/processed/ptp1b_inhibition_chembl335/data/inhibition_external_final.parquet"), ["smiles_canonical", "y_inhib_active"]),
    ]
    print("check\tstatus\tdetail")
    failed = 0
    for name, path, req in checks:
        if path.suffix == ".json":
            ok = path.exists()
            detail = "ok" if ok else f"missing file: {path}"
        else:
            ok, detail = cols_ok(path, req)
        status = "PASS" if ok else "FAIL"
        failed += 0 if ok else 1
        print(f"{name}\t{status}\t{detail}")

    ptr = out / "step6" / "run_pointer.json"
    if ptr.exists():
        data = json.loads(ptr.read_text(encoding="utf-8"))
        run_dir = Path(data.get("run_dir", ""))
        feature_schema = run_dir / "artifacts" / "feature_schema.json"
        ok = feature_schema.exists()
        print(f"step12_schema\t{'PASS' if ok else 'FAIL'}\t{'ok' if ok else f'missing file: {feature_schema}'}")
        failed += 0 if ok else 1

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
