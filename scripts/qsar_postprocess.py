#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Step 2 QSAR post-processing: generate core data tables.")
    parser.add_argument("--config", help="Optional pipeline config path (stub compatibility mode)")
    parser.add_argument("--input", required=True, help="Input CSV from Step 1")
    parser.add_argument("--outdir", required=True, help="Output data directory")
    parser.add_argument("--endpoint", default="IC50")
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--units_keep", nargs="+", default=["nM"])
    parser.add_argument("--relation_keep", nargs="+", default=["="])
    parser.add_argument("--aggregate", choices=["best", "median", "mean"], default="best")
    parser.add_argument("--prefer_pchembl", action="store_true")
    parser.add_argument("--svg", action="store_true", help="Retained for interface compatibility")
    return parser.parse_args()


def parse_args_compat() -> tuple[argparse.Namespace, list[str]]:
    """Allow pipeline stub-style invocation (`--config` only) without breaking CLI runs."""
    if "--input" in set(sys.argv):
        return parse_args(), []

    parser = argparse.ArgumentParser(description="Step 2 QSAR post-processing: generate core data tables.")
    parser.add_argument("--config", help="Optional pipeline config path (stub compatibility mode)")
    return parser.parse_known_args()


def _require_column(df: pd.DataFrame, name: str) -> str:
    mapping = {c.lower(): c for c in df.columns}
    if name.lower() not in mapping:
        raise ValueError(f"Required column missing: {name}")
    return mapping[name.lower()]


def _to_pic50(row: pd.Series, standard_value_col: str, pchembl_col: str | None, prefer_pchembl: bool) -> float:
    if prefer_pchembl and pchembl_col and pd.notna(row[pchembl_col]):
        return float(row[pchembl_col])
    value = row[standard_value_col]
    if pd.isna(value):
        return math.nan
    value = float(value)
    if value <= 0:
        return math.nan
    return 9.0 - math.log10(value)


def compute_properties(smiles: str) -> dict[str, float]:
    try:
        from rdkit import Chem
        from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
    except Exception:
        return {
            "mw": np.nan,
            "logp": np.nan,
            "hbd": np.nan,
            "hba": np.nan,
            "tpsa": np.nan,
            "rotatable_bonds": np.nan,
            "aromatic_rings": np.nan,
        }

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return {
            "mw": np.nan,
            "logp": np.nan,
            "hbd": np.nan,
            "hba": np.nan,
            "tpsa": np.nan,
            "rotatable_bonds": np.nan,
            "aromatic_rings": np.nan,
        }

    return {
        "mw": Descriptors.MolWt(mol),
        "logp": Descriptors.MolLogP(mol),
        "hbd": Lipinski.NumHDonors(mol),
        "hba": Lipinski.NumHAcceptors(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "aromatic_rings": Lipinski.RingCount(mol),
    }


def main() -> None:
    args, unknown = parse_args_compat()
    if not getattr(args, "input", None) or not getattr(args, "outdir", None):
        print(f"Executed {__file__} with config={getattr(args, 'config', None)} extra={unknown}")
        return
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.input)

    col_compound = _require_column(df, "molecule_chembl_id")
    col_smiles = _require_column(df, "canonical_smiles")
    col_endpoint = _require_column(df, "standard_type")
    col_units = _require_column(df, "standard_units")
    col_relation = _require_column(df, "standard_relation")
    col_value = _require_column(df, "standard_value")
    pchembl_col = {c.lower(): c for c in df.columns}.get("pchembl_value")

    rows_before = len(df)
    endpoint_mask = df[col_endpoint].astype(str).str.upper() == str(args.endpoint).upper()
    df = df.loc[endpoint_mask].copy()
    rows_after_endpoint = len(df)

    units_keep = {u.upper() for u in args.units_keep}
    rel_keep = set(args.relation_keep)
    df = df[
        df[col_units].astype(str).str.upper().isin(units_keep)
        & df[col_relation].astype(str).isin(rel_keep)
    ].copy()
    rows_after_units_relations = len(df)

    df["pIC50"] = df.apply(
        lambda row: _to_pic50(row, col_value, pchembl_col, args.prefer_pchembl), axis=1
    )
    df = df[df["pIC50"].notna()].copy()
    rows_after_pic50 = len(df)

    row_path = outdir / "row_level_with_pIC50.csv"
    df.to_csv(row_path, index=False)

    agg_func = {"best": "max", "median": "median", "mean": "mean"}[args.aggregate]
    grouped = (
        df.groupby([col_compound, col_smiles], dropna=False)["pIC50"]
        .agg([agg_func, "count"])
        .reset_index()
    )
    grouped.columns = ["molecule_chembl_id", "canonical_smiles", "pIC50", "n_measurements"]
    grouped["activity_label"] = (grouped["pIC50"] >= args.threshold).astype(int)
    comp_path = outdir / "compound_level_pIC50.csv"
    grouped.to_csv(comp_path, index=False)

    props = grouped[["molecule_chembl_id", "canonical_smiles"]].copy()
    props_df = props["canonical_smiles"].apply(compute_properties).apply(pd.Series)
    compound_props = pd.concat([grouped, props_df], axis=1)
    comp_props_path = outdir / "compound_level_with_properties.csv"
    compound_props.to_csv(comp_props_path, index=False)

    summary = pd.DataFrame(
        {
            "metric": [
                "rows_before_filtering",
                "rows_after_endpoint_filter",
                "rows_after_units_relations_filter",
                "rows_after_pic50",
                "n_compounds",
                "n_actives",
                "n_inactives",
            ],
            "value": [
                rows_before,
                rows_after_endpoint,
                rows_after_units_relations,
                rows_after_pic50,
                len(grouped),
                int(grouped["activity_label"].sum()),
                int((grouped["activity_label"] == 0).sum()),
            ],
        }
    )
    summary_path = outdir / "summary.csv"
    summary.to_csv(summary_path, index=False)

    run_config = {
        "endpoint": args.endpoint,
        "threshold": args.threshold,
        "units_keep": args.units_keep,
        "relation_keep": args.relation_keep,
        "aggregate": args.aggregate,
        "prefer_pchembl": bool(args.prefer_pchembl),
        "plotting_params": None,
    }
    (outdir / "postprocess_run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")
    print(f"Wrote: {row_path}, {comp_path}, {comp_props_path}, {summary_path}")


if __name__ == "__main__":
    main()
