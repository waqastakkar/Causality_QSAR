#!/usr/bin/env python
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import BRICS

from stats_utils import fisher_exact, multiple_testing_correction

FUNCTIONAL_GROUP_SMARTS = {
    "amide": "[NX3][CX3](=[OX1])[#6]",
    "sulfonamide": "[NX3][SX4](=[OX1])(=[OX1])[#6]",
    "urea": "[NX3][CX3](=[OX1])[NX3]",
    "carboxylic_acid": "C(=O)[OH]",
    "heteroaromatic": "[a;r5,r6]",
    "halogen": "[F,Cl,Br,I]",
    "ether": "[OD2]([#6])[#6]",
    "amine": "[NX3;H2,H1;!$(NC=O)]",
}


@dataclass
class FragmentOutputs:
    fragment_library: pd.DataFrame
    fragment_enrichment: pd.DataFrame
    functional_group_enrichment: pd.DataFrame
    quality_report: pd.DataFrame


def _extract_fragments(smiles: str, method: str = "brics") -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    if method == "brics":
        return set(BRICS.BRICSDecompose(mol))
    return {Chem.MolToSmiles(mol)}


def _functional_tags(smiles: str) -> set[str]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return set()
    tags = set()
    for name, smarts in FUNCTIONAL_GROUP_SMARTS.items():
        patt = Chem.MolFromSmarts(smarts)
        if patt is not None and mol.HasSubstructMatch(patt):
            tags.add(name)
    return tags


def _enrichment_from_binary(feature_df: pd.DataFrame, positive_col: str, group_name: str, correction: str = "bh") -> pd.DataFrame:
    rows = []
    feats = sorted(feature_df["feature"].unique())
    for feat in feats:
        sub = feature_df[feature_df["feature"] == feat]
        pos_has = int(((sub["present"] == 1) & (sub[positive_col] == 1)).sum())
        pos_not = int(((sub["present"] == 0) & (sub[positive_col] == 1)).sum())
        neg_has = int(((sub["present"] == 1) & (sub[positive_col] == 0)).sum())
        neg_not = int(((sub["present"] == 0) & (sub[positive_col] == 0)).sum())
        orr, p = fisher_exact(pos_has, pos_not, neg_has, neg_not)
        rows.append({"feature": feat, "comparison": group_name, "odds_ratio": orr, "p_value": p, "pos_has": pos_has, "neg_has": neg_has})
    out = pd.DataFrame(rows)
    out["q_value"] = multiple_testing_correction(out["p_value"].values, method=correction)
    return out.sort_values(["q_value", "p_value", "odds_ratio"], ascending=[True, True, False])


def run_fragment_analysis(df: pd.DataFrame, counterfactuals: pd.DataFrame | None = None, bbb: pd.DataFrame | None = None, method: str = "brics", correction: str = "bh") -> FragmentOutputs:
    frag_rows = []
    fg_rows = []
    quality = []
    for _, r in df.iterrows():
        mid = r.get("molecule_id")
        smi = r.get("smiles")
        if not isinstance(smi, str):
            quality.append({"molecule_id": mid, "status": "invalid_smiles"})
            continue
        frags = _extract_fragments(smi, method=method)
        fgs = _functional_tags(smi)
        for f in frags:
            frag_rows.append({"molecule_id": mid, "feature": f, "feature_type": "fragment", "present": 1})
        for fg in fgs:
            fg_rows.append({"molecule_id": mid, "feature": fg, "feature_type": "functional_group", "present": 1})
    frag_df = pd.DataFrame(frag_rows)
    fg_df = pd.DataFrame(fg_rows)

    pot = df[["molecule_id", "pIC50"]].dropna().copy() if "pIC50" in df.columns else pd.DataFrame(columns=["molecule_id", "pIC50"])
    if not pot.empty:
        qh, ql = pot["pIC50"].quantile([0.75, 0.25]).tolist()
        pot["high_potency"] = (pot["pIC50"] >= qh).astype(int)
    enrichments = []
    fg_enrichments = []

    def add_comparison(base_df: pd.DataFrame, label_df: pd.DataFrame, label_col: str, comp_name: str, out_list: list[pd.DataFrame]):
        if base_df.empty or label_df.empty:
            return
        matrix = label_df[["molecule_id", label_col]].merge(base_df[["molecule_id", "feature"]].drop_duplicates(), on="molecule_id", how="left")
        matrix["present"] = matrix["feature"].notna().astype(int)
        for feat in matrix["feature"].dropna().unique():
            pass
        all_features = pd.DataFrame({"feature": base_df["feature"].drop_duplicates()})
        expanded = label_df[["molecule_id", label_col]].assign(key=1).merge(all_features.assign(key=1), on="key").drop(columns="key")
        present = base_df[["molecule_id", "feature"]].drop_duplicates().assign(present=1)
        expanded = expanded.merge(present, on=["molecule_id", "feature"], how="left")
        expanded["present"] = expanded["present"].fillna(0).astype(int)
        out_list.append(_enrichment_from_binary(expanded, label_col, comp_name, correction=correction))

    if not pot.empty:
        add_comparison(frag_df, pot, "high_potency", "high_vs_low_potency", enrichments)
        add_comparison(fg_df, pot, "high_potency", "high_vs_low_potency", fg_enrichments)

    if counterfactuals is not None and not counterfactuals.empty:
        cdf = counterfactuals.copy()
        score_col = "improvement" if "improvement" in cdf.columns else ("delta_yhat" if "delta_yhat" in cdf.columns else None)
        if score_col:
            thr = cdf[score_col].quantile(0.8)
            labels = cdf[["molecule_id", score_col]].dropna().copy()
            labels["top_counterfactual"] = (labels[score_col] >= thr).astype(int)
            add_comparison(frag_df, labels, "top_counterfactual", "top_counterfactual_vs_baseline", enrichments)
            add_comparison(fg_df, labels, "top_counterfactual", "top_counterfactual_vs_baseline", fg_enrichments)

    if bbb is not None and not bbb.empty:
        label_col = "cns_like" if "cns_like" in bbb.columns else ("is_cns" if "is_cns" in bbb.columns else None)
        if label_col:
            labels = bbb[["molecule_id", label_col]].dropna().copy()
            labels[label_col] = labels[label_col].astype(int)
            add_comparison(frag_df, labels, label_col, "cns_like_vs_non_cns", enrichments)
            add_comparison(fg_df, labels, label_col, "cns_like_vs_non_cns", fg_enrichments)

    frag_enrich = pd.concat(enrichments, ignore_index=True) if enrichments else pd.DataFrame()
    fg_enrich = pd.concat(fg_enrichments, ignore_index=True) if fg_enrichments else pd.DataFrame()
    quality_df = pd.DataFrame(quality) if quality else pd.DataFrame([{"status": "ok", "n_molecules": len(df)}])
    return FragmentOutputs(frag_df, frag_enrich, fg_enrich, quality_df)
