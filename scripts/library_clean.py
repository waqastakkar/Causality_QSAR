from __future__ import annotations

from collections import Counter

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def clean_library(parsed_df: pd.DataFrame, strip_salts: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    rows = []
    fail = Counter()
    remover = rdMolStandardize.FragmentParent if strip_salts else None
    for r in parsed_df.itertuples(index=False):
        rec = r._asdict()
        if rec.get("parse_status") != "ok":
            fail[f"parse_{rec.get('parse_error', 'unknown')}"] += 1
            continue
        smi = str(rec.get("smiles") or "").strip()
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fail["invalid_smiles"] += 1
            continue
        try:
            Chem.SanitizeMol(mol)
            if remover is not None:
                mol = remover(mol)
                Chem.SanitizeMol(mol)
            can = Chem.MolToSmiles(mol, canonical=True)
            ik = Chem.MolToInchiKey(mol)
        except Exception:
            fail["sanitization_failure"] += 1
            continue
        rec["canonical_smiles"] = can
        rec["inchikey"] = ik
        rows.append(rec)

    clean = pd.DataFrame(rows)
    if clean.empty:
        dedup = clean.copy()
    else:
        agg = {c: "first" for c in clean.columns if c != "compound_id"}
        agg["original_ids"] = ("compound_id", lambda s: sorted({str(x) for x in s if str(x)}))
        dedup = clean.assign(compound_id=clean["compound_id"].astype(str)).groupby("inchikey", as_index=False).agg(**agg)
        dedup["compound_id"] = dedup["original_ids"].map(lambda x: x[0] if isinstance(x, list) and x else None)

    report = {
        "clean_input_rows": int(len(parsed_df)),
        "clean_valid_rows": int(len(clean)),
        "clean_dedup_rows": int(len(dedup)),
        "deduplicated_count": int(len(clean) - len(dedup)),
        **{f"fail_{k}": int(v) for k, v in fail.items()},
    }
    return clean, dedup, report
