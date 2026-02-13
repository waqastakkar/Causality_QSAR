#!/usr/bin/env python
from __future__ import annotations

import numpy as np
import pandas as pd


def fingerprint_ad(train_df: pd.DataFrame, test_df: pd.DataFrame, radius: int = 2, nbits: int = 2048) -> pd.DataFrame:
    out = test_df[["molecule_id"]].copy()
    try:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem

        train_fp = []
        for s in train_df["smiles"].fillna(""):
            m = Chem.MolFromSmiles(str(s))
            if m is not None:
                train_fp.append(AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nbits))
        dists = []
        for s in test_df["smiles"].fillna(""):
            m = Chem.MolFromSmiles(str(s))
            if m is None or not train_fp:
                dists.append(np.nan)
                continue
            fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=radius, nBits=nbits)
            sims = DataStructs.BulkTanimotoSimilarity(fp, train_fp)
            dists.append(1.0 - float(max(sims)) if sims else np.nan)
        out["ad_distance_fingerprint"] = dists
    except Exception:
        out["ad_distance_fingerprint"] = np.nan
    return out


def embedding_ad(train_z: pd.DataFrame, test_z: pd.DataFrame, k: int = 1) -> pd.DataFrame:
    out = test_z[["molecule_id"]].copy()
    zcols = [c for c in train_z.columns if c.startswith("z_inv_") and c in test_z.columns]
    if not zcols:
        out["ad_distance_embedding"] = np.nan
        return out
    tr = train_z[zcols].to_numpy(dtype=float)
    te = test_z[zcols].to_numpy(dtype=float)
    d = np.sqrt(((te[:, None, :] - tr[None, :, :]) ** 2).sum(axis=2))
    dsort = np.sort(d, axis=1)
    out["ad_distance_embedding"] = dsort[:, : max(1, k)].mean(axis=1)
    return out


def binned_relationship(df: pd.DataFrame, xcol: str, ycol: str, bins: int = 10, out_x: str = "ad_bin") -> pd.DataFrame:
    work = df[[xcol, ycol]].dropna().copy()
    if work.empty:
        return pd.DataFrame(columns=[out_x, "y_mean", "n"])
    work[out_x] = pd.qcut(work[xcol], q=min(bins, max(2, work[xcol].nunique())), duplicates="drop")
    g = work.groupby(out_x, observed=False).agg(y_mean=(ycol, "mean"), n=(ycol, "size"), x_mean=(xcol, "mean")).reset_index()
    return g
