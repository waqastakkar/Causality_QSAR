#!/usr/bin/env python
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def extract_zinv_embeddings(
    run_dir: str | Path,
    dataset_df: pd.DataFrame,
    molecule_ids: list,
    label_col: str = "pIC50",
    env_col: str = "env_id_manual",
    batch_size: int = 128,
) -> pd.DataFrame:
    """Extract z_inv embeddings for selected molecule IDs using a trained checkpoint.

    If model tooling is unavailable, returns an empty DataFrame with molecule_id only.
    """
    run_dir = Path(run_dir)
    work = dataset_df[dataset_df["molecule_id"].isin(set(molecule_ids))].copy()
    if work.empty:
        return pd.DataFrame(columns=["molecule_id"])
    if label_col not in work.columns:
        work[label_col] = 0.0
    if env_col not in work.columns:
        work[env_col] = 0

    try:
        import torch
        from torch_geometric.loader import DataLoader

        from data_graph import GraphBuildConfig, dataframe_to_graphs
        from evaluate_cross_endpoint import _infer_model_dims
        from model_gnn import CausalQSARModel
    except Exception:
        return work[["molecule_id"]].drop_duplicates().reset_index(drop=True)

    ckpt = run_dir / "checkpoints" / "best.pt"
    if not ckpt.exists():
        return work[["molecule_id"]].drop_duplicates().reset_index(drop=True)

    state = torch.load(ckpt, map_location="cpu")
    dims = _infer_model_dims(state)
    gcfg = GraphBuildConfig(smiles_col="smiles_canonical" if "smiles_canonical" in work.columns else "smiles", id_col="molecule_id", label_col=label_col, env_col=env_col)
    graphs = dataframe_to_graphs(work, gcfg)
    model = CausalQSARModel(node_dim=129, edge_dim=7, z_dim=dims["z_dim"], z_inv_dim=dims["z_inv_dim"], z_spu_dim=dims["z_spu_dim"], n_envs=dims["n_envs"], task="regression")
    model.load_state_dict(state)
    model.eval()

    rows = []
    with torch.no_grad():
        for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
            out = model(batch)
            z = out["z_inv"].detach().cpu().numpy()
            for i in range(batch.num_graphs):
                row = {"molecule_id": batch.molecule_id[i]}
                for j in range(z.shape[1]):
                    row[f"z_inv_{j}"] = float(z[i, j])
                rows.append(row)
    if not rows:
        return work[["molecule_id"]].drop_duplicates().reset_index(drop=True)
    return pd.DataFrame(rows)
