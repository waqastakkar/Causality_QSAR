#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import platform
import random
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import accuracy_score
from torch import nn
from torch_geometric.loader import DataLoader

from data_graph import GraphBuildConfig, dataframe_to_graphs, ensure_required_columns, read_split_ids, remap_env_ids, split_dataframe_by_ids
from metrics import (
    classification_metrics,
    expected_calibration_error,
    linear_probe_env_predictability,
    per_environment_metrics,
    regression_calibration,
    regression_metrics,
)
from model_gnn import CausalQSARModel
from plot_style import PlotStyle, configure_matplotlib, parse_palette, style_axis


@dataclass
class TrainConfig:
    target: str
    dataset_parquet: str
    splits_dir: str
    split_name: str
    outdir: str
    task: str
    label_col: str
    env_col: str
    encoder: str
    z_dim: int
    z_inv_dim: int
    z_spu_dim: int
    lambda_adv: float
    lambda_irm: float
    lambda_dis: float
    epochs: int
    batch_size: int
    lr: float
    seed: int
    bbb_parquet: str | None
    svg: bool
    font: str
    bold_text: bool
    palette: str
    font_title: int
    font_label: int
    font_tick: int
    font_legend: int
    run_id: str


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def irm_penalty(yhat: torch.Tensor, y: torch.Tensor, env: torch.Tensor) -> torch.Tensor:
    penalty = torch.tensor(0.0, device=yhat.device)
    scale = torch.tensor(1.0, device=yhat.device, requires_grad=True)
    for e in env.unique():
        m = env == e
        if m.sum() < 2:
            continue
        l = F.mse_loss(yhat[m] * scale, y[m])
        grad = torch.autograd.grad(l, [scale], create_graph=True)[0]
        penalty = penalty + grad.pow(2)
    return penalty


def disentangle_penalty(z_inv: torch.Tensor, z_spu: torch.Tensor) -> torch.Tensor:
    zi = z_inv - z_inv.mean(0, keepdim=True)
    zs = z_spu - z_spu.mean(0, keepdim=True)
    c = (zi.T @ zs) / max(1, zi.shape[0] - 1)
    return c.pow(2).mean()


def evaluate(model, loader, device, task, lambda_grl=1.0):
    model.eval()
    rows = []
    z_all = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch, lambda_grl=lambda_grl)
            pred = out["yhat"].detach().cpu().numpy()
            if task == "classification":
                pred = 1 / (1 + np.exp(-pred))
            envhat = out["envhat"].argmax(dim=1).cpu().numpy()
            for i in range(batch.num_graphs):
                rows.append(
                    {
                        "molecule_id": batch.molecule_id[i],
                        "y_true": float(batch.y[i].cpu()),
                        "y_pred": float(pred[i]),
                        "env_id_manual": int(batch.env[i].cpu()),
                        "env_pred": int(envhat[i]),
                    }
                )
            z_all.append(out["z_inv"].cpu().numpy())
    df = pd.DataFrame(rows)
    z = np.concatenate(z_all, axis=0) if z_all else np.empty((0, 1))
    return df, z


def make_dirs(root: Path):
    for sub in [
        "checkpoints",
        "configs",
        "logs",
        "predictions",
        "reports",
        "figures",
        "provenance",
    ]:
        (root / sub).mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--target", required=True)
    p.add_argument("--dataset_parquet", required=True)
    p.add_argument("--splits_dir", required=True)
    p.add_argument("--split_name", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--task", choices=["regression", "classification"], required=True)
    p.add_argument("--label_col", required=True)
    p.add_argument("--env_col", required=True)
    p.add_argument("--encoder", default="gine", choices=["gine", "graph_transformer"])
    p.add_argument("--z_dim", type=int, default=128)
    p.add_argument("--z_inv_dim", type=int, default=64)
    p.add_argument("--z_spu_dim", type=int, default=64)
    p.add_argument("--lambda_adv", type=float, default=0.5)
    p.add_argument("--lambda_irm", type=float, default=0.0)
    p.add_argument("--lambda_dis", type=float, default=0.1)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--bbb_parquet", default=None)
    p.add_argument("--svg", action="store_true")
    p.add_argument("--font", default="Times New Roman")
    p.add_argument("--bold_text", action="store_true")
    p.add_argument("--palette", default="nature5")
    p.add_argument("--font_title", type=int, default=16)
    p.add_argument("--font_label", type=int, default=14)
    p.add_argument("--font_tick", type=int, default=12)
    p.add_argument("--font_legend", type=int, default=12)
    p.add_argument("--run_id", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    cfg = TrainConfig(**vars(args), run_id=run_id)
    seed_everything(cfg.seed)

    run_root = Path(cfg.outdir) / cfg.target / cfg.split_name / cfg.run_id
    make_dirs(run_root)

    logging.basicConfig(filename=run_root / "logs/train.log", level=logging.INFO)
    style = PlotStyle(
        font_family=cfg.font,
        font_title=cfg.font_title,
        font_label=cfg.font_label,
        font_tick=cfg.font_tick,
        font_legend=cfg.font_legend,
        bold_text=cfg.bold_text,
        palette=parse_palette(cfg.palette),
    )
    configure_matplotlib(style, svg=cfg.svg)

    with (run_root / "configs/train_config.yaml").open("w") as f:
        yaml.safe_dump(vars(args), f)
    with (run_root / "configs/resolved_config.yaml").open("w") as f:
        yaml.safe_dump(asdict(cfg), f)

    df = pd.read_parquet(cfg.dataset_parquet)
    ensure_required_columns(df, ["molecule_id", "smiles", cfg.label_col, cfg.env_col])
    df, env_map = remap_env_ids(df, cfg.env_col)
    split_ids = read_split_ids(cfg.splits_dir, cfg.split_name)
    split_df = split_dataframe_by_ids(df, split_ids)

    gcfg = GraphBuildConfig(label_col=cfg.label_col, env_col=cfg.env_col)
    train_graphs = dataframe_to_graphs(split_df["train"], gcfg)
    val_graphs = dataframe_to_graphs(split_df["val"], gcfg)
    test_graphs = dataframe_to_graphs(split_df["test"], gcfg)

    train_loader = DataLoader(train_graphs, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_graphs, batch_size=cfg.batch_size, shuffle=False)

    node_dim = train_graphs[0].x.shape[1]
    edge_dim = train_graphs[0].edge_attr.shape[1]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = CausalQSARModel(
        node_dim=node_dim,
        edge_dim=edge_dim,
        z_dim=cfg.z_dim,
        z_inv_dim=cfg.z_inv_dim,
        z_spu_dim=cfg.z_spu_dim,
        n_envs=len(env_map),
        task=cfg.task,
        encoder=cfg.encoder,
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    pred_loss_fn = nn.HuberLoss() if cfg.task == "regression" else nn.BCEWithLogitsLoss()

    best_val = float("inf")
    history = []
    jsonl = (run_root / "logs/metrics.jsonl").open("w")

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        losses = {"pred": 0.0, "adv": 0.0, "irm": 0.0, "dis": 0.0, "total": 0.0}
        n = 0
        for batch in train_loader:
            batch = batch.to(device)
            out = model(batch, lambda_grl=1.0)
            y = batch.y.float()
            yhat = out["yhat"]
            l_pred = pred_loss_fn(yhat, y)
            l_adv = F.cross_entropy(out["envhat"], batch.env)
            l_irm = irm_penalty(yhat, y, batch.env) if cfg.lambda_irm > 0 else torch.tensor(0.0, device=device)
            l_dis = disentangle_penalty(out["z_inv"], out["z_spu"])
            loss = l_pred + cfg.lambda_adv * l_adv + cfg.lambda_irm * l_irm + cfg.lambda_dis * l_dis

            opt.zero_grad()
            loss.backward()
            opt.step()

            bs = batch.num_graphs
            n += bs
            losses["pred"] += l_pred.item() * bs
            losses["adv"] += l_adv.item() * bs
            losses["irm"] += float(l_irm.item()) * bs
            losses["dis"] += l_dis.item() * bs
            losses["total"] += loss.item() * bs

        train_pred_df, _ = evaluate(model, train_loader, device, cfg.task)
        val_pred_df, _ = evaluate(model, val_loader, device, cfg.task)
        if cfg.task == "regression":
            vmetric = regression_metrics(val_pred_df["y_true"], val_pred_df["y_pred"])
            score = vmetric["rmse"]
        else:
            vmetric = classification_metrics(val_pred_df["y_true"], val_pred_df["y_pred"])
            score = -float(vmetric.get("auc", 0.0) if not np.isnan(vmetric.get("auc", np.nan)) else 0.0)

        row = {
            "epoch": epoch,
            **{f"loss_{k}": losses[k] / max(1, n) for k in losses},
            **{f"val_{k}": v for k, v in vmetric.items()},
        }
        history.append(row)
        jsonl.write(json.dumps(row) + "\n")
        jsonl.flush()

        if score < best_val:
            best_val = score
            torch.save(model.state_dict(), run_root / "checkpoints/best.pt")
        torch.save(model.state_dict(), run_root / "checkpoints/last.pt")

    jsonl.close()
    model.load_state_dict(torch.load(run_root / "checkpoints/best.pt", map_location=device))

    pred_dfs = {}
    z_splits = {}
    for split, loader in [("train", train_loader), ("val", val_loader), ("test", test_loader)]:
        pred_df, z = evaluate(model, loader, device, cfg.task)
        pred_dfs[split] = pred_df
        z_splits[split] = z
        pred_df.to_parquet(run_root / f"predictions/{split}_predictions.parquet", index=False)

    test_df = pred_dfs["test"]
    if cfg.task == "regression":
        ms = regression_metrics(test_df["y_true"], test_df["y_pred"])
    else:
        ms = classification_metrics(test_df["y_true"], test_df["y_pred"])
    pd.DataFrame([{"split": "test", **ms}]).to_csv(run_root / "reports/metrics_summary.csv", index=False)

    per_env = per_environment_metrics(test_df, cfg.task, "env_id_manual", "y_true", "y_pred")
    per_env.to_csv(run_root / "reports/per_env_metrics.csv", index=False)

    env_adv_acc = accuracy_score(test_df["env_id_manual"], test_df["env_pred"]) if len(test_df) else np.nan
    z_probe = linear_probe_env_predictability(z_splits["test"], test_df["env_id_manual"].to_numpy()) if len(test_df) else np.nan
    if cfg.task == "regression" and "rmse" in per_env.columns:
        perf_var = float(per_env["rmse"].var())
    elif cfg.task == "classification" and "auc" in per_env.columns:
        perf_var = float(per_env["auc"].var())
    else:
        perf_var = float("nan")
    inv = pd.DataFrame(
        [
            {"metric": "adversary_accuracy", "value": env_adv_acc},
            {"metric": "linear_probe_env_predictability", "value": z_probe},
            {"metric": "performance_variance_across_env", "value": perf_var},
        ]
    )
    inv.to_csv(run_root / "reports/invariance_checks.csv", index=False)

    if cfg.task == "classification":
        ece, cal_df = expected_calibration_error(test_df["y_true"], test_df["y_pred"]) 
        cal_df["ece"] = ece
    else:
        cal_df = regression_calibration(test_df["y_true"], test_df["y_pred"])
    cal_df.to_csv(run_root / "reports/calibration.csv", index=False)

    bbb_metrics = pd.DataFrame()
    if cfg.bbb_parquet:
        bbb = pd.read_parquet(cfg.bbb_parquet)
        merged = test_df.merge(bbb, on="molecule_id", how="left")
        strat_cols = [c for c in ["cns_like", "cns_mpo_bin"] if c in merged.columns]
        rows = []
        for c in strat_cols:
            for value, g in merged.groupby(c):
                if len(g) < 2:
                    continue
                m = regression_metrics(g.y_true, g.y_pred) if cfg.task == "regression" else classification_metrics(g.y_true, g.y_pred)
                rows.append({"stratum": c, "value": value, **m, "n": len(g)})
        bbb_metrics = pd.DataFrame(rows)
    bbb_metrics.to_csv(run_root / "reports/bbb_metrics.csv", index=False)

    ablation = pd.DataFrame(
        [
            {
                "run_id": cfg.run_id,
                "split_name": cfg.split_name,
                "seed": cfg.seed,
                "encoder": cfg.encoder,
                "lambda_adv": cfg.lambda_adv,
                "lambda_irm": cfg.lambda_irm,
                "lambda_dis": cfg.lambda_dis,
                **ms,
            }
        ]
    )
    ablation.to_csv(run_root / "reports/ablation_table.csv", index=False)

    hist = pd.DataFrame(history)
    fig, ax = plt.subplots(figsize=(8, 5))
    for c in ["loss_pred", "loss_adv", "loss_irm", "loss_dis", "loss_total"]:
        if c in hist:
            ax.plot(hist["epoch"], hist[c], label=c)
    style_axis(ax, style, "Learning Curves", "Epoch", "Loss")
    ax.legend()
    fig.tight_layout(); fig.savefig(run_root / "figures/fig_learning_curves.svg")

    fig, ax = plt.subplots(figsize=(6, 4))
    vals = []
    for split in ["train", "val", "test"]:
        if cfg.task == "regression":
            vals.append(regression_metrics(pred_dfs[split].y_true, pred_dfs[split].y_pred)["rmse"])
        else:
            vals.append(classification_metrics(pred_dfs[split].y_true, pred_dfs[split].y_pred)["auc"])
    ax.bar(["train", "val", "test"], vals)
    style_axis(ax, style, "Performance by Split", "Split", "RMSE" if cfg.task == "regression" else "AUC")
    fig.tight_layout(); fig.savefig(run_root / "figures/fig_perf_by_split.svg")

    fig, ax = plt.subplots(figsize=(7, 4))
    metric_col = "rmse" if cfg.task == "regression" else "auc"
    if metric_col in per_env.columns:
        ax.bar(per_env["env_id_manual"].astype(str), per_env[metric_col])
    style_axis(ax, style, "Performance by Environment", "env_id_manual", metric_col)
    fig.tight_layout(); fig.savefig(run_root / "figures/fig_perf_by_env.svg")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["adversary_acc"], [env_adv_acc])
    style_axis(ax, style, "Environment Adversary Accuracy", "", "Accuracy")
    fig.tight_layout(); fig.savefig(run_root / "figures/fig_env_adversary_accuracy.svg")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(["linear_probe"], [z_probe])
    style_axis(ax, style, "z_inv Environment Predictability", "", "Accuracy")
    fig.tight_layout(); fig.savefig(run_root / "figures/fig_zinv_env_predictability.svg")

    fig, ax = plt.subplots(figsize=(6, 4))
    if cfg.task == "classification" and not cal_df.empty and {"confidence", "accuracy"}.issubset(cal_df.columns):
        ax.plot(cal_df["confidence"], cal_df["accuracy"], marker="o", label="observed")
        ax.plot([0, 1], [0, 1], "--", label="ideal")
        ax.legend()
        style_axis(ax, style, "Calibration", "Confidence", "Accuracy")
    else:
        if not cal_df.empty and {"pred_mean", "obs_mean"}.issubset(cal_df.columns):
            ax.plot(cal_df["pred_mean"], cal_df["obs_mean"], marker="o")
            lo, hi = cal_df[["pred_mean", "obs_mean"]].min().min(), cal_df[["pred_mean", "obs_mean"]].max().max()
            ax.plot([lo, hi], [lo, hi], "--")
        style_axis(ax, style, "Calibration", "Predicted", "Observed")
    fig.tight_layout(); fig.savefig(run_root / "figures/fig_calibration.svg")

    fig, ax = plt.subplots(figsize=(7, 4))
    if not bbb_metrics.empty and ("rmse" in bbb_metrics.columns or "auc" in bbb_metrics.columns):
        mcol = "rmse" if "rmse" in bbb_metrics.columns else "auc"
        ax.bar(bbb_metrics["value"].astype(str), bbb_metrics[mcol])
    style_axis(ax, style, "BBB Stratified Performance", "Stratum", "Metric")
    fig.tight_layout(); fig.savefig(run_root / "figures/fig_bbb_stratified_perf.svg")

    if cfg.bbb_parquet and not bbb_metrics.empty and "cns_mpo_bin" in bbb_metrics["stratum"].astype(str).tolist():
        fig, ax = plt.subplots(figsize=(6, 4))
        mpo = bbb_metrics[bbb_metrics["stratum"] == "cns_mpo_bin"]
        ycol = "rmse" if "rmse" in mpo.columns else "auc"
        ax.scatter(mpo["value"].astype(str), mpo[ycol])
        style_axis(ax, style, "Pareto: potency vs CNS", "CNS MPO bin", ycol)
        fig.tight_layout(); fig.savefig(run_root / "figures/fig_pareto_potency_vs_cns.svg")

    scripts = [
        "scripts/train_causal_qsar.py",
        "scripts/model_gnn.py",
        "scripts/data_graph.py",
        "scripts/metrics.py",
        "scripts/plot_style.py",
    ]
    prov = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_args": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit_hash": git_commit(),
        "sha256_scripts": {s: sha256_file(Path(s)) for s in scripts if Path(s).exists()},
        "sha256_inputs": {
            "dataset_parquet": sha256_file(Path(cfg.dataset_parquet)),
            "train_ids": sha256_file(Path(cfg.splits_dir) / cfg.split_name / "train_ids.csv"),
            "val_ids": sha256_file(Path(cfg.splits_dir) / cfg.split_name / "val_ids.csv"),
            "test_ids": sha256_file(Path(cfg.splits_dir) / cfg.split_name / "test_ids.csv"),
        },
        "run_id": cfg.run_id,
        "seed": cfg.seed,
        "split_name": cfg.split_name,
        "model_hyperparams": {"encoder": cfg.encoder, "z_dim": cfg.z_dim, "z_inv_dim": cfg.z_inv_dim, "z_spu_dim": cfg.z_spu_dim},
        "loss_weights": {"lambda_adv": cfg.lambda_adv, "lambda_irm": cfg.lambda_irm, "lambda_dis": cfg.lambda_dis},
        "number_of_envs": len(env_map),
        "class_balance": float(df[cfg.label_col].mean()) if cfg.task == "classification" else None,
        "cns_stats": bbb_metrics.groupby("stratum").size().to_dict() if not bbb_metrics.empty else None,
    }
    (run_root / "provenance/provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    (run_root / "provenance/run_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    freeze = subprocess.run(["python", "-m", "pip", "freeze"], capture_output=True, text=True)
    (run_root / "provenance/environment.txt").write_text(freeze.stdout + freeze.stderr, encoding="utf-8")


if __name__ == "__main__":
    main()
