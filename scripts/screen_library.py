#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from applicability_domain import build_or_load_train_fingerprint_index, fingerprint_ad
from bbb_rules import add_bbb_metrics
from featurize import featurize_library
from infer import run_inference
from library_clean import clean_library
from library_io import parse_library
from plot_style import add_plot_style_args, configure_matplotlib, style_axis, style_from_args
from property_calc import compute_properties
from screening_reports import build_rankings


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def _bool(s: str | bool) -> bool:
    if isinstance(s, bool):
        return s
    return str(s).lower() in {"1", "true", "yes", "y"}


def _mkdirs(root: Path):
    for s in ["input", "processed", "predictions", "ranking", "figures", "figure_data", "provenance"]:
        (root / s).mkdir(parents=True, exist_ok=True)


def parse_args():
    p = argparse.ArgumentParser(description="Step 12: library screening")
    p.add_argument("--target", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--input_path", required=True)
    p.add_argument("--input_format", choices=["smi", "csv"], required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--screen_id", default=None)
    p.add_argument("--smi_layout", choices=["smiles_id", "smiles_name_id", "smiles_only"], default="smiles_id")
    p.add_argument("--header", default="auto")
    p.add_argument("--comment_prefix", default="#")
    p.add_argument("--allow_cpp_comments", default="true")
    p.add_argument("--name_is_rest", default="true")
    p.add_argument("--smi_quoted_name", default="false")
    p.add_argument("--sep", default=",")
    p.add_argument("--quotechar", default='"')
    p.add_argument("--smiles_col")
    p.add_argument("--id_col")
    p.add_argument("--name_col")
    p.add_argument("--use_ensemble_manifest")
    p.add_argument("--compute_bbb", default="true")
    p.add_argument("--cns_mpo_threshold", type=float, default=4.0)
    p.add_argument("--compute_ad", default="true")
    p.add_argument("--ad_mode", default="fingerprint", choices=["fingerprint", "embedding", "fingerprint+embedding"])
    p.add_argument("--ad_threshold", type=float, default=None)
    p.add_argument("--topk", type=int, default=500)
    add_plot_style_args(p)
    return p.parse_args()


def main():
    args = parse_args()
    run_dir = Path(args.run_dir)
    screen_id = args.screen_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = Path(args.outdir) if Path(args.outdir).name == screen_id else Path(args.outdir) / args.target / screen_id
    _mkdirs(out)

    style = style_from_args(args)
    configure_matplotlib(style, svg=True)

    parsed, manifest = parse_library(
        input_path=args.input_path,
        input_format=args.input_format,
        smi_layout=args.smi_layout,
        header=args.header,
        comment_prefix=args.comment_prefix,
        allow_cpp_comments=_bool(args.allow_cpp_comments),
        name_is_rest=_bool(args.name_is_rest),
        smi_quoted_name=_bool(args.smi_quoted_name),
        sep=args.sep,
        quotechar=args.quotechar,
        smiles_col=args.smiles_col,
        id_col=args.id_col,
        name_col=args.name_col,
    )
    parsed.to_parquet(out / "processed/library_raw_parsed.parquet", index=False)
    pd.DataFrame([manifest]).to_csv(out / "input/library_manifest.csv", index=False)
    (out / "input/input_fingerprint.json").write_text(json.dumps({"sha256": _sha256(Path(args.input_path))}, indent=2))

    clean, dedup, clean_report = clean_library(parsed)
    clean.to_parquet(out / "processed/library_clean.parquet", index=False)
    dedup.to_parquet(out / "processed/library_dedup.parquet", index=False)

    with_props = compute_properties(dedup, smiles_col="canonical_smiles")
    if _bool(args.compute_bbb):
        with_props = add_bbb_metrics(with_props, cns_mpo_threshold=args.cns_mpo_threshold)
    with_props.to_parquet(out / "processed/library_with_props.parquet", index=False)

    graphs, feat_df, feat_report = featurize_library(with_props, run_dir)
    single, ens = run_inference(graphs, run_dir, ensemble_manifest=args.use_ensemble_manifest)
    single.to_parquet(out / "predictions/scored_single_model.parquet", index=False)
    ens.to_parquet(out / "predictions/scored_ensemble.parquet", index=False)

    scored = feat_df.merge(ens[["compound_id", "score_mean", "score_std"]], on="compound_id", how="inner")
    scored["ad_distance"] = np.nan
    if _bool(args.compute_ad) and "fingerprint" in args.ad_mode:
        idx = build_or_load_train_fingerprint_index(run_dir)
        ad = fingerprint_ad(idx, scored.rename(columns={"canonical_smiles": "smiles", "compound_id": "molecule_id"}))
        scored = scored.merge(ad.rename(columns={"molecule_id": "compound_id", "ad_distance_fingerprint": "ad_distance"}), on="compound_id", how="left")
    cols = ["compound_id", "compound_name", "canonical_smiles", "inchikey", "score_mean", "score_std", "ad_distance"]
    scored.to_parquet(out / "predictions/scored_with_uncertainty.parquet", index=False)

    ranks, sel = build_rankings(scored, args.cns_mpo_threshold, args.ad_threshold, args.topk)
    for k in ["ranked_all", "ranked_cns_like", "ranked_in_domain", "ranked_cns_like_in_domain"]:
        ranks[k].to_parquet(out / f"ranking/{k}.parquet", index=False)
    ranks["best"].head(100).to_csv(out / "ranking/top_100.csv", index=False)
    ranks["best"].head(500).to_csv(out / "ranking/top_500.csv", index=False)
    sel.to_csv(out / "ranking/selection_report.csv", index=False)

    report = {
        "parsed_total": int(len(parsed)),
        "parsed_ok": int((parsed["parse_status"] == "ok").sum()) if "parse_status" in parsed else 0,
        "parsed_fail": int((parsed["parse_status"] != "ok").sum()) if "parse_status" in parsed else 0,
        **clean_report,
        **feat_report,
        "final_scored_count": int(len(scored)),
    }
    pd.DataFrame([{"metric": k, "value": v} for k, v in report.items()]).to_csv(out / "processed/featurization_report.csv", index=False)

    # figures + figure_data
    s = scored[["score_mean"]].dropna(); s.to_csv(out / "figure_data/score_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(s["score_mean"], bins=30); style_axis(ax, style, "Score distribution", "score_mean", "Count"); fig.tight_layout(); fig.savefig(out / "figures/fig_score_distribution.svg"); plt.close(fig)

    u = scored[["score_std"]].dropna(); u.to_csv(out / "figure_data/uncertainty_distribution.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4)); ax.hist(u["score_std"], bins=30); style_axis(ax, style, "Uncertainty distribution", "score_std", "Count"); fig.tight_layout(); fig.savefig(out / "figures/fig_uncertainty_distribution.svg"); plt.close(fig)

    p = scored[["score_mean", "cns_mpo"]].dropna() if "cns_mpo" in scored.columns else pd.DataFrame(columns=["score_mean", "cns_mpo"])
    p.to_csv(out / "figure_data/pareto_score_vs_cns.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not p.empty: ax.scatter(p["score_mean"], p["cns_mpo"], s=10, alpha=0.6)
    style_axis(ax, style, "Score vs CNS MPO", "score_mean", "cns_mpo"); fig.tight_layout(); fig.savefig(out / "figures/fig_pareto_score_vs_cns.svg"); plt.close(fig)

    a = scored[["score_mean", "ad_distance"]].dropna()
    a.to_csv(out / "figure_data/score_vs_ad.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if not a.empty: ax.scatter(a["ad_distance"], a["score_mean"], s=10, alpha=0.6)
    style_axis(ax, style, "Score vs AD", "ad_distance", "score_mean"); fig.tight_layout(); fig.savefig(out / "figures/fig_score_vs_ad.svg"); plt.close(fig)

    props = [c for c in ["MW", "LogP", "TPSA"] if c in scored.columns]
    top = ranks["best"].head(args.topk)
    box = pd.concat([scored[props].assign(group="all"), top[props].assign(group=f"top_{args.topk}")], ignore_index=True) if props else pd.DataFrame()
    box.to_csv(out / "figure_data/topk_property_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6, 4));
    if props and not box.empty:
        ax.boxplot([box[box["group"] == "all"][props[0]].dropna(), box[box["group"] != "all"][props[0]].dropna()], labels=["all", f"top_{args.topk}"])
        ylabel = props[0]
    else:
        ylabel = "value"
    style_axis(ax, style, "TopK property summary", "Group", ylabel); fig.tight_layout(); fig.savefig(out / "figures/fig_topk_property_summary.svg"); plt.close(fig)

    cfg = vars(args)
    (out / "provenance/run_config.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    prov = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "input_hash": _sha256(Path(args.input_path)),
        "run_artifact_hashes": {str(p.relative_to(run_dir)): _sha256(p) for p in [run_dir / "artifacts/feature_schema.json", run_dir / "configs/resolved_config.yaml"] if p.exists()},
        "checkpoint_hashes": {str(p): _sha256(Path(p)) for p in ([str(run_dir / "checkpoints/best.pt")] if not args.use_ensemble_manifest else json.loads(Path(args.use_ensemble_manifest).read_text()).get("checkpoint_paths", [])) if Path(p).exists()},
        "script_hashes": {str(Path(__file__).name): _sha256(Path(__file__))},
    }
    (out / "provenance/provenance.json").write_text(json.dumps(prov, indent=2), encoding="utf-8")
    (out / "provenance/environment.txt").write_text(subprocess.check_output(["pip", "freeze"], text=True), encoding="utf-8")


if __name__ == "__main__":
    main()
