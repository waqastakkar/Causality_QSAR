#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import font_manager

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_style import NATURE5, PlotStyle, configure_matplotlib, style_axis


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Step 2 report figures + provenance.")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--endpoint", default="IC50")
    parser.add_argument("--threshold", type=float, default=6.0)
    parser.add_argument("--units_keep", nargs="+", default=["nM"])
    parser.add_argument("--relation_keep", nargs="+", default=["="])
    parser.add_argument("--aggregate", choices=["best", "median", "mean"], default="best")
    parser.add_argument("--prefer_pchembl", action="store_true")
    parser.add_argument("--svg", action="store_true")
    parser.add_argument("--font", default="Times New Roman")
    parser.add_argument("--bold_text", action="store_true")
    parser.add_argument("--palette", default="nature5")
    parser.add_argument("--font_title", type=int, default=16)
    parser.add_argument("--font_label", type=int, default=14)
    parser.add_argument("--font_tick", type=int, default=12)
    parser.add_argument("--font_legend", type=int, default=12)
    return parser.parse_args()



def resolve_font_status(font_name: str) -> dict[str, str | bool]:
    fallback = font_manager.findfont(font_name, fallback_to_default=True)
    try:
        exact = font_manager.findfont(font_name, fallback_to_default=False)
        return {"requested": font_name, "available": True, "resolved_path": exact}
    except Exception:
        return {"requested": font_name, "available": False, "resolved_path": fallback}

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str | None:
    try:
        result = subprocess.run(["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
        return result.stdout.strip()
    except Exception:
        return None


def load_metrics(summary_path: Path) -> dict[str, int]:
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    return {str(r["metric"]): int(r["value"]) for _, r in summary.iterrows()}


def save_class_balance(df: pd.DataFrame, figures_dir: Path, style: PlotStyle) -> None:
    counts = df["activity_label"].value_counts().reindex([1, 0], fill_value=0)
    labels = ["Active", "Inactive"]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, counts.values, color=[style.palette[1], style.palette[3]])
    style_axis(ax, style, title="Class balance", ylabel="Compound count")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_class_balance.svg")
    plt.close(fig)


def save_spider(df: pd.DataFrame, figures_dir: Path, style: PlotStyle) -> None:
    props = ["mw", "logp", "hbd", "hba", "tpsa"]
    available = [c for c in props if c in df.columns]
    if len(available) < 3:
        return
    grouped = df.groupby("activity_label")[available].median()
    if grouped.empty:
        return

    norm = grouped / grouped.max(axis=0)
    angles = [n / float(len(available)) * 2 * 3.141592653589793 for n in range(len(available))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    for idx, (label, row) in enumerate(norm.iterrows()):
        values = row.tolist() + row.tolist()[:1]
        name = "Active" if int(label) == 1 else "Inactive"
        color = style.palette[idx % len(style.palette)]
        ax.plot(angles, values, linewidth=2, label=name, color=color)
        ax.fill(angles, values, alpha=0.15, color=color)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([p.upper() for p in available])
    style_axis(ax, style, title="Property profile: active vs inactive")
    ax.legend(loc="upper right", bbox_to_anchor=(1.2, 1.1))
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_spider_properties_active_vs_inactive.svg")
    plt.close(fig)


def save_bubble(df: pd.DataFrame, figures_dir: Path, style: PlotStyle) -> None:
    needed = {"mw", "logp", "pIC50"}
    if not needed.issubset(df.columns):
        return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = df["activity_label"].map({1: style.palette[2], 0: style.palette[4]}).fillna(style.palette[0])
    size = (df["pIC50"].clip(lower=0) + 1) * 20
    ax.scatter(df["mw"], df["logp"], s=size, c=colors, alpha=0.7)
    style_axis(ax, style, title="MW vs LogP bubble plot", xlabel="Molecular weight", ylabel="LogP")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_bubble_mw_vs_logp.svg")
    plt.close(fig)


def save_pic50_dist(row_df: pd.DataFrame, figures_dir: Path, style: PlotStyle) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(row_df["pIC50"].dropna(), bins=30, color=style.palette[0], edgecolor="black", alpha=0.8)
    style_axis(ax, style, title="pIC50 distribution", xlabel="pIC50", ylabel="Frequency")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_pIC50_distribution.svg")
    plt.close(fig)


def save_endpoint_units_relations(row_df: pd.DataFrame, figures_dir: Path, style: PlotStyle) -> None:
    cols = [c for c in ["standard_type", "standard_units", "standard_relation"] if c in row_df.columns]
    if not cols:
        return
    fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4))
    if len(cols) == 1:
        axes = [axes]
    for i, col in enumerate(cols):
        counts = row_df[col].astype(str).value_counts().head(10)
        axes[i].bar(counts.index, counts.values, color=style.palette[i % len(style.palette)])
        axes[i].tick_params(axis="x", rotation=45)
        style_axis(axes[i], style, title=col.replace("_", " ").title(), ylabel="Rows")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_endpoint_units_relations.svg")
    plt.close(fig)


def save_missingness(df: pd.DataFrame, figures_dir: Path, style: PlotStyle) -> None:
    props = [c for c in ["mw", "logp", "hbd", "hba", "tpsa", "rotatable_bonds", "aromatic_rings"] if c in df.columns]
    if not props:
        return
    miss = df[props].isna().mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(miss.index, miss.values, color=style.palette[3])
    ax.tick_params(axis="x", rotation=45)
    style_axis(ax, style, title="Property missingness", ylabel="Missing fraction")
    fig.tight_layout()
    fig.savefig(figures_dir / "fig_missingness_properties.svg")
    plt.close(fig)


def write_environment(path: Path) -> None:
    proc = subprocess.run([sys.executable, "-m", "pip", "freeze"], capture_output=True, text=True)
    path.write_text(proc.stdout + ("\n" + proc.stderr if proc.stderr else ""), encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    outdir = Path(args.outdir)
    figures_dir = outdir / "figures"
    provenance_dir = outdir / "provenance"
    figures_dir.mkdir(parents=True, exist_ok=True)
    provenance_dir.mkdir(parents=True, exist_ok=True)

    palette = NATURE5 if args.palette.lower() == "nature5" else NATURE5
    font_status = resolve_font_status(args.font)
    style = PlotStyle(
        font_family=args.font,
        font_title=args.font_title,
        font_label=args.font_label,
        font_tick=args.font_tick,
        font_legend=args.font_legend,
        palette=tuple(palette),
    )
    configure_matplotlib(style)

    row_path = input_dir / "row_level_with_pIC50.csv"
    comp_props_path = input_dir / "compound_level_with_properties.csv"
    summary_path = input_dir / "summary.csv"
    row_df = pd.read_csv(row_path)
    comp_df = pd.read_csv(comp_props_path)

    save_class_balance(comp_df, figures_dir, style)
    save_spider(comp_df, figures_dir, style)
    save_bubble(comp_df, figures_dir, style)
    save_pic50_dist(row_df, figures_dir, style)
    save_endpoint_units_relations(row_df, figures_dir, style)
    save_missingness(comp_df, figures_dir, style)

    run_config = {
        "endpoint": args.endpoint,
        "threshold": args.threshold,
        "units_keep": args.units_keep,
        "relation_keep": args.relation_keep,
        "aggregate": args.aggregate,
        "prefer_pchembl": bool(args.prefer_pchembl),
        "plotting_params": {
            "font": args.font,
            "font_status": font_status,
            "bold_text": bool(args.bold_text),
            "palette": args.palette,
            "font_title": args.font_title,
            "font_label": args.font_label,
            "font_tick": args.font_tick,
            "font_legend": args.font_legend,
            "savefig.format": "svg",
            "svg.fonttype": "none",
        },
    }
    (provenance_dir / "run_config.json").write_text(json.dumps(run_config, indent=2), encoding="utf-8")

    metrics = load_metrics(summary_path)
    provenance = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command_line_args": sys.argv[1:],
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "font_status": font_status,
        "git_commit_hash": git_commit(),
        "script_sha256": {
            "scripts/qsar_postprocess.py": sha256_file(Path("scripts/qsar_postprocess.py")),
            "scripts/qsar_postprocess_report.py": sha256_file(Path("scripts/qsar_postprocess_report.py")),
        },
        "input_files_sha256": {
            str(row_path): sha256_file(row_path),
            str(comp_props_path): sha256_file(comp_props_path),
            str(summary_path): sha256_file(summary_path) if summary_path.exists() else None,
        },
        "counts": {
            "rows_before_filtering": metrics.get("rows_before_filtering"),
            "rows_after_endpoint_filter": metrics.get("rows_after_endpoint_filter"),
            "rows_after_units_relations_filter": metrics.get("rows_after_units_relations_filter"),
            "rows_after_pic50": metrics.get("rows_after_pic50"),
        },
    }
    (provenance_dir / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    write_environment(provenance_dir / "environment.txt")

    print(f"Wrote figures to {figures_dir} and provenance to {provenance_dir}")


if __name__ == "__main__":
    main()
