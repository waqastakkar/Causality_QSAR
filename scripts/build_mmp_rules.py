#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from plot_style import add_plot_style_args, configure_matplotlib, style_axis, style_from_args


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pick_col(df: pd.DataFrame, choices: list[str], required: bool = True) -> str | None:
    lc = {c.lower(): c for c in df.columns}
    for c in choices:
        if c.lower() in lc:
            return lc[c.lower()]
    if required:
        raise ValueError(f"Missing required columns. expected one of: {choices}")
    return None


def extract_rules(smiles: list[str], max_cut_bonds: int) -> list[tuple[str, str, str]]:
    try:
        from rdkit import Chem
        from rdkit.Chem import rdMMPA
    except Exception:
        # fallback lexical edit rules
        out: list[tuple[str, str, str]] = []
        for s in smiles:
            if isinstance(s, str) and len(s) > 4:
                out.append((s[:-1], s[-1:], "LEXICAL"))
        return out

    by_core: dict[str, list[str]] = defaultdict(list)
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        for core, chains in rdMMPA.FragmentMol(mol, maxCuts=max_cut_bonds, maxCutBonds=max_cut_bonds, resultsAsMols=False):
            if not core or not chains:
                continue
            side = chains.split(".")[0]
            by_core[core].append(side)

    rules: list[tuple[str, str, str]] = []
    for core, sides in by_core.items():
        uniq = sorted(set(sides))
        for i in range(len(uniq)):
            for j in range(len(uniq)):
                if i == j:
                    continue
                rules.append((uniq[i], uniq[j], core))
    return rules


def main() -> None:
    parser = argparse.ArgumentParser(description="Build MMP transformation rules for counterfactual generation.")
    parser.add_argument("--target", required=True)
    parser.add_argument("--input_parquet", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--min_support", type=int, default=3)
    parser.add_argument("--max_cut_bonds", type=int, default=1)
    add_plot_style_args(parser)
    args = parser.parse_args()
    try:
        import pandas as pd
    except Exception as exc:
        raise SystemExit("pandas is required to run build_mmp_rules.py") from exc

    outdir = Path(args.outdir)
    rules_dir = outdir / "rules"
    fig_dir = outdir / "figures"
    rules_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_parquet)
    smi_col = pick_col(df, ["canonical_smiles", "smiles"])
    p_col = pick_col(df, ["pIC50", "pic50", "y"], required=False)
    series_col = pick_col(df, ["series_id", "series", "scaffold_id"], required=False)

    rules_raw = extract_rules(df[smi_col].dropna().astype(str).tolist(), args.max_cut_bonds)
    counter = Counter((lhs, rhs, ctx) for lhs, rhs, ctx in rules_raw)

    def build_rows(min_support: int) -> list[dict[str, object]]:
        filtered_rows = []
        for idx, ((lhs, rhs, ctx), support) in enumerate(counter.items(), start=1):
            if support < min_support:
                continue
            filtered_rows.append(
                {
                    "rule_id": f"R{idx:06d}",
                    "lhs_fragment": lhs,
                    "rhs_fragment": rhs,
                    "context_fragment": ctx,
                    "transformation": f"{lhs}>>{rhs}",
                    "support_count": int(support),
                    "median_delta_pIC50": 0.0,
                    "series_scope": "global" if series_col is None else "series_aware",
                }
            )
        return filtered_rows

    support_attempts = [10] if args.min_support <= 3 else [args.min_support]
    if args.min_support not in support_attempts:
        support_attempts.append(args.min_support)

    rows = []
    effective_min_support = support_attempts[0]
    fallback_applied = False
    fallback_message = ""
    for i, support_threshold in enumerate(support_attempts):
        rows = build_rows(support_threshold)
        effective_min_support = support_threshold
        if rows or i == len(support_attempts) - 1:
            break

    if not rows and effective_min_support > 3:
        # Safety net for sparse/diverse datasets when caller chooses a high threshold.
        fallback_applied = True
        effective_min_support = 3
        fallback_message = (
            f"No rules found with min_support={args.min_support}; automatically rebuilt with min_support=3."
        )
        warnings.warn(fallback_message)
        rows = build_rows(effective_min_support)
    elif effective_min_support != args.min_support:
        fallback_applied = True
        fallback_message = (
            f"No rules found with min_support={support_attempts[0]}; automatically rebuilt with min_support={effective_min_support}."
        )
        warnings.warn(fallback_message)
    rules_df = pd.DataFrame(rows).sort_values("support_count", ascending=False) if rows else pd.DataFrame(columns=[
        "rule_id", "lhs_fragment", "rhs_fragment", "context_fragment", "transformation", "support_count", "median_delta_pIC50", "series_scope"
    ])

    rules_path = rules_dir / "mmp_rules.parquet"
    stats_path = rules_dir / "rule_stats.csv"
    prov_path = rules_dir / "rule_provenance.json"
    rules_df.to_parquet(rules_path, index=False)

    stats = pd.DataFrame(
        {
            "metric": [
                "n_input_rows",
                "n_unique_smiles",
                "n_rules_before_filter",
                "n_rules_after_filter",
                "requested_min_support",
                "effective_min_support",
                "fallback_applied",
                "fallback_note",
                "max_cut_bonds",
            ],
            "value": [
                len(df),
                df[smi_col].nunique(),
                len(counter),
                len(rules_df),
                args.min_support,
                effective_min_support,
                int(fallback_applied),
                fallback_message or "none",
                args.max_cut_bonds,
            ],
        }
    )
    stats.to_csv(stats_path, index=False)

    style = style_from_args(args)
    configure_matplotlib(style, svg=True)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    top = rules_df.head(15).copy()
    if not top.empty:
        top["rule_label"] = top["lhs_fragment"].str.slice(0, 12) + "→" + top["rhs_fragment"].str.slice(0, 12)
        ax.barh(top["rule_label"], top["support_count"])
    style_axis(ax, style, title="Edit Type Distribution", xlabel="Support count", ylabel="Rule")
    fig.tight_layout()
    fig.savefig(fig_dir / "fig_edit_type_distribution.svg")
    plt.close(fig)

    provenance = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "cli_args": vars(args),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "git_commit": subprocess.getoutput("git rev-parse HEAD"),
        "script_hash": sha256_file(Path(__file__)),
        "input_dataset_hash": sha256_file(Path(args.input_parquet)),
        "n_rules": int(len(rules_df)),
        "requested_min_support": int(args.min_support),
        "effective_min_support": int(effective_min_support),
        "fallback_applied": bool(fallback_applied),
        "fallback_note": fallback_message or None,
    }
    prov_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"Wrote rules to {rules_path}")


if __name__ == "__main__":
    main()
