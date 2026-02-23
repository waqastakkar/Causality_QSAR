#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
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


def extract_rules(smiles: list[str], max_cut_bonds: int) -> tuple[list[tuple[str, str, str]], int]:
    from rdkit import Chem
    from rdkit.Chem import rdMMPA

    by_core: dict[str, list[str]] = defaultdict(list)
    n_valid_molecules = 0
    for smi in smiles:
        mol = Chem.MolFromSmiles(smi) if isinstance(smi, str) else None
        if mol is None:
            continue
        n_valid_molecules += 1
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
    return rules, n_valid_molecules


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
    try:
        from rdkit.Chem import rdMMPA  # noqa: F401
    except Exception as e:
        raise SystemExit(
            "RDKit rdMMPA is not available. MMP rule extraction cannot run. "
            "Install a full RDKit build that includes rdMMPA, or use a fallback rule generator."
        ) from e

    outdir = Path(args.outdir)
    rules_dir = outdir / "rules"
    fig_dir = outdir / "figures"
    rules_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.input_parquet)
    smi_col = pick_col(df, ["canonical_smiles", "smiles"])
    p_col = pick_col(df, ["pIC50", "pic50", "y"], required=False)
    series_col = pick_col(df, ["series_id", "series", "scaffold_id"], required=False)

    input_smiles = df[smi_col].dropna().astype(str).tolist()
    rules_raw, n_valid_molecules = extract_rules(input_smiles, args.max_cut_bonds)
    print(
        "Rule extraction diagnostics: "
        f"n_molecules_loaded={len(input_smiles)}, "
        f"n_valid_rdkit_molecules={n_valid_molecules}, "
        f"n_candidate_pairs_pre_aggregation={len(rules_raw)}",
        file=sys.stderr,
    )
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

    rows = build_rows(args.min_support)
    n_rules_raw = len(counter)
    n_rules_after_min_support = len(rows)
    effective_min_support = args.min_support
    fallback_applied = False
    fallback_message = ""
    if n_rules_raw == 0:
        warnings.warn(
            "n_rules_raw=0. No raw MMP pairs were extracted. "
            "This is not fixed by min_support. Check RDKit MMPA availability and input standardization."
        )

    print(
        "Rule extraction counts: "
        f"n_rules_raw={n_rules_raw}, "
        f"n_rules_after_min_support={n_rules_after_min_support}, "
        f"min_support_used={effective_min_support}",
        file=sys.stderr,
    )
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
                "n_rules_raw",
                "n_rules_after_min_support",
                "requested_min_support",
                "min_support_used",
                "effective_min_support",
                "fallback_applied",
                "fallback_note",
                "max_cut_bonds",
            ],
            "value": [
                len(df),
                df[smi_col].nunique(),
                n_rules_raw,
                len(rules_df),
                n_rules_raw,
                n_rules_after_min_support,
                args.min_support,
                effective_min_support,
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

    fig_path = fig_dir / "fig_edit_type_distribution.svg"
    top = rules_df.head(15).copy()
    if top.empty:
        fig, ax = plt.subplots(figsize=(4, 2.4))
        ax.axis("off")
        ax.text(0.5, 0.5, "NO DATA (0 rows)", ha="center", va="center", fontsize=14, fontweight="bold")
        fig.tight_layout()
        fig.savefig(fig_path)
        plt.close(fig)
    else:
        fig, ax = plt.subplots(figsize=(7, 4))
        top["rule_label"] = top["lhs_fragment"].str.slice(0, 12) + "→" + top["rhs_fragment"].str.slice(0, 12)
        ax.barh(top["rule_label"], top["support_count"])
        style_axis(ax, style, title="Edit Type Distribution", xlabel="Support count", ylabel="Rule")
        fig.tight_layout()
        fig.savefig(fig_path)
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
