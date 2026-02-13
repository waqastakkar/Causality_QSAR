from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path
from typing import Any

from manuscript_mapping import default_mapping
from provenance_utils import (
    basename_run_id,
    collect_tool_versions,
    file_size,
    find_first_existing,
    git_commit_or_unknown,
    recursive_collect_by_suffix,
    safe_load_config,
    sha256_file,
    try_read_csv_rows,
    utc_now_iso,
    write_csv,
    write_environment_txt,
    write_json,
)

NATURE5 = ["#E69F00", "#009E73", "#0072B2", "#D55E00", "#CC79A7"]


def str2bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in {"1", "true", "t", "yes", "y"}:
        return True
    if s in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {v}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 15 manuscript pack builder")
    p.add_argument("--paper_id", required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--run_dir", required=True)
    p.add_argument("--interpret_dir", required=True)
    p.add_argument("--robust_dir", required=False, default="")
    p.add_argument("--cross_endpoint_dir", required=False, default="")
    p.add_argument("--screen_dir", required=False, default="")
    p.add_argument("--screen_analysis_dir", required=False, default="")
    p.add_argument("--screen_match_dir", required=False, default="")
    p.add_argument("--outdir", required=True)
    p.add_argument("--copy_only", type=str2bool, default=True)
    p.add_argument("--svg_only", type=str2bool, default=True)
    p.add_argument("--export_tables_csv", type=str2bool, default=True)
    p.add_argument("--export_tables_xlsx", type=str2bool, default=True)
    p.add_argument("--font", default="Times New Roman")
    p.add_argument("--bold_text", action="store_true")
    p.add_argument("--palette", default="nature5")
    return p.parse_args()


def find_candidate(base_dir: Path, candidates: list[str]) -> Path | None:
    expanded: list[Path] = []
    for rel in candidates:
        candidate = base_dir / rel
        if "*" in rel or "?" in rel or "[" in rel:
            expanded.extend(sorted(base_dir.glob(rel)))
        else:
            expanded.append(candidate)
    return find_first_existing(expanded)


def gather_discovered_artifacts(source_dirs: dict[str, Path | None]) -> dict[str, list[str]]:
    found: dict[str, list[str]] = {"figures_svg": [], "reports_csv": [], "selections_csv": [], "parquet": []}
    for p in source_dirs.values():
        if p is None or not p.exists():
            continue
        found["figures_svg"].extend([str(x) for x in p.glob("figures/*.svg")])
        found["reports_csv"].extend([str(x) for x in p.glob("reports/*.csv")])
        found["selections_csv"].extend([str(x) for x in p.glob("selections/*.csv")])
        found["parquet"].extend([str(x) for x in recursive_collect_by_suffix(p, (".parquet",))])
    return found


def copy_artifact(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def read_table_rows(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            next(reader)
        except StopIteration:
            return 0
        return sum(1 for _ in reader)


def extract_seeds_splits_ablations(run_dir: Path, robust_dir: Path) -> dict[str, list[str]]:
    seeds: set[str] = set()
    splits: set[str] = set()
    ablations: set[str] = set()

    cfg_candidates = [run_dir / "configs" / "resolved_config.yaml", run_dir / "configs" / "resolved_config.json"]
    for cfg in cfg_candidates:
        data = safe_load_config(cfg)
        if not data:
            continue
        stack = [data]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                for k, v in item.items():
                    lk = str(k).lower()
                    if "seed" in lk and isinstance(v, (int, str)):
                        seeds.add(str(v))
                    if "split" in lk and isinstance(v, str):
                        splits.add(v)
                    if "ablation" in lk and isinstance(v, str):
                        ablations.add(v)
                    if isinstance(v, (dict, list)):
                        stack.append(v)
            elif isinstance(item, list):
                stack.extend(item)

    if len(run_dir.parts) >= 3 and run_dir.parts[-3] == "runs":
        splits.add(run_dir.parent.name)

    for csv_name in ["runs_index.csv", "groups_index.csv"]:
        rows = try_read_csv_rows(robust_dir / csv_name)
        for row in rows:
            for k, v in row.items():
                if not v:
                    continue
                lk = k.lower()
                if "seed" in lk:
                    seeds.add(v)
                if "split" in lk:
                    splits.add(v)
                if "ablation" in lk or "group" in lk:
                    ablations.add(v)

    return {
        "seeds": sorted(seeds, key=lambda x: (len(x), x)),
        "splits": sorted(splits),
        "ablations": sorted(ablations),
    }


def export_xlsx(table_manifest: list[dict[str, Any]], xlsx_path: Path) -> str:
    try:
        from openpyxl import Workbook  # type: ignore

        wb = Workbook()
        wb.remove(wb.active)
        for row in table_manifest:
            if row["status"] != "present":
                continue
            src = Path(row["dest_path"])
            if not src.exists():
                continue
            ws = wb.create_sheet(title=row["table_id"][:31])
            with src.open("r", encoding="utf-8", newline="") as fh:
                for csv_row in csv.reader(fh):
                    ws.append(csv_row)
        wb.save(xlsx_path)
        return "ok"
    except Exception as exc:
        try:
            import pandas as pd  # type: ignore

            with pd.ExcelWriter(xlsx_path) as writer:
                for row in table_manifest:
                    if row["status"] != "present":
                        continue
                    src = Path(row["dest_path"])
                    if not src.exists():
                        continue
                    df = pd.read_csv(src)
                    sheet = row["table_id"][:31]
                    df.to_excel(writer, sheet_name=sheet, index=False)
            return "ok"
        except Exception as exc2:
            return f"failed: {exc}; fallback_failed: {exc2}"


def build_checklist(
    checklist_path: Path,
    args: argparse.Namespace,
    fig_manifest_path: Path,
    table_manifest_path: Path,
    fig_rows: list[dict[str, Any]],
    table_rows: list[dict[str, Any]],
    source_dirs: dict[str, Path | None],
    config_hash: str,
    checkpoint_hashes: dict[str, str],
    dataset_hash: str,
    screening_fingerprint_hash: str,
    git_commit: str,
) -> None:
    run_id = basename_run_id(args.run_dir)
    timestamp = utc_now_iso()
    seeds_splits = extract_seeds_splits_ablations(Path(args.run_dir), Path(args.robust_dir) if args.robust_dir else Path("/__missing__"))

    missing: list[str] = []
    suspicious: list[str] = []

    lines: list[str] = ["## Reproducibility Fingerprint", "```json"]
    block = {
        "paper_id": args.paper_id,
        "target": args.target,
        "run_dir": args.run_dir,
        "run_id": run_id,
        "robust_dir": args.robust_dir or None,
        "screen_dir": args.screen_dir or None,
        "cross_endpoint_dir": args.cross_endpoint_dir or None,
        "interpret_dir": args.interpret_dir or None,
        "git_commit": git_commit,
        "timestamp_utc": timestamp,
        "sha256": {
            "figure_manifest_csv": sha256_file(fig_manifest_path),
            "table_manifest_csv": sha256_file(table_manifest_path),
            "run_config": config_hash,
            "checkpoints": checkpoint_hashes,
            "dataset_parquet": dataset_hash,
            "screen_input_fingerprint": screening_fingerprint_hash,
        },
    }
    lines += [json.dumps(block, indent=2), "```", "", "## Artifact Presence Checklist", ""]
    lines += ["| Artifact | Status | Notes |", "|---|---|---|"]

    def status_for_row(row: dict[str, Any], is_figure: bool) -> tuple[str, str]:
        if row["status"] != "present":
            missing.append(f"{row.get('fig_id') or row.get('table_id')}")
            return "❌ missing", "not found"
        dest = Path(row["dest_path"])
        if file_size(dest) == 0:
            suspicious.append(f"{row.get('fig_id') or row.get('table_id')}")
            return "⚠️ suspicious", "empty file"
        if is_figure:
            txt = dest.read_text(encoding="utf-8", errors="ignore")
            if dest.suffix.lower() != ".svg":
                suspicious.append(row["fig_id"])
                return "⚠️ suspicious", "not svg"
            if "<text" not in txt.lower():
                suspicious.append(row["fig_id"])
                return "⚠️ suspicious", "no <text> element detected"
        else:
            nrows = read_table_rows(dest)
            if nrows == 0:
                suspicious.append(row["table_id"])
                return "⚠️ suspicious", "0 data rows"
        return "✅ present", "ok"

    for row in fig_rows:
        status, note = status_for_row(row, is_figure=True)
        lines.append(f"| {row['fig_id']} ({row['fig_title']}) | {status} | {note} |")
    for row in table_rows:
        status, note = status_for_row(row, is_figure=False)
        lines.append(f"| {row['table_id']} ({row['table_title']}) | {status} | {note} |")

    missing_main_fig = [r["fig_id"] for r in fig_rows if r["category"] == "main" and r["status"] != "present"]
    missing_supp_fig = [r["fig_id"] for r in fig_rows if r["category"] == "supp" and r["status"] != "present"]
    missing_main_tab = [r["table_id"] for r in table_rows if r["category"] == "main" and r["status"] != "present"]
    missing_supp_tab = [r["table_id"] for r in table_rows if r["category"] == "supp" and r["status"] != "present"]

    lines += [
        "",
        "## Missing",
        f"- Missing required main figures: {missing_main_fig if missing_main_fig else 'none'}",
        f"- Missing supplementary figures: {missing_supp_fig if missing_supp_fig else 'none'}",
        f"- Missing required main tables: {missing_main_tab if missing_main_tab else 'none'}",
        f"- Missing supplementary tables: {missing_supp_tab if missing_supp_tab else 'none'}",
        "",
        "## Seeds, Splits, Ablations Included",
        f"- splits: {seeds_splits['splits'] if seeds_splits['splits'] else ['unknown']}",
        f"- seeds: {seeds_splits['seeds'] if seeds_splits['seeds'] else ['unknown']}",
        f"- ablations: {seeds_splits['ablations'] if seeds_splits['ablations'] else ['none-detected']}",
        "",
        "## Source Discovery Summary",
    ]

    for k, v in source_dirs.items():
        lines.append(f"- {k}: {v}")

    checklist_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    dirs = {
        "main_figures": outdir / "main_figures",
        "supp_figures": outdir / "supp_figures",
        "main_tables": outdir / "main_tables",
        "supp_tables": outdir / "supp_tables",
        "manifests": outdir / "manifests",
        "assets": outdir / "assets",
        "provenance": outdir / "provenance",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    source_dirs: dict[str, Path | None] = {
        "run_dir": Path(args.run_dir),
        "interpret_dir": Path(args.interpret_dir),
        "robust_dir": Path(args.robust_dir) if args.robust_dir else None,
        "cross_endpoint_dir": Path(args.cross_endpoint_dir) if args.cross_endpoint_dir else None,
        "screen_dir": Path(args.screen_dir) if args.screen_dir else None,
        "screen_analysis_dir": Path(args.screen_analysis_dir) if args.screen_analysis_dir else None,
        "screen_match_dir": Path(args.screen_match_dir) if args.screen_match_dir else None,
    }

    discovered = gather_discovered_artifacts(source_dirs)
    mapping = default_mapping()
    figure_rows: list[dict[str, Any]] = []
    table_rows: list[dict[str, Any]] = []

    for spec in mapping["figures"]:
        source_base = source_dirs.get(spec["source_key"], Path(""))
        src = find_candidate(source_base, spec["candidates"]) if source_base is not None else None
        category_dir = dirs["main_figures"] if spec["category"] == "main" else dirs["supp_figures"]
        dst = category_dir / spec["dest_name"]
        status = "missing"
        sha = ""
        src_path = ""
        if src and src.exists() and (not args.svg_only or src.suffix.lower() == ".svg"):
            src_path = str(src)
            copy_artifact(src, dst)
            status = "present"
            sha = sha256_file(dst)
        figure_rows.append(
            {
                "fig_id": spec["id"],
                "fig_title": spec["title"],
                "category": spec["category"],
                "source_path": src_path,
                "dest_path": str(dst) if status == "present" else "",
                "step_origin": spec["step_origin"],
                "status": status,
                "sha256": sha,
            }
        )

    for spec in mapping["tables"]:
        source_base = source_dirs.get(spec["source_key"], Path(""))
        src = find_candidate(source_base, spec["candidates"]) if source_base is not None else None
        category_dir = dirs["main_tables"] if spec["category"] == "main" else dirs["supp_tables"]
        dst = category_dir / spec["dest_name"]
        status = "missing"
        sha = ""
        src_path = ""
        if src and src.exists() and args.export_tables_csv:
            src_path = str(src)
            copy_artifact(src, dst)
            status = "present"
            sha = sha256_file(dst)
        table_rows.append(
            {
                "table_id": spec["id"],
                "table_title": spec["title"],
                "category": spec["category"],
                "source_path": src_path,
                "dest_path": str(dst) if status == "present" else "",
                "step_origin": spec["step_origin"],
                "status": status,
                "sha256": sha,
            }
        )

    fig_manifest = dirs["manifests"] / "figure_manifest.csv"
    table_manifest = dirs["manifests"] / "table_manifest.csv"
    write_csv(
        fig_manifest,
        figure_rows,
        ["fig_id", "fig_title", "category", "source_path", "dest_path", "step_origin", "status", "sha256"],
    )
    write_csv(
        table_manifest,
        table_rows,
        ["table_id", "table_title", "category", "source_path", "dest_path", "step_origin", "status", "sha256"],
    )

    citations = dirs["manifests"] / "citations_sources.txt"
    citations.write_text(
        "\n".join([r["source_path"] for r in figure_rows + table_rows if r["source_path"]]) + "\n",
        encoding="utf-8",
    )

    style_contract = dirs["assets"] / "style_contract.txt"
    style_contract.write_text(
        "Style Contract\n"
        "- SVG only figures\n"
        "- Editable text in SVG (fonttype none expected)\n"
        f"- Font: {args.font}\n"
        f"- Bold text: {'enabled' if args.bold_text else 'disabled'}\n"
        "- Palette: Nature 5 (#E69F00,#009E73,#0072B2,#D55E00,#CC79A7)\n"
        "- figure_data artifacts retained from source steps\n"
        "- Reproducibility manifests and provenance required\n",
        encoding="utf-8",
    )
    write_json(dirs["assets"] / "nature_palette.json", {"name": "nature5", "colors": NATURE5})

    run_cfg = source_dirs["run_dir"] / "configs" / "resolved_config.yaml"  # type: ignore[operator]
    if not run_cfg.exists():
        run_cfg = source_dirs["run_dir"] / "configs" / "resolved_config.json"  # type: ignore[operator]
    run_config_hash = sha256_file(run_cfg)
    if run_cfg.exists():
        shutil.copy2(run_cfg, dirs["provenance"] / "run_config.json")
    else:
        write_json(dirs["provenance"] / "run_config.json", {})

    ckpts = sorted(source_dirs["run_dir"].glob("checkpoints/*.pt"))  # type: ignore[union-attr]
    checkpoint_hashes = {str(p): sha256_file(p) for p in ckpts}

    dataset_candidate = Path(f"data/processed/environments/{args.target}/data/multienv_compound_level.parquet")
    dataset_hash = sha256_file(dataset_candidate)
    screen_fp = source_dirs["screen_dir"] / "input" / "input_fingerprint.json" if args.screen_dir and source_dirs["screen_dir"] is not None else Path("/__missing__")
    screen_fp_hash = sha256_file(screen_fp)

    prov_manifest = {
        "paper_id": args.paper_id,
        "target": args.target,
        "run_id": basename_run_id(args.run_dir),
        "timestamp_utc": utc_now_iso(),
        "git_commit": git_commit_or_unknown(Path.cwd()),
        "tool_versions": collect_tool_versions(),
        "key_input_hashes": {
            "run_config": run_config_hash,
            "checkpoints": checkpoint_hashes,
            "dataset_parquet": dataset_hash,
            "screening_input_fingerprint": screen_fp_hash,
        },
        "discovered_artifacts": discovered,
    }
    write_json(dirs["manifests"] / "provenance_manifest.json", prov_manifest)

    xlsx_status = "not-requested"
    if args.export_tables_xlsx:
        xlsx_status = export_xlsx(table_rows, dirs["main_tables"] / "manuscript_tables.xlsx")

    write_environment_txt(dirs["provenance"] / "environment.txt")
    full_provenance = {
        "cli_args": vars(args),
        "timestamp_utc": utc_now_iso(),
        "git_commit": prov_manifest["git_commit"],
        "copied_figure_hashes": {r["fig_id"]: r["sha256"] for r in figure_rows if r["status"] == "present"},
        "copied_table_hashes": {r["table_id"]: r["sha256"] for r in table_rows if r["status"] == "present"},
        "xlsx_export_status": xlsx_status,
    }
    write_json(dirs["provenance"] / "provenance.json", full_provenance)

    build_checklist(
        checklist_path=dirs["manifests"] / "manuscript_checklist.md",
        args=args,
        fig_manifest_path=fig_manifest,
        table_manifest_path=table_manifest,
        fig_rows=figure_rows,
        table_rows=table_rows,
        source_dirs=source_dirs,
        config_hash=run_config_hash,
        checkpoint_hashes=checkpoint_hashes,
        dataset_hash=dataset_hash,
        screening_fingerprint_hash=screen_fp_hash,
        git_commit=prov_manifest["git_commit"],
    )


if __name__ == "__main__":
    main()
