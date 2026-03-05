#!/usr/bin/env python
from __future__ import annotations

import csv
from pathlib import Path

import requests

API_URL = "https://www.ebi.ac.uk/chembl/api/data/activity"
OUTPUT_CSV = Path("data/external/raw/ptp1b_inhibition_chembl335.csv")
PAGE_LIMIT = 1000
COLUMNS = [
    "molecule_chembl_id",
    "canonical_smiles",
    "standard_value",
    "standard_units",
    "standard_type",
    "assay_chembl_id",
    "target_chembl_id",
    "document_chembl_id",
]


def fetch_all_activities() -> list[dict]:
    params = {
        "target_chembl_id": "CHEMBL335",
        "standard_type": "IC50",
        "standard_units": "nM",
        "limit": PAGE_LIMIT,
        "format": "json",
    }
    records: list[dict] = []
    next_url: str | None = API_URL

    with requests.Session() as session:
        session.trust_env = False
        while next_url:
            response = session.get(next_url, params=params if next_url == API_URL else None, timeout=60)
            response.raise_for_status()
            payload = response.json()
            records.extend(payload.get("activities", []))
            next_url = (payload.get("page_meta") or {}).get("next")
            params = None

    return records


def valid_row(row: dict) -> bool:
    smiles = row.get("canonical_smiles")
    value = row.get("standard_value")
    return smiles not in (None, "") and value not in (None, "")


def main() -> None:
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    records = [row for row in fetch_all_activities() if valid_row(row)]

    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()
        for row in records:
            writer.writerow({key: row.get(key) for key in COLUMNS})

    print(f"Downloaded {len(records)} inhibition records")


if __name__ == "__main__":
    main()
