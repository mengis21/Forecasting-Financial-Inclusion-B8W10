from __future__ import annotations

import argparse
import json
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


@dataclass(frozen=True)
class EnrichmentEntry:
    change_type: str  # "new_record" | "field_update"
    record_id: str
    record_type: str
    indicator_code: str | None
    indicator: str | None
    observation_date: str | None
    field: str | None
    old_value: str | None
    new_value: str | None
    source_name: str | None
    source_url: str | None
    notes: str | None


def _read_excel_sheet(path: Path, sheet_name: str) -> pd.DataFrame:
    return pd.read_excel(path, sheet_name=sheet_name)


def export_raw_csvs(raw_dir: Path) -> dict[str, Path]:
    """Export starter Excel sheets into CSVs under data/raw.

    Returns a dict of output names -> paths.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)

    unified_xlsx = raw_dir / "ethiopia_fi_unified_data.xlsx"
    ref_xlsx = raw_dir / "reference_codes.xlsx"

    unified_df = _read_excel_sheet(unified_xlsx, "ethiopia_fi_unified_data")
    impact_df = _read_excel_sheet(unified_xlsx, "Impact_sheet")
    ref_df = _read_excel_sheet(ref_xlsx, "reference_codes")

    out_unified = raw_dir / "ethiopia_fi_unified_data.csv"
    out_impact = raw_dir / "impact_links.csv"
    out_ref = raw_dir / "reference_codes.csv"

    unified_df.to_csv(out_unified, index=False)
    impact_df.to_csv(out_impact, index=False)
    ref_df.to_csv(out_ref, index=False)

    return {
        "unified": out_unified,
        "impact_links": out_impact,
        "reference_codes": out_ref,
    }


def _world_bank_wdi_fetch(country_code: str, indicator_code: str) -> list[dict[str, Any]]:
    base = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}"
    url = base.format(country=urllib.parse.quote(country_code), indicator=urllib.parse.quote(indicator_code))
    url += "?format=json&per_page=20000"

    req = urllib.request.Request(url, headers={"User-Agent": "ethiopia-fi-forecast/1.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read().decode("utf-8"))

    # Response is [metadata, records]
    if not isinstance(data, list) or len(data) < 2 or not isinstance(data[1], list):
        raise ValueError("Unexpected World Bank API response shape")

    return data[1]


def fetch_mobile_subscriptions_per_100(country_code: str = "ETH") -> pd.DataFrame:
    """Fetch IT.CEL.SETS.P2 from World Bank WDI.

    Metric: Mobile cellular subscriptions (per 100 people)
    We treat this as a percent-like penetration measure.
    """
    records = _world_bank_wdi_fetch(country_code=country_code, indicator_code="IT.CEL.SETS.P2")
    rows: list[dict[str, Any]] = []

    for r in records:
        year = r.get("date")
        value = r.get("value")
        if year is None or value is None:
            continue
        try:
            year_int = int(year)
            value_float = float(value)
        except (TypeError, ValueError):
            continue

        rows.append({"year": year_int, "value": value_float})

    out = pd.DataFrame(rows).sort_values("year")
    return out


def _next_rec_id(existing_ids: Iterable[str]) -> str:
    max_n = 0
    for rid in existing_ids:
        m = re.search(r"^REC_(\d+)$", str(rid).strip())
        if m:
            max_n = max(max_n, int(m.group(1)))
    return f"REC_{max_n + 1:04d}"


def enrich_unified_dataset(
    unified_df: pd.DataFrame,
    *,
    collected_by: str = "aln_lvr",
    collection_date: date | None = None,
) -> tuple[pd.DataFrame, list[EnrichmentEntry]]:
    """Return an enriched copy of the unified dataset + an enrichment log."""
    if collection_date is None:
        collection_date = date.today()

    df = unified_df.copy()
    entries: list[EnrichmentEntry] = []

    # Enrichment 1: Add World Bank WDI mobile subscription penetration as extra observations
    wdi = fetch_mobile_subscriptions_per_100("ETH")
    wdi = wdi[(wdi["year"] >= 2014) & (wdi["year"] <= 2023)].copy()

    template_cols = list(df.columns)
    obs_year = pd.to_datetime(df.get("observation_date"), errors="coerce").dt.year
    existing_pairs = set(
        zip(
            df.get("indicator_code").fillna(""),
            obs_year.fillna(-1).astype(int),
        )
    )

    # Deterministic REC id allocation
    max_rec_n = 0
    for rid in df.get("record_id").tolist():
        m = re.search(r"^REC_(\d+)$", str(rid).strip())
        if m:
            max_rec_n = max(max_rec_n, int(m.group(1)))
    next_rec_n = max_rec_n + 1

    source_url = "https://api.worldbank.org/v2/country/ETH/indicator/IT.CEL.SETS.P2?format=json"
    new_rows: list[dict[str, Any]] = []
    for _, row in wdi.iterrows():
        year = int(row["year"])
        key = ("ACC_MOBILE_PEN", year)
        if key in existing_pairs:
            continue

        record_id = f"REC_{next_rec_n:04d}"
        next_rec_n += 1

        new_row = {c: pd.NA for c in template_cols}
        new_row.update(
            {
                "record_id": record_id,
                "record_type": "observation",
                "pillar": "ACCESS",
                "indicator": "Mobile Subscription Penetration",
                "indicator_code": "ACC_MOBILE_PEN",
                "indicator_direction": "higher_better",
                "value_numeric": float(row["value"]),
                "value_type": "percentage",
                "unit": "%",
                "observation_date": datetime(year, 12, 31),
                "gender": "all",
                "location": "national",
                "source_name": "World Bank WDI",
                "source_type": "api",
                "source_url": source_url,
                "confidence": "high",
                "collected_by": collected_by,
                "collection_date": pd.Timestamp(collection_date),
                "original_text": "World Bank WDI: Mobile cellular subscriptions (per 100 people) (IT.CEL.SETS.P2)",
                "notes": "Per-100-people subscriptions; not unique individuals.",
            }
        )

        new_rows.append(new_row)
        entries.append(
            EnrichmentEntry(
                change_type="new_record",
                record_id=record_id,
                record_type="observation",
                indicator_code="ACC_MOBILE_PEN",
                indicator="Mobile Subscription Penetration",
                observation_date=f"{year}-12-31",
                field=None,
                old_value=None,
                new_value=str(float(row["value"])),
                source_name="World Bank WDI",
                source_url=source_url,
                notes="Added WDI yearly series (2014–2023).",
            )
        )

    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)

    return df, entries


def write_enrichment_log_md(path: Path, entries: list[EnrichmentEntry]) -> None:
    lines: list[str] = []
    lines.append("# Data enrichment log\n")
    lines.append("This log lists **only the changes introduced by `src/build_dataset.py`**.\n")
    lines.append("\n## Changes\n")

    if not entries:
        lines.append("No enrichment changes were applied.\n")
        path.write_text("\n".join(lines), encoding="utf-8")
        return

    lines.append(
        "| change_type | record_id | record_type | indicator_code | observation_date | field | old_value | new_value | source_name | source_url | notes |"
    )
    lines.append(
        "|---|---|---|---|---|---|---|---|---|---|---|"
    )

    def _safe(v: Any) -> str:
        if v is None:
            return ""
        s = str(v)
        return s.replace("\n", " ").replace("|", "\\|")

    for e in entries:
        lines.append(
            "| "
            + " | ".join(
                [
                    _safe(e.change_type),
                    _safe(e.record_id),
                    _safe(e.record_type),
                    _safe(e.indicator_code),
                    _safe(e.observation_date),
                    _safe(e.field),
                    _safe(e.old_value),
                    _safe(e.new_value),
                    _safe(e.source_name),
                    _safe(e.source_url),
                    _safe(e.notes),
                ]
            )
            + " |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Build raw CSV exports and an enriched processed dataset.")
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the ethiopia-fi-forecast project directory",
    )
    parser.add_argument("--collected-by", type=str, default="aln_lvr")
    args = parser.parse_args()

    project_dir: Path = args.project_dir
    raw_dir = project_dir / "data" / "raw"
    processed_dir = project_dir / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    export_raw_csvs(raw_dir)

    unified_df = pd.read_csv(raw_dir / "ethiopia_fi_unified_data.csv")
    # Normalize known date columns for downstream work
    for col in ["observation_date", "period_start", "period_end", "collection_date"]:
        if col in unified_df.columns:
            unified_df.loc[:, col] = pd.to_datetime(unified_df[col], errors="coerce")

    enriched_df, entries = enrich_unified_dataset(unified_df, collected_by=args.collected_by)

    out_processed = processed_dir / "enriched_unified_data.csv"
    enriched_df.to_csv(out_processed, index=False)

    write_enrichment_log_md(project_dir / "data_enrichment_log.md", entries)

    print(f"Wrote: {out_processed}")
    print(f"Wrote: {project_dir / 'data_enrichment_log.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
