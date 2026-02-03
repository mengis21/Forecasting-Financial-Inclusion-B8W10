from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class DatasetPaths:
    raw_unified: Path
    raw_impact_links: Path
    raw_reference_codes: Path
    processed_enriched: Path


def get_default_paths(project_dir: Path | None = None) -> DatasetPaths:
    if project_dir is None:
        project_dir = Path(__file__).resolve().parents[1]

    return DatasetPaths(
        raw_unified=project_dir / "data" / "raw" / "ethiopia_fi_unified_data.csv",
        raw_impact_links=project_dir / "data" / "raw" / "impact_links.csv",
        raw_reference_codes=project_dir / "data" / "raw" / "reference_codes.csv",
        processed_enriched=project_dir / "data" / "processed" / "enriched_unified_data.csv",
    )


def load_enriched_unified(project_dir: Path | None = None) -> pd.DataFrame:
    paths = get_default_paths(project_dir)
    df = pd.read_csv(paths.processed_enriched)
    df = _standardize_dates(df)
    return df


def load_raw_impact_links(project_dir: Path | None = None) -> pd.DataFrame:
    paths = get_default_paths(project_dir)
    df = pd.read_csv(paths.raw_impact_links)
    df = _standardize_dates(df)
    return df


def _standardize_dates(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for c in ["observation_date", "period_start", "period_end", "collection_date"]:
        if c in out.columns:
            out[c] = pd.to_datetime(out[c], errors="coerce")
    return out
