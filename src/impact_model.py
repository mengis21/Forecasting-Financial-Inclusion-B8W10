from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


def _sign(direction: str | float | None) -> float:
    if direction is None or (isinstance(direction, float) and np.isnan(direction)):
        return 1.0
    d = str(direction).strip().lower()
    if d in {"positive", "+", "increase", "higher"}:
        return 1.0
    if d in {"negative", "-", "decrease", "lower"}:
        return -1.0
    return 1.0


@dataclass(frozen=True)
class EventIndicatorMatrix:
    long_table: pd.DataFrame
    matrix: pd.DataFrame


def build_event_indicator_matrix(
    unified_df: pd.DataFrame,
    impact_links_df: pd.DataFrame | None = None,
) -> EventIndicatorMatrix:
    """Build an event–indicator association matrix.

    Notes:
    - We keep events pillar-empty (unbiased), and apply pillars on links.
    - The numeric matrix uses: sign(impact_direction) * impact_magnitude.
    """
    events = unified_df[unified_df["record_type"] == "event"].copy()

    if impact_links_df is None:
        links = unified_df[unified_df["record_type"] == "impact_link"].copy()
    else:
        links = impact_links_df.copy()

    # Normalize link parent_id column naming: some exports use relationship_type/parent_id.
    parent_col = None
    for c in ["parent_id", "relationship_type"]:
        if c in links.columns:
            parent_col = c
            break
    if parent_col is None:
        raise ValueError("No parent_id-like column found for impact links")

    links = links.rename(columns={parent_col: "parent_id"})

    mag = pd.Series(np.nan, index=links.index)
    if "impact_estimate" in links.columns:
        mag = pd.to_numeric(links["impact_estimate"], errors="coerce")
    if "impact_magnitude" in links.columns:
        mag2 = pd.to_numeric(links["impact_magnitude"], errors="coerce")
        mag = mag.fillna(mag2)

    links = links.assign(
        impact_signed=mag * links.get("impact_direction").apply(_sign),
        lag_months=pd.to_numeric(links.get("lag_months"), errors="coerce"),
    )

    # Join links -> events to get event details.
    events_small = events[["record_id", "category", "observation_date", "source_name", "notes"]].rename(
        columns={
            "record_id": "event_id",
            "category": "event_category",
            "observation_date": "event_date",
        }
    )

    joined = links.merge(
        events_small,
        how="left",
        left_on="parent_id",
        right_on="event_id",
        validate="m:1",
    )

    # Build a clean long table.
    long_cols = {
        "event_id": "event_id",
        "event_category": "event_category",
        "event_date": "event_date",
        "pillar": "pillar",
        "related_indicator": "related_indicator",
        "impact_direction": "impact_direction",
        "impact_magnitude": "impact_magnitude",
        "impact_signed": "impact_signed",
        "lag_months": "lag_months",
        "evidence_basis": "evidence_basis",
        "comparable_country": "comparable_country",
    }

    long_df = joined[list(long_cols.keys())].rename(columns=long_cols)

    # Matrix indexed by event_id, columns by related_indicator.
    matrix = (
        long_df.pivot_table(
            index=["event_id", "event_category", "event_date"],
            columns="related_indicator",
            values="impact_signed",
            aggfunc="sum",
        )
        .sort_index()
        .sort_index(axis=1)
    )

    return EventIndicatorMatrix(long_table=long_df.sort_values(["event_date", "event_id"]), matrix=matrix)


def export_event_indicator_matrix(
    unified_df: pd.DataFrame,
    impact_links_df: pd.DataFrame | None = None,
    *,
    out_dir: Path,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    res = build_event_indicator_matrix(unified_df, impact_links_df)

    matrix_csv = out_dir / "event_indicator_matrix.csv"
    long_csv = out_dir / "event_indicator_links_long.csv"

    res.matrix.to_csv(matrix_csv)
    res.long_table.to_csv(long_csv, index=False)

    return {"matrix_csv": matrix_csv, "long_csv": long_csv}
