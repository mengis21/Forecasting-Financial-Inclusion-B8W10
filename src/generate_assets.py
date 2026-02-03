from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from src.data import load_enriched_unified, load_raw_impact_links
from src.forecasting import (
    build_forecast_table,
    forecast_access_account_ownership,
    forecast_usage_p2p_count,
)
from src.impact_model import build_event_indicator_matrix


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _save_fig(fig: go.Figure, path: Path) -> None:
    fig.write_image(str(path), scale=2)


def _access_forecast_fig(all_fc: pd.DataFrame) -> go.Figure:
    df = all_fc[all_fc["indicator_code"] == "ACC_OWNERSHIP"].copy()
    fig = px.line(
        df,
        x="year",
        y="forecast_value",
        color="scenario",
        markers=True,
        title="Forecast — Account Ownership Rate (Access)",
        labels={"forecast_value": "% of adults"},
    )
    fig.update_layout(legend_title_text="Scenario")
    return fig


def _usage_forecast_fig(all_fc: pd.DataFrame) -> go.Figure:
    df = all_fc[all_fc["indicator_code"] == "USG_P2P_COUNT"].copy()
    fig = px.line(
        df,
        x="year",
        y="forecast_value",
        color="scenario",
        markers=True,
        title="Forecast — P2P Transactions (Usage proxy)",
        labels={"forecast_value": "transactions"},
    )
    fig.update_layout(legend_title_text="Scenario")
    return fig


def _event_indicator_heatmap(matrix: pd.DataFrame) -> go.Figure:
    # Flatten index for labeling
    idx = matrix.index.to_frame(index=False)
    y = idx["event_id"].astype(str).tolist()

    fig = go.Figure(
        data=
        go.Heatmap(
            z=matrix.to_numpy(),
            x=matrix.columns.astype(str).tolist(),
            y=y,
            colorscale="RdBu",
            zmid=0,
            colorbar=dict(title="Signed impact"),
        )
    )
    fig.update_layout(
        title="Event–Indicator Association Matrix (from impact_links)",
        xaxis_title="Indicator",
        yaxis_title="Event ID",
        margin=dict(l=80, r=40, t=80, b=80),
    )
    return fig


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate final-report figures and forecast tables.")
    parser.add_argument("--out", default="reports/figures", help="Output directory for figures")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parents[1]
    out_dir = project_dir / args.out
    _ensure_dir(out_dir)

    df = load_enriched_unified(project_dir)

    # Forecasts (three scenarios)
    access_base = forecast_access_account_ownership(df, scenario="base")
    access_opt = forecast_access_account_ownership(df, scenario="optimistic")
    access_pes = forecast_access_account_ownership(df, scenario="pessimistic")

    usage_base = forecast_usage_p2p_count(df, scenario="base")
    usage_opt = forecast_usage_p2p_count(df, scenario="optimistic")
    usage_pes = forecast_usage_p2p_count(df, scenario="pessimistic")

    fc_table = build_forecast_table([access_base, access_opt, access_pes, usage_base, usage_opt, usage_pes])
    fc_csv = project_dir / "models" / "forecast_table.csv"
    _ensure_dir(fc_csv.parent)
    fc_table.to_csv(fc_csv, index=False)

    _save_fig(_access_forecast_fig(fc_table), out_dir / "forecast_access.png")
    _save_fig(_usage_forecast_fig(fc_table), out_dir / "forecast_usage_p2p.png")

    # Event-indicator matrix
    impact_links = load_raw_impact_links(project_dir)
    m = build_event_indicator_matrix(df, impact_links)
    matrix_csv = project_dir / "models" / "event_indicator_matrix.csv"
    m.matrix.to_csv(matrix_csv)

    _save_fig(_event_indicator_heatmap(m.matrix), out_dir / "event_indicator_matrix.png")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
