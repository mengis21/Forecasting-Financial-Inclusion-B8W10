from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "processed" / "enriched_unified_data.csv"
FORECAST_PATH = PROJECT_DIR / "models" / "forecast_table.csv"
MATRIX_PATH = PROJECT_DIR / "models" / "event_indicator_matrix.csv"


@st.cache_data
def load_unified() -> pd.DataFrame:
	df = pd.read_csv(DATA_PATH)
	for c in ["observation_date", "period_start", "period_end", "collection_date"]:
		if c in df.columns:
			df[c] = pd.to_datetime(df[c], errors="coerce")
	return df


@st.cache_data
def load_forecasts() -> pd.DataFrame | None:
	if not FORECAST_PATH.exists():
		return None
	return pd.read_csv(FORECAST_PATH)


@st.cache_data
def load_matrix() -> pd.DataFrame | None:
	if not MATRIX_PATH.exists():
		return None
	# Multi-index saved to CSV; read raw then keep first 3 cols as index
	m = pd.read_csv(MATRIX_PATH)
	if {"event_id", "event_category", "event_date"}.issubset(m.columns):
		m["event_date"] = pd.to_datetime(m["event_date"], errors="coerce")
		m = m.set_index(["event_id", "event_category", "event_date"])
	return m


def fig_indicator_ts(obs: pd.DataFrame, indicator_code: str) -> go.Figure:
	d = obs[obs["indicator_code"] == indicator_code].copy()
	d = d.dropna(subset=["observation_date", "value_numeric"])
	d["year"] = d["observation_date"].dt.year

	fig = px.line(
		d.sort_values("observation_date"),
		x="observation_date",
		y="value_numeric",
		color="gender" if "gender" in d.columns and d["gender"].nunique() > 1 else None,
		markers=True,
		title=f"{indicator_code}: {d['indicator'].dropna().iloc[0] if len(d) else ''}",
	)
	fig.update_layout(legend_title_text="Gender")
	return fig


def fig_matrix_heatmap(matrix: pd.DataFrame) -> go.Figure:
	idx = matrix.index.to_frame(index=False)
	y = idx["event_id"].astype(str).tolist() if "event_id" in idx.columns else list(range(len(matrix)))
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
		title="Event–Indicator Association Matrix",
		xaxis_title="Indicator",
		yaxis_title="Event ID",
		margin=dict(l=80, r=40, t=80, b=80),
	)
	return fig


st.set_page_config(page_title="Ethiopia Financial Inclusion", layout="wide")

st.title("Forecasting Financial Inclusion — Ethiopia")

df = load_unified()
obs = df[df["record_type"] == "observation"].copy()
events = df[df["record_type"] == "event"].copy()

page = st.sidebar.radio(
	"Navigate",
	["Overview", "Trends", "Event Impacts", "Forecasts"],
)

st.sidebar.markdown("---")
st.sidebar.caption("Data source: enriched unified dataset")


if page == "Overview":
	c1, c2, c3, c4 = st.columns(4)
	c1.metric("Observations", int((df.record_type == "observation").sum()))
	c2.metric("Events", int((df.record_type == "event").sum()))
	c3.metric("Indicators", int(obs["indicator_code"].nunique()))
	c4.metric("Latest year", int(pd.to_datetime(obs["observation_date"], errors="coerce").dt.year.max()))

	st.subheader("Access (Account Ownership)")
	st.plotly_chart(fig_indicator_ts(obs, "ACC_OWNERSHIP"), use_container_width=True)

	st.subheader("Usage proxy (P2P Transactions)")
	if (obs["indicator_code"] == "USG_P2P_COUNT").any():
		st.plotly_chart(fig_indicator_ts(obs, "USG_P2P_COUNT"), use_container_width=True)
	else:
		st.info("No USG_P2P_COUNT observations found.")

	st.subheader("Download")
	st.download_button(
		"Download enriched dataset (CSV)",
		data=DATA_PATH.read_bytes(),
		file_name="enriched_unified_data.csv",
		mime="text/csv",
	)


elif page == "Trends":
	st.subheader("Interactive indicator trends")
	indicator_codes = sorted(obs["indicator_code"].dropna().unique().tolist())
	code = st.selectbox("Indicator code", indicator_codes, index=indicator_codes.index("ACC_OWNERSHIP") if "ACC_OWNERSHIP" in indicator_codes else 0)
	st.plotly_chart(fig_indicator_ts(obs, code), use_container_width=True)

	st.subheader("Events timeline")
	e = events.dropna(subset=["observation_date"]).sort_values("observation_date")
	if len(e):
		timeline = px.scatter(
			e,
			x="observation_date",
			y="category",
			hover_data=["record_id", "notes", "source_name"],
			title="Cataloged events",
		)
		st.plotly_chart(timeline, use_container_width=True)
	else:
		st.info("No event dates found.")


elif page == "Event Impacts":
	st.subheader("Impact links (event → indicator)")
	matrix = load_matrix()
	if matrix is None:
		st.warning("Event–indicator matrix not found. Run `python -m src.generate_assets` first.")
	else:
		st.plotly_chart(fig_matrix_heatmap(matrix.fillna(0)), use_container_width=True)
		st.caption("Signed impact = impact_magnitude × direction (+/-).")

		st.download_button(
			"Download event–indicator matrix (CSV)",
			data=MATRIX_PATH.read_bytes(),
			file_name="event_indicator_matrix.csv",
			mime="text/csv",
		)


elif page == "Forecasts":
	st.subheader("Forecasts (2025–2027)")
	fc = load_forecasts()
	if fc is None:
		st.warning("Forecast table not found. Run `python -m src.generate_assets` first.")
	else:
		scenario = st.selectbox("Scenario", ["base", "optimistic", "pessimistic"], index=0)
		f = fc[fc["scenario"] == scenario].copy()

		st.markdown("**Access — Account Ownership**")
		acc = f[f["indicator_code"] == "ACC_OWNERSHIP"]
		if len(acc):
			fig = px.line(acc, x="year", y="forecast_value", markers=True, title="Account ownership forecast")
			fig.update_yaxes(title_text="% of adults")
			st.plotly_chart(fig, use_container_width=True)
		else:
			st.info("No Access forecast rows found.")

		st.markdown("**Usage — P2P Transactions (proxy)**")
		usg = f[f["indicator_code"] == "USG_P2P_COUNT"]
		if len(usg):
			fig = px.line(usg, x="year", y="forecast_value", markers=True, title="P2P transactions forecast")
			fig.update_yaxes(title_text="transactions")
			st.plotly_chart(fig, use_container_width=True)
		else:
			st.info("No Usage forecast rows found.")

		st.subheader("Download")
		st.download_button(
			"Download forecast table (CSV)",
			data=FORECAST_PATH.read_bytes(),
			file_name="forecast_table.csv",
			mime="text/csv",
		)
