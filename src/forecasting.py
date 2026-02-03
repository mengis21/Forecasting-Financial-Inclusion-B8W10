from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


def _year(dt: pd.Series) -> pd.Series:
    return pd.to_datetime(dt, errors="coerce").dt.year


def _logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def _inv_logit(z: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-z))


@dataclass(frozen=True)
class ForecastResult:
    target_code: str
    target_name: str
    history: pd.DataFrame
    forecast: pd.DataFrame


def forecast_access_account_ownership(
    unified_df: pd.DataFrame,
    *,
    years: list[int] | None = None,
    scenario: Literal["base", "optimistic", "pessimistic"] = "base",
) -> ForecastResult:
    """Forecast account ownership rate for 2025–2027.

    Uses a logit-linear trend on Findex national 'all' observations.
    Scenario adjusts the slope slightly to create uncertainty bands.
    """
    if years is None:
        years = [2025, 2026, 2027]

    obs = unified_df[(unified_df.record_type == "observation") & (unified_df.indicator_code == "ACC_OWNERSHIP")].copy()
    obs = obs[(obs.gender == "all") & (obs.location == "national")]

    obs = obs.assign(year=_year(obs.observation_date))
    obs = obs.dropna(subset=["year", "value_numeric"]).sort_values("year")

    # Convert percent to proportion
    y = obs["value_numeric"].astype(float).to_numpy() / 100.0
    x = obs["year"].astype(int).to_numpy()

    if len(obs) < 3:
        raise ValueError("Not enough history to fit access forecast")

    # Fit logit(y) = a + b*(year - year0)
    year0 = int(x.min())
    X = x - year0
    z = _logit(y)

    b, a = np.polyfit(X, z, deg=1)  # returns [slope, intercept]

    # Scenario adjustment on slope (small, transparent, non-overfitted)
    if scenario == "optimistic":
        b = b * 1.15
    elif scenario == "pessimistic":
        b = b * 0.85

    f_years = np.array(years, dtype=int)
    f_X = f_years - year0
    f_z = a + b * f_X
    f_y = _inv_logit(f_z) * 100.0

    hist = obs[["year", "value_numeric", "source_name", "confidence"]].rename(columns={"value_numeric": "value"})
    fc = pd.DataFrame(
        {
            "year": f_years,
            "value": f_y,
            "scenario": scenario,
        }
    )

    return ForecastResult(
        target_code="ACC_OWNERSHIP",
        target_name="Account Ownership Rate (%, adults 15+)",
        history=hist,
        forecast=fc,
    )


def forecast_usage_p2p_count(
    unified_df: pd.DataFrame,
    *,
    years: list[int] | None = None,
    scenario: Literal["base", "optimistic", "pessimistic"] = "base",
) -> ForecastResult:
    """Forecast P2P transaction count as a Usage proxy for 2025–2027.

    Fits a log-linear trend on available annual observations (very sparse),
    then applies scenario multipliers to growth.
    """
    if years is None:
        years = [2026, 2027]

    obs = unified_df[(unified_df.record_type == "observation") & (unified_df.indicator_code == "USG_P2P_COUNT")].copy()
    obs = obs.assign(year=_year(obs.observation_date))
    obs = obs.dropna(subset=["year", "value_numeric"]).sort_values("year")

    if len(obs) < 2:
        raise ValueError("Not enough history to fit usage forecast")

    x = obs["year"].astype(int).to_numpy()
    y = obs["value_numeric"].astype(float).to_numpy()

    # log(y) = a + b*(year-year0)
    year0 = int(x.min())
    X = x - year0
    ly = np.log(np.clip(y, 1e-9, None))
    b, a = np.polyfit(X, ly, deg=1)

    if scenario == "optimistic":
        b = b * 1.25
    elif scenario == "pessimistic":
        b = b * 0.75

    f_years = np.array(years, dtype=int)
    f_X = f_years - year0
    f_ly = a + b * f_X
    f_y = np.exp(f_ly)

    hist = obs[["year", "value_numeric", "source_name", "confidence"]].rename(columns={"value_numeric": "value"})
    fc = pd.DataFrame({"year": f_years, "value": f_y, "scenario": scenario})

    return ForecastResult(
        target_code="USG_P2P_COUNT",
        target_name="P2P Transactions (count, proxy for digital usage)",
        history=hist,
        forecast=fc,
    )


def build_forecast_table(results: list[ForecastResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        for _, row in r.forecast.iterrows():
            rows.append(
                {
                    "indicator_code": r.target_code,
                    "indicator": r.target_name,
                    "year": int(row["year"]),
                    "scenario": str(row["scenario"]),
                    "forecast_value": float(row["value"]),
                }
            )
    return pd.DataFrame(rows).sort_values(["indicator_code", "scenario", "year"])
