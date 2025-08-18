# triathlon/pmc.py
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class PMCParams:
    atl_days: int = 7
    ctl_days: int = 42

def complete_daily_index(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure continuous daily index; fill missing days with 0 load."""
    if daily_df.empty:
        today = pd.Timestamp.now().normalize()
        return pd.DataFrame({"date": [today], "load": [0.0]})

    daily = daily_df.copy()
    # NEW: normalize and coerce
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce").dt.normalize()

    idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    return (daily.set_index("date")
                 .reindex(idx, fill_value=0.0)
                 .rename_axis("date")
                 .reset_index())


def ema(series: pd.Series, N: int) -> pd.Series:
    """Classic EMA with period N."""
    alpha = 2 / (N + 1)
    out = np.empty(len(series), dtype=float)
    prev = np.nan
    for i, v in enumerate(series.to_numpy(float)):
        prev = v if i == 0 or np.isnan(prev) else prev + alpha * (v - prev)
        out[i] = prev
    return pd.Series(out, index=series.index)

def compute_daily_load(df_acts: pd.DataFrame, method: str = "hr_proxy") -> pd.DataFrame:
    """
    Build a daily load series from activities.
    method:
      - 'hr_proxy': load = movingDuration_hours * averageHR
      - 'srpe'    : load = movingDuration_hours * sRPE * 10   (requires 'sRPE')
      - 'tss'     : load = trainingStressScore                (requires 'trainingStressScore')
    Returns a DataFrame with columns ['date','load'] (date normalized to 00:00).
    """
    if df_acts.empty:
        return pd.DataFrame({"date": [], "load": []})
    df = df_acts.copy()
    df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"], errors="coerce")
    df["movingDuration"] = pd.to_numeric(df["movingDuration"], errors="coerce")
    df["averageHR"]      = pd.to_numeric(df.get("averageHR"), errors="coerce")

    if method == "tss" and "trainingStressScore" in df.columns:
        load = pd.to_numeric(df["trainingStressScore"], errors="coerce").fillna(0.0)
    elif method == "srpe" and "sRPE" in df.columns:
        load = df["movingDuration"].fillna(0.0) * pd.to_numeric(df["sRPE"], errors="coerce").fillna(0.0) * 10.0
    else:
        load = df["movingDuration"].fillna(0.0) * df["averageHR"].fillna(0.0)

    daily = (pd.DataFrame({
        "date": df["startTimeLocal"].dt.normalize(),
        "load": load
    }).groupby("date", as_index=False)["load"].sum())

    return complete_daily_index(daily)

def compute_atl_ctl(daily_load_df: pd.DataFrame, params: PMCParams = PMCParams()) -> pd.DataFrame:
    """Compute ATL/CTL time series from daily load."""
    dl = daily_load_df.copy()
    dl["ATL"] = ema(dl["load"], N=params.atl_days)
    dl["CTL"] = ema(dl["load"], N=params.ctl_days)
    return dl  # cols: date, load, ATL, CTL

def extend_with_plan(daily_hist: pd.DataFrame,
                     future_plan: pd.DataFrame) -> pd.DataFrame:
    """
    Concatenate historical daily load with future planned load.
    Inputs:
      daily_hist: ['date','load'] (historical, continuous)
      future_plan: ['date','load'] (future dates only, any gaps allowed)
    Returns continuous ['date','load'] covering hist + plan.
    """
    df = pd.concat([daily_hist, future_plan], ignore_index=True)
    df = df.drop_duplicates(subset=["date"]).sort_values("date")
    return complete_daily_index(df)

def forecast_atl_ctl(daily_hist: pd.DataFrame,
                     future_plan: pd.DataFrame | None = None,
                     params: PMCParams = PMCParams()) -> pd.DataFrame:
    """
    Forecast ATL/CTL forward. If future_plan is None, projects using the mean of
    the last 14 days as a flat plan.
    Returns DataFrame with ['date','load','ATL','CTL'] for hist + forecast.
    """
    hist = daily_hist.copy()
    hist = complete_daily_index(hist)

    if future_plan is None or future_plan.empty:
        # Flat plan based on recent average (14 days or tail of series)
        tail = hist.tail(min(14, len(hist)))
        mean_load = float(tail["load"].mean()) if not tail.empty else 0.0
        future_dates = pd.date_range(hist["date"].max() + pd.Timedelta(days=1),
                                     periods=28, freq="D")
        future_plan = pd.DataFrame({"date": future_dates, "load": mean_load})
    else:
        future_plan = future_plan.copy()
        future_plan["date"] = pd.to_datetime(future_plan["date"], errors="coerce").dt.normalize()
        future_plan["load"] = pd.to_numeric(future_plan["load"], errors="coerce").fillna(0.0)

    joined = extend_with_plan(hist, future_plan)
    return compute_atl_ctl(joined, params=params)
