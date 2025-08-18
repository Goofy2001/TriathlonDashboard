# triathlon/pmc.py
from __future__ import annotations
import pandas as pd
import numpy as np
from dataclasses import dataclass

@dataclass(frozen=True)
class PMCParams:
    adl_days: int = 7    # ADL (a.k.a. ATL)
    ctl_days: int = 42   # CTL

def complete_daily_index(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Ensure continuous daily index; fill missing days with 0 load."""
    if daily_df.empty:
        today = pd.Timestamp.now().normalize()
        return pd.DataFrame({"date": [today], "load": [0.0]})
    daily = daily_df.copy()
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
    vals = series.to_numpy(dtype=float, copy=False)
    for i, v in enumerate(vals):
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
    Returns ['date','load'].
    """
    if df_acts.empty:
        return pd.DataFrame({"date": [], "load": []})

    df = df_acts.copy()
    df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"], errors="coerce")
    # your pipeline already converts movingDuration to hours
    dur_h = pd.to_numeric(df["movingDuration"], errors="coerce").fillna(0.0)

    if method == "tss" and "trainingStressScore" in df.columns:
        load = pd.to_numeric(df["trainingStressScore"], errors="coerce").fillna(0.0)
    elif method == "srpe" and "sRPE" in df.columns:
        srpe = pd.to_numeric(df["sRPE"], errors="coerce").fillna(0.0)
        load = dur_h * srpe * 10.0
    else:
        hr = pd.to_numeric(df.get("averageHR"), errors="coerce").fillna(0.0)
        load = dur_h * hr

    daily = (pd.DataFrame({"date": df["startTimeLocal"].dt.normalize(), "load": load})
               .groupby("date", as_index=False)["load"].sum())
    return complete_daily_index(daily)

def compute_adl_ctl(daily_load_df: pd.DataFrame, params: PMCParams = PMCParams()) -> pd.DataFrame:
    """Compute ADL(=ATL) and CTL from a daily load frame."""
    dl = complete_daily_index(daily_load_df)
    dl["ADL"] = ema(dl["load"], N=params.adl_days)  # 7-day default
    dl["CTL"] = ema(dl["load"], N=params.ctl_days)  # 42-day default
    return dl  # cols: date, load, ADL, CTL

def add_ctl_bands(adl_ctl_df: pd.DataFrame,
                  caution_ratio: float = 1.30,
                  danger_ratio: float = 1.50,
                  lower_ratio: float | None = 0.80) -> pd.DataFrame:
    """
    Add dynamic bands around CTL to visualize safe/caution/danger.
    - caution (amber) upper = CTL * caution_ratio
    - danger (red)   upper = CTL * danger_ratio
    - optional 'lower' = CTL * lower_ratio (underloading band)
    """
    df = adl_ctl_df.copy()
    df["CTL_caution"] = df["CTL"] * caution_ratio
    df["CTL_danger"]  = df["CTL"] * danger_ratio
    if lower_ratio is not None:
        df["CTL_lower"] = df["CTL"] * lower_ratio
    return df
