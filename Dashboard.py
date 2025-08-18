"""
Improved Streamlit dashboard for triathlon training data
======================================================

This revision refactors the original dashboard to be more performant,
readable and maintainable. Key improvements include:

* **Caching heavy operations** with ``st.cache_data`` so that the CSV files
  are only read and parsed once per session. Column calculations
  (pace conversion, zone classification, etc.) are also cached.
* **Modular functions** for repeated tasks like filtering by timeframe and
  discipline, computing summary tables, and rendering charts. This makes
  it easy to extend or adjust the dashboard without copy‑pasting code.
* **Improved UX**: consistent labels and units, single column layouts on
  mobile (via ``use_container_width=True``), and guard clauses when no
  data is available to prevent runtime errors.

This file is meant to illustrate how you might apply the suggestions
discussed in conversation. It is not a drop‑in replacement for your
production code—rather, a starting point for further iteration.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import ast  # needed to unpack activityType dictionaries

###############################################################################
# Caching and data preparation
###############################################################################

@st.cache_data(show_spinner=False)
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw Garmin CSV files from disk.

    Returns
    -------
    (df_activities, df_health)
        Two DataFrames containing the activity and health metrics.  The
        ``parse_dates`` keyword is not used here to avoid converting all
        columns—dates are parsed in ``prepare_activities``.
    """
    df_activities = pd.read_csv("garmin_full_activities.csv")
    df_health = pd.read_csv("garmin_metrics_log.csv")
    return df_activities, df_health


def _safe_div(a: float, b: float, default: float = np.nan) -> float:
    """Safely divide two numbers, returning ``default`` if b is zero or NaN."""
    return default if pd.isna(a) or pd.isna(b) or b == 0 else a / b


def classify_discipline(type_key: str) -> str:
    """Map Garmin ``typeKey`` values to broad disciplines."""
    if type_key in ("running", "trail_running", "treadmill_running"):
        return "Run"
    if type_key in ("cycling", "indoor_cycling"):
        return "Bike"
    if type_key in ("open_water_swimming", "lap_swimming"):
        return "Swim"
    return "Other"


def convert_pace(row: pd.Series) -> float:
    """Convert speed to sport‑specific pace (min/km or km/h).

    Returns NaN if ``averageSpeed`` is missing or zero.  For swimming,
    the pace is expressed in minutes per 100m.
    """
    s = row.get("averageSpeed")
    d = row.get("Discipline")
    if pd.isna(s) or s == 0:
        return np.nan
    if d == "Swim":
        return _safe_div(100, s) / 60  # min/100m
    if d == "Bike":
        return s * 3.6  # km/h
    if d == "Run":
        return _safe_div(1000, s) / 60  # min/km
    return s


def convert_max_pace(row: pd.Series) -> float:
    """Convert max speed to sport‑specific pace (min/km or km/h)."""
    s = row.get("maxSpeed")
    d = row.get("Discipline")
    if pd.isna(s) or s == 0:
        return np.nan
    if d == "Swim":
        return _safe_div(100, s) / 60
    if d == "Bike":
        return s * 3.6
    if d == "Run":
        return _safe_div(1000, s) / 60
    return s


def classify_zone(row: pd.Series) -> str:
    """Classify each workout into a training zone based on HR time in zones."""
    z1 = (row.get("hrTimeInZone_1") or 0) or 0
    z2 = (row.get("hrTimeInZone_2") or 0) or 0
    z3 = (row.get("hrTimeInZone_3") or 0) or 0
    z4 = (row.get("hrTimeInZone_4") or 0) or 0
    z5 = (row.get("hrTimeInZone_5") or 0) or 0
    an = (row.get("anaerobicTrainingEffect") or 0) or 0
    # VO²max efforts: high anaerobic load + zone 5 time
    if z5 > 60 and an > 3:
        return "Z5 - VO²Max"
    # Threshold training: significant zone 4 time and high anaerobic TE
    if z4 > 180 and an > 3:
        return "Z4 - Threshold"
    # Tempo: zone 3 dominant
    if z3 > max(z1, z2):
        return "Z3 - Tempo"
    # Endurance: zone 2 dominant
    if z2 > max(z3, z1):
        return "Z2 - Endurance"
    # Recovery: mostly zone 1
    if z1 > max(z2, z3, z4, z5):
        return "Z1 - Recovery"
    return "Unknown"


@st.cache_data(show_spinner=False)
def prepare_activities(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and enrich the raw activities DataFrame.

    This function performs the following steps:
    * Parse date columns
    * Compute ISO calendar fields (day, week, year, yearweek)
    * Drop unneeded columns
    * Classify discipline and compute sport‑specific paces
    * Classify HR zone for each workout
    * Compute derived metrics (distance in km, duration in hours, aerobic efficiency)

    Returns
    -------
    pd.DataFrame
        A cleaned and enriched DataFrame ready for visualisation.
    """
    df = df_raw.copy()
    # -------------------------------------------------------------------------
    # Unpack the `activityType` column. Garmin encodes additional information
    # (including `typeKey`) inside this nested dictionary or string. If we
    # don’t expand it, the `typeKey` column used for discipline mapping will
    # not exist, resulting in a KeyError.  We convert strings using
    # ``ast.literal_eval`` and leave dictionaries unchanged.  Unknown or
    # malformed entries are mapped to an empty dict.
    if "activityType" in df.columns:
        def _to_dict(x):
            if isinstance(x, dict):
                return x
            if isinstance(x, str):
                try:
                    return ast.literal_eval(x)
                except Exception:
                    return {}
            return {}
        df["activityType_dict"] = df["activityType"].apply(_to_dict)
        # Expand the dictionary into separate columns
        activity_type_df = df["activityType_dict"].apply(pd.Series)
        df = pd.concat([df, activity_type_df], axis=1)
        # Drop the original columns to avoid confusion
        df.drop(columns=["activityType", "activityType_dict"], inplace=True, errors="ignore")
    # -------------------------------------------------------------------------
    # Parse timestamps
    df["startTimeLocal"] = pd.to_datetime(df["startTimeLocal"], format="%Y-%m-%d %H:%M:%S")
    # ISO calendar fields
    iso = df["startTimeLocal"].dt.isocalendar()
    df["Day"] = iso.day.astype(int)
    df["Week"] = iso.week.astype(int)
    df["Year"] = iso.year.astype(int)
    df["YearWeek"] = df["Year"].astype(str) + "-W" + df["Week"].astype(str)
    df["WeekIndex"] = df["Year"] * 53 + df["Week"]
    # Drop columns that are not needed for this dashboard
    drop_cols = [
        'activityId','startTimeGMT','eventType','duration','elapsedDuration','startLatitude',
        'startLongitude','hasPolyline','hasImages','ownerId','ownerDisplayName','ownerFullName',
        'ownerProfileImageUrlSmall','ownerProfileImageUrlMedium','ownerProfileImageUrlLarge',
        'bmrCalories','steps','userRoles','privacy','userPro','hasVideo','timeZoneId',
        'beginTimestamp','sportTypeId','deviceId','minElevation','maxElevation','maxDoubleCadence',
        'summarizedDiveInfo','maxVerticalSpeed','manufacturer','locationName','lapCount','endLatitude',
        'endLongitude','waterEstimated','minRespirationRate','maxRespirationRate','avgRespirationRate',
        'trainingEffectLabel','minActivityLapDuration','splitSummaries','hasSplits',
        'moderateIntensityMinutes','vigorousIntensityMinutes','avgGradeAdjustedSpeed','differenceBodyBattery',
        'hasHeatMap','fastestSplit_1000','fastestSplit_1609','fastestSplit_5000','endTimeGMT',
        'qualifyingDive','purposeful','manualActivity','autoCalcCalories','elevationCorrected',
        'atpActivity','favorite','decoDive','pr','parent','strokes','avgStrokeDistance',
        'minTemperature','maxTemperature','courseId','summarizedExerciseSets','totalSets','activeSets',
        'totalReps','vO2MaxValue','max20MinPower','trainingStressScore','intensityFactor','maxAvgPower_1',
        'maxAvgPower_2','maxAvgPower_5','maxAvgPower_10','maxAvgPower_20','maxAvgPower_30','maxAvgPower_60',
        'maxAvgPower_120','maxAvgPower_300','maxAvgPower_600','maxAvgPower_1200','maxAvgPower_1800',
        'maxAvgPower_3600','excludeFromPowerCurveReports','fastestSplit_40000','powerTimeInZone_6',
        'powerTimeInZone_7','fastestSplit_10000','workoutId','fastestSplit_21098','description',
        'avgStress','startStress','endStress','differenceStress','maxStress','activeLengths','poolLength',
        'unitOfPoolLength','avgStrokes','fastestSplit_100','grit','avgFlow','isHidden','restricted',
        'trimmable'
    ]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # Discipline classification
    df["Discipline"] = df["typeKey"].apply(classify_discipline)
    # Pace and max pace
    df["sportPace"] = df.apply(convert_pace, axis=1)
    df["sportMaxPace"] = df.apply(convert_max_pace, axis=1)
    # Zone classification
    df["classifiedZone"] = df.apply(classify_zone, axis=1)
    # Convert distance (m → km) and duration (s → hours)
    df["distance"] = df["distance"] / 1000.0
    df["movingDuration"] = df["movingDuration"] / 3600.0
    # Derived metrics
    df["AerobicEfficiency"] = df["sportPace"] * df["averageHR"]
    df["AerobicEfficiencyBike"] = df["averageHR"] / (df["sportPace"] / 60)
    return df


@st.cache_data(show_spinner=False)
def prepare_health(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Clean and sort the raw health DataFrame."""
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"], format="%Y-%m-%d")
    df = df.sort_values("date")
    # ISO calendar fields for weekly aggregation later
    iso = df["date"].dt.isocalendar()
    df["Week"] = iso.week.astype(int)
    df["Year"] = iso.year.astype(int)
    return df


###############################################################################
# Helper functions for summaries and performance management
###############################################################################


def filter_timeframe(df: pd.DataFrame, days: int | None) -> pd.DataFrame:
    """Return a subset of the DataFrame limited to the last ``days`` days.

    If ``days`` is ``None``, the original DataFrame is returned unchanged.
    """
    if days is None:
        return df.copy()
    cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
    return df[df["startTimeLocal"] >= cutoff].copy()


def summarize_by(df: pd.DataFrame, group_field: str, days: int | None) -> pd.DataFrame:
    """Aggregate distance and duration by a given field (discipline or zone).

    Parameters
    ----------
    df : pd.DataFrame
        Activity data.
    group_field : str
        Column name to group by (e.g. "Discipline" or "classifiedZone").
    days : int | None
        Number of days to include.  ``None`` means all time.

    Returns
    -------
    pd.DataFrame
        Aggregated table with total movingDuration and distance.
    """
    subset = filter_timeframe(df, days)
    return (
        subset.groupby(group_field)[["movingDuration", "distance"]]
        .sum()
        .reset_index()
    )


def compute_training_load(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate a training load score and compute chronic/acute load and TSB.

    We estimate training load as duration in hours multiplied by average heart
    rate.  This is a rough proxy for TRIMP/TSS when more detailed power or
    RPE data are unavailable.  See e.g. Bannister (1991) and Skiba (2013).

    The function then aggregates the load by day, fills in missing dates
    with zero load, and calculates the exponential moving averages for CTL
    (42‑day time constant) and ATL (7‑day time constant).  TSB is CTL minus
    ATL.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ``date``, ``load``, ``CTL``, ``ATL``, ``TSB``.
    """
    # Approximate load per workout
    df = df.copy()
    df["load"] = df["movingDuration"] * df["averageHR"]
    daily = (
        df.assign(date=df["startTimeLocal"].dt.date)
        .groupby("date", as_index=False)["load"].sum()
    )
    # Fill missing days with zero load
    idx = pd.date_range(daily["date"].min(), daily["date"].max(), freq="D")
    daily = daily.set_index("date").reindex(idx, fill_value=0).rename_axis("date").reset_index()
    # Exponential moving averages
    def exp_moving_average(x: pd.Series, tau: float) -> pd.Series:
        alpha = 2 / (tau + 1)
        ema = []
        prev = 0.0
        for i, v in enumerate(x):
            prev = alpha * v + (1 - alpha) * prev if i > 0 else v
            ema.append(prev)
        return pd.Series(ema, index=x.index)
    daily["CTL"] = exp_moving_average(daily["load"], tau=42)
    daily["ATL"] = exp_moving_average(daily["load"], tau=7)
    daily["TSB"] = daily["CTL"] - daily["ATL"]
    return daily


###############################################################################
# UI components
###############################################################################

def render_kpi_cards(df: pd.DataFrame) -> None:
    """Display KPI cards for number of workouts, total time, distance and average HR."""
    if df.empty:
        st.info("No data available for this selection.")
        return
    # Aggregate metrics
    total_workouts = len(df)
    total_time = df["movingDuration"].sum()
    total_dist = df["distance"].sum()
    avg_hr = df["averageHR"].mean()
    # Helper
    def hhmm(hours: float) -> str:
        total_minutes = int(hours * 60)
        return f"{total_minutes // 60:d}h {total_minutes % 60:02d}m"
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Workouts", total_workouts)
    k2.metric("Time", hhmm(total_time))
    k3.metric("Distance", f"{total_dist:.1f} km")
    k4.metric("Avg HR", f"{avg_hr:.0f} bpm" if not np.isnan(avg_hr) else "—")


def render_training_load_chart(df: pd.DataFrame) -> None:
    """Render the performance management chart (CTL, ATL, TSB)."""
    pmc = compute_training_load(df)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=pmc["date"], y=pmc["CTL"], name="CTL (42d)", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=pmc["date"], y=pmc["ATL"], name="ATL (7d)", line=dict(width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=pmc["date"], y=pmc["TSB"], name="TSB", line=dict(width=2, dash="dot")))
    fig.update_layout(
        title="Performance Management Chart", xaxis_title="Date", yaxis_title="Load", legend_title="Metric",
        height=400
    )
    st.plotly_chart(fig, use_container_width=True)


def render_sport_section(
    df: pd.DataFrame, sport: str, label_speed: str, units_speed: str, ae_col: str
) -> None:
    """Render charts and a table for a single sport (Swim/Bike/Run).

    Parameters
    ----------
    df : pd.DataFrame
        Prepared activities DataFrame.
    sport : str
        Discipline to filter by ("Swim", "Bike", or "Run").
    label_speed : str
        Axis label for pace/speed charts (e.g. "Pace (min/km)").
    units_speed : str
        Name of the pace column in df ("sportPace" or "sportMaxPace").
    ae_col : str
        Column name for aerobic efficiency (e.g. ``AerobicEfficiencyBike``).
    """
    st.markdown(f"## {sport} analysis")
    # Filter by sport
    df_sport = df[df["Discipline"] == sport].copy()
    # Timeframe selection
    timeframe_label = st.selectbox(
        f"Timeframe for {sport}", list(timeframe_days.keys()), key=f"tf_{sport}"
    )
    days = timeframe_days[timeframe_label]
    df_time = filter_timeframe(df_sport, days)
    if df_time.empty:
        st.info("No data available for the selected timeframe.")
        return
    # Weekly summary (for stacked bar charts)
    summary_week = (
        df_time.groupby(["WeekIndex", "YearWeek", "classifiedZone"])
        .agg(movingDuration=("movingDuration", "sum"), distance=("distance", "sum"))
        .reset_index()
    )
    # KPI cards
    render_kpi_cards(df_time)
    # Stacked bar: weekly duration by zone
    fig_time = px.bar(
        summary_week,
        x="WeekIndex",
        y="movingDuration",
        color="classifiedZone",
        labels={"movingDuration": "Duration (hr)", "WeekIndex": "Week"},
        title=f"Weekly {sport.lower()} time by zone",
    )
    fig_time.update_layout(xaxis=dict(
        tickmode="array",
        tickvals=summary_week["WeekIndex"],
        ticktext=summary_week["YearWeek"],
        title="Week"
    ))
    st.plotly_chart(fig_time, use_container_width=True)
    # Stacked bar: weekly distance by zone (km)
    fig_dist = px.bar(
      summary_week,
      x="WeekIndex",
      y="distance",
      color="classifiedZone",
      labels={"distance": "Distance (km)", "WeekIndex": "Week"},
      title=f"Weekly {sport.lower()} distance by zone",
    )
    fig_dist.update_layout(
      xaxis=dict(
          tickmode="array",
          tickvals=summary_week["WeekIndex"],
          ticktext=summary_week["YearWeek"],
          title="Week",
      )
    )
    st.plotly_chart(fig_dist, use_container_width=True)
    # Scatter: pace/speed vs date
    fig_speed = px.scatter(
        df_time,
        x="startTimeLocal",
        y="sportPace",
        color="classifiedZone",
        labels={"sportPace": label_speed, "startTimeLocal": "Date"},
        title=f"{sport} {label_speed} over time",
    )
    st.plotly_chart(fig_speed, use_container_width=True)
    # Aerobic efficiency scatter
    if ae_col in df_time.columns:
        fig_ae = px.scatter(
            df_time,
            x="startTimeLocal",
            y=ae_col,
            color="classifiedZone",
            title=f"{sport} Aerobic Efficiency",
            labels={ae_col: "AE", "startTimeLocal": "Date"}
        )
        st.plotly_chart(fig_ae, use_container_width=True)
    # Table of workouts
    columns_to_show = [
        "activityName", "startTimeLocal", "YearWeek", "typeKey", "distance",
        "classifiedZone", "movingDuration", "sportPace", "sportMaxPace", "averageHR", "maxHR", ae_col
    ]
    table = df_time[columns_to_show].copy()
    table.rename(columns={
        "activityName": "Name",
        "startTimeLocal": "Time",
        "YearWeek": "Week",
        "typeKey": f"{sport} type",
        "distance": "Distance (km)",
        "movingDuration": "Duration (h)",
        "sportPace": label_speed,
        "sportMaxPace": f"Max {label_speed}",
        "averageHR": "Avg HR",
        "maxHR": "Max HR",
        ae_col: "AE"
    }, inplace=True)
    st.dataframe(table)


###############################################################################
# Main Streamlit app
###############################################################################

st.set_page_config(page_title="Triathlon Dashboard", layout="wide")

st.title("Triathlon Training Dashboard")

# Sidebar timeframe selection
timeframe_days = {
    "Last month": 30,
    "Last three months": 90,
    "Last six months": 183,
    "Last year": 365,
    "All time": None,
}

# Load and prepare data
df_activities_raw, df_health_raw = load_raw_data()
df_activities = prepare_activities(df_activities_raw)
df_health = prepare_health(df_health_raw)

# Tabs
overview_tab, health_tab, swim_tab, bike_tab, run_tab, pmc_tab = st.tabs([
    "Overview", "Health", "Swim", "Bike", "Run", "Performance Mgmt"])

# Overview
with overview_tab:
    st.markdown("### Training overview")
    timeframe_label = st.selectbox("Global timeframe", list(timeframe_days.keys()), key="tf_overview")
    days = timeframe_days[timeframe_label]
    df_time = filter_timeframe(df_activities, days)
    # KPI cards for all sports combined
    render_kpi_cards(df_time)
    # Pie charts: discipline share of duration and distance
    summary_disc = summarize_by(df_time, "Discipline", None)  # Already filtered by timeframe
    fig_time = go.Figure(go.Pie(
        labels=summary_disc["Discipline"],
        values=summary_disc["movingDuration"],
        hole=0.4,
        title=f"{timeframe_label} – Time spent (hr)"
    ))
    fig_dist = go.Figure(go.Pie(
        labels=summary_disc["Discipline"],
        values=summary_disc["distance"],
        hole=0.4,
        title=f"{timeframe_label} – Distance (km)"
    ))
    col1, col2 = st.columns(2)
    col1.plotly_chart(fig_time, use_container_width=True)
    col2.plotly_chart(fig_dist, use_container_width=True)

# Health tab
with health_tab:
    st.markdown("### Health metrics")
    # Filter health data
    timeframe_label = st.selectbox("Timeframe for health", list(timeframe_days.keys()), key="tf_health")
    days = timeframe_days[timeframe_label]
    if days is not None:
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=days)
        df_health_time = df_health[df_health["date"] >= cutoff].copy()
    else:
        df_health_time = df_health.copy()
    if df_health_time.empty:
        st.info("No health data available for this timeframe.")
    else:
        # Plot sleep, resting HR and HRV
        for col_name, title, color in [
            ("sleep_score", "Sleep Score", "#FFD700"),
            ("resting_hr", "Resting HR (bpm)", "#FF00C8"),
            ("avg_hrv", "Nightly HRV", "#00FFDD"),
        ]:
            fig = px.line(df_health_time, x="date", y=col_name, title=title)
            fig.update_traces(line=dict(color=color))
            st.plotly_chart(fig, use_container_width=True)

# Individual sport tabs
with swim_tab:
    render_sport_section(df_activities, "Swim", "Pace (min/100m)", "sportPace", "AerobicEfficiency")
with bike_tab:
    render_sport_section(df_activities, "Bike", "Speed (km/h)", "sportPace", "AerobicEfficiencyBike")
with run_tab:
    render_sport_section(df_activities, "Run", "Pace (min/km)", "sportPace", "AerobicEfficiency")






