# === Imports ===
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import ast

# === Functions ===
def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
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
    if type_key in ("cycling", "indoor_cycling", "virtual_ride"):
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
    df["hrTimeInZone_1"] = df["hrTimeInZone_1"] / 3600.0
    df["hrTimeInZone_2"] = df["hrTimeInZone_2"] / 3600.0
    df["hrTimeInZone_3"] = df["hrTimeInZone_3"] / 3600.0
    df["hrTimeInZone_4"] = df["hrTimeInZone_4"] / 3600.0
    df["hrTimeInZone_5"] = df["hrTimeInZone_5"] / 3600.0
    # Derived metrics
    df["AerobicEfficiency"] = df["sportPace"] * df["averageHR"]
    df["AerobicEfficiencyBike"] = df["averageHR"] / (df["sportPace"] / 60)
    return df

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

def get_filtered_data(df: pd.DataFrame, sport: str) -> pd.DataFrame:
    """Filter dataframe for given sport and timeframe."""
    df_sport = df[df["Discipline"] == sport].copy()
    timeframe_label = st.selectbox(
        f"Timeframe for {sport}", list(timeframe_days.keys()), key=f"tf_{sport}"
    )
    days = timeframe_days[timeframe_label]
    return filter_timeframe(df_sport, days)

def compute_weekly_summary(df_time: pd.DataFrame, zone_type: str) -> pd.DataFrame:
    """Compute weekly summary for duration (classified or HR zones) and distance (always classified)."""

    # --- Duration ---
    if zone_type == "Classified zone":
        summary_duration = (
            df_time.groupby(["WeekIndex", "YearWeek", "classifiedZone"])
            .agg(Duration=("movingDuration", "sum"))
            .reset_index()
            .rename(columns={"classifiedZone": "Zone"})
        )

    elif zone_type == "HR Zone":
        hr_cols = ["hrTimeInZone_1", "hrTimeInZone_2", "hrTimeInZone_3",
                   "hrTimeInZone_4", "hrTimeInZone_5"]

        # Melt into long format
        df_melt = df_time.melt(
            id_vars=["WeekIndex", "YearWeek"],
            value_vars=hr_cols,
            var_name="Zone",
            value_name="Duration"
        )
        # Extract zone number (hrTimeInZone_1 -> 1)
        df_melt["Zone"] = df_melt["Zone"].str.extract(r"(\d+)").astype(int)

        summary_duration = (
            df_melt.groupby(["WeekIndex", "YearWeek", "Zone"])
            .agg(Duration=("Duration", "sum"))
            .reset_index()
        )

    else:
        raise ValueError(f"Unknown zone_type: {zone_type}")

    # --- Distance (always classified zone) ---
    summary_distance = (
        df_time.groupby(["WeekIndex", "YearWeek", "classifiedZone"])
        .agg(distance=("distance", "sum"))
        .reset_index()
        .rename(columns={"classifiedZone": "Zone"})
    )

    return summary_duration, summary_distance


def plot_weekly_duration(summary_week: pd.DataFrame, sport: str, zone_type: str):
    """Stacked bar: weekly duration by zone type (classified or HR zones)."""
    
    df_plot = summary_week.copy()
    fig = px.bar(
        df_plot,
        x="WeekIndex",
        y="Duration",
        color="Zone",
        color_discrete_map=color_map_HR,
        labels={"Duration": "Duration (hr)", "WeekIndex": "Week"},
        title=f"Weekly {sport.lower()} time by {zone_type}",
    )

    if "YearWeek" in df_plot.columns:
        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=df_plot["WeekIndex"].unique(),
                ticktext=df_plot.groupby("WeekIndex")["YearWeek"].first().values,
                title="Week",
            )
        )

    st.plotly_chart(fig, use_container_width=True)


def plot_weekly_distance(summary_distance: pd.DataFrame, sport: str):
    """Stacked bar: weekly distance by classified zone only."""
    
    df_plot = summary_distance.copy()
    fig = px.bar(
        df_plot,
        x="WeekIndex",
        y="distance",
        color="Zone",
        color_discrete_map=color_map_HR,
        labels={"distance": "Distance (km)", "WeekIndex": "Week"},
        title=f"Weekly {sport.lower()} distance by classified zone",
    )

    if "YearWeek" in df_plot.columns:
        fig.update_layout(
            xaxis=dict(
                tickmode="array",
                tickvals=df_plot["WeekIndex"].unique(),
                ticktext=df_plot.groupby("WeekIndex")["YearWeek"].first().values,
                title="Week",
            )
        )

    st.plotly_chart(fig, use_container_width=True)



def plot_efficiency_trend(df_time: pd.DataFrame, sport: str, metric_col: str):
    if df_time.empty:
        st.info("No data available for this timeframe.")
        return

    # Ensure date is datetime
    df_time = df_time.copy()
    df_time["startTimeLocal"] = pd.to_datetime(df_time["startTimeLocal"])

    # Scatter with trendline
    fig = px.scatter(
        df_time,
        x="averageHR",
        y=metric_col,
        color="startTimeLocal",  # color gradient by date
        trendline="ols",
        hover_data=["startTimeLocal", "movingDuration", "distance", "typeKey"],
        labels={
            "averageHR": "Average HR",
            metric_col: "Pace/Power",
            "startTimeLocal": "Date"
        },
        title=f"{sport} Efficiency: HR vs {metric_col}"
    )

    # Style markers
    fig.update_traces(marker=dict(size=10, opacity=0.8))

    st.plotly_chart(fig, use_container_width=True)

def render_workout_table(df_time: pd.DataFrame, sport: str, label_speed: str, ae_col: str):
    """Render workouts table."""
    columns_to_show = [
        "activityName", "startTimeLocal", "YearWeek", "typeKey", "distance",
        "classifiedZone", "movingDuration", "sportPace", "sportMaxPace",
        "averageHR", "maxHR", ae_col
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
        ae_col: "AE",
    }, inplace=True)
    st.dataframe(table)

def render_sport_section(
    df: pd.DataFrame, sport: str, label_speed: str, units_speed: str, ae_col: str
) -> None:
    """Main orchestrator for sport analysis."""

    st.markdown(f"## {sport} analysis")

    # --- Filter data by sport & timeframe ---
    df_time = get_filtered_data(df, sport)
    if df_time.empty:
        st.info("No data available for the selected timeframe.")
        return

    # --- Zone type selection for stacked bars ---
    zone_type = st.radio(
        "Zone type for weekly summary",
        options=["Classified zone", "HR Zone"],
        index=0,
        horizontal=True,
        key=f"zone_type_{sport}"
    )

    # --- Metric selection for efficiency scatter ---
    metric = st.selectbox(
    "Metric to analyze",
    options=["sportPace", "avgPower"],
    index=0 if sport in ["Swim", "Run"] else 1,
    key=f"metric_{sport}"
    )


    # --- Weekly summaries for stacked bars ---
    summary_duration, summary_distance = compute_weekly_summary(df_time, zone_type)

    # --- Plots ---
    plot_weekly_duration(summary_duration, sport, zone_type)
    plot_weekly_distance(summary_distance, sport)
    plot_efficiency_trend(df_time, sport, metric)

    # --- Table ---
    render_workout_table(df_time, sport, label_speed, ae_col)

#--- Colors ---
color_map_discipline = {
    "Bike": "#2ca02c",   # blue
    "Run": "#ff7f0e",    # orange
    "Swim": "#1f77b4",    # green
    "Other": "#D3D3D3"
}

color_map_HR = {
    "1": "#808080",
    "2": "#636EFA",
    "3": "#00CC96",
    "4": "#FFA15A",
    "5": "#EF553B",
    "Unknown": "#FFFFFF",       # grey
    "Z1 - Recovery": "#808080",
    "Z2 - Endurance": "#636EFA",
    "Z3 - Tempo": "#00CC96",
    "Z4 - Threshold": "#FFA15A",
    "Z5 - VO²Max": "#EF553B"
}


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
overview_tab, health_tab, swim_tab, bike_tab, run_tab = st.tabs([
    "Overview", "Health", "Swim", "Bike", "Run"])

# Overview
with overview_tab:
    st.markdown("### Training overview")
    timeframe_label = st.selectbox("Global timeframe", list(timeframe_days.keys()), key="tf_overview")
    days = timeframe_days[timeframe_label]
    df_time = filter_timeframe(df_activities, days)
    # Pie charts: discipline share of duration and distance
    summary_disc = summarize_by(df_time, "Discipline", None)  # Already filtered by timeframe
    fig_time = go.Figure(go.Pie(
        labels=summary_disc["Discipline"],
        values=summary_disc["movingDuration"],
        hole=0.4,
        title=f"{timeframe_label} – Time spent (hr)",
        marker=dict(colors=[color_map_discipline[d] for d in summary_disc["Discipline"]])
    ))
    fig_dist = go.Figure(go.Pie(
        labels=summary_disc["Discipline"],
        values=summary_disc["distance"],
        hole=0.4,
        title=f"{timeframe_label} – Distance (km)",
        marker=dict(colors=[color_map_discipline[d] for d in summary_disc["Discipline"]])
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















