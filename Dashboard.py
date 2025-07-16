# === Imports ===
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
import ast

# === Functions ===
def classify_training_zone(row):
    z1 = row.get('hrTimeInZone_1', 0)
    z2 = row.get('hrTimeInZone_2', 0)
    z3 = row.get('hrTimeInZone_3', 0)
    z4 = row.get('hrTimeInZone_4', 0)
    z5 = row.get('hrTimeInZone_5', 0)
    total = z1 + z2 + z3 + z4 + z5
    if total == 0:
        return 'Unknown'
    aerobic_time = z1 + z2 + z3
    anaerobic_time = z4 + z5
    aerobic_pct = aerobic_time / total
    max_speed = row.get("maxSpeed", 0)
    avg_speed = row.get("averageSpeed", 0)
    speed_ratio = max_speed / avg_speed if avg_speed else 0
    if aerobic_pct >= 0.8 and z1 > z2 and z1 > z3:
        return "Zone 1"
    elif aerobic_pct >= 0.8 and z2 > z1 and z2 > z3:
        return "Zone 2"
    elif aerobic_pct >= 0.8 and z3 > z1 and z3 > z2:
        return "Zone 3"
    elif anaerobic_time > 0 and speed_ratio >= 1.5:
        return "Anaerobic"
    else:
        return "Unknown"

color_map = {
    "Zone 1": "#cfcfcf",
    "Zone 2": "#1f77b4",
    "Zone 3": "#00cc96",
    "Anaerobic": "#ef553b",
    "Unknown":  "#9b59b6"
}

def add_week_and_year_columns(df, date_column='startTimeLocal'):
    df[date_column] = pd.to_datetime(df[date_column], dayfirst=True, errors='coerce')
    df['Week+Jaar'] = (
        df[date_column].dt.isocalendar().year.astype(str)
        + "-W"
        + df[date_column].dt.isocalendar().week.astype(str).str.zfill(2)
    )
    df['Week_Index'] = df[date_column].dt.isocalendar().year * 53 + df[date_column].dt.isocalendar().week
    return df

def speed_to_tempo_min_per_km(speed_m_s):
    if speed_m_s is None or speed_m_s == 0:
        return None
    return (1000 / speed_m_s) / 60

def format_tempo_min_per_km(tempo):
    if tempo is None:
        return None
    minutes = int(tempo)
    seconds = int(round((tempo - minutes) * 60))
    return f"{minutes}:{seconds:02d}"

def convert_seconds_to_minutes(seconds):
    try:
        return seconds / 60 if pd.notnull(seconds) else None
    except:
        return None

# === Load Data ===
df_activities = pd.read_csv("garmin_full_activities.csv")
df_activities["startTimeLocal"] = df_activities["startTimeLocal"].astype(str).str.strip()
df_activities["startTimeLocal"] = pd.to_datetime(df_activities["startTimeLocal"], errors="coerce")
df_activities["activityType_dict"] = df_activities["activityType"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {})
activity_type_df = df_activities["activityType_dict"].apply(pd.Series)
df_activities = pd.concat([df_activities, activity_type_df], axis=1).drop(columns=["activityType", "activityType_dict"])

df_health = pd.read_csv("garmin_metrics_log.csv")
df_health["date"] = pd.to_datetime(df_health["date"], errors="coerce").dt.strftime("%Y-%m-%d")

# === run data
df_lopen = df_activities[df_activities["typeKey"].isin(["running", "trail_running"])]
df_lopen["Training zone"] = df_lopen.apply(classify_training_zone, axis=1)
df_lopen = add_week_and_year_columns(df_lopen)
df_lopen["Tempo_from_speed"] = df_lopen["averageSpeed"].apply(speed_to_tempo_min_per_km)
df_lopen["Tempo_str"] = df_lopen["Tempo_from_speed"].apply(format_tempo_min_per_km)
df_lopen["Tijd bewogen_num"] = df_lopen["movingDuration"].apply(convert_seconds_to_minutes)
df_lopen["AE"] = 1000 / (df_lopen["Tempo_from_speed"] * df_lopen["averageHR"])
df_lopen["distance"] = df_lopen["distance"] / 1000

# === bike data
df_fiets = df_activities[df_activities["typeKey"].isin(["cycling", "indoor_cycling"])]
df_fiets['Training zone'] = df_fiets.apply(classify_training_zone, axis=1)
df_fiets = add_week_and_year_columns(df_fiets)
df_fiets["Tijd bewogen_num"] = df_fiets["movingDuration"].apply(convert_seconds_to_minutes)
df_fiets["distance"] = df_fiets["distance"] / 1000
df_fiets["Tempo_from_speed"] = df_fiets["averageSpeed"].apply(speed_to_tempo_min_per_km)
df_fiets["AE"] = 1000 / (df_fiets["Tempo_from_speed"] * df_fiets["averageHR"])

# === swim data
df_swim = df_activities[df_activities["typeKey"].isin(["open_water_swimming", "lap_swimming"])]
df_swim['Training zone'] = df_swim.apply(classify_training_zone, axis=1)
df_swim = add_week_and_year_columns(df_swim)
df_swim["Tijd bewogen_num"] = df_swim["movingDuration"].apply(convert_seconds_to_minutes)
df_swim["distance"] = df_swim["distance"] / 1000
df_swim["Tempo_from_speed"] = df_swim["averageSpeed"].apply(speed_to_tempo_min_per_km)
df_swim["AE"] = 1000 / (df_swim["Tempo_from_speed"] * df_swim["averageHR"])

# === setup Tab ===
st.set_page_config(layout="wide")

overview_tab, run_tab, bike_tab, swim_tab = st.tabs(["📊 Overzicht", "🏃‍♂️ Loopactiviteiten", "🚴 Fietsactiviteiten", "🏊 Zwemactiviteiten"])

# === Overzicht Tab ===
with overview_tab:
    st.title("📊 Algemeen Overzicht")

    st.markdown("### 🥧 Verdeling beweegtijd per sport")

    tijd_per_sport = (df_activities[df_activities["typeKey"].isin(["running", "trail_running", "cycling", "indoor_cycling", "lap_swimming", "open_water_swimming"])]
    .assign(Sport=lambda df: df["typeKey"].replace({
        "running": "Lopen",
        "trail_running": "Lopen",
        "cycling": "Fietsen",
        "indoor_cycling": "Fietsen",
        "lap_swimming": "Zwemmen",
        "open_water_swimming": "Zwemmen"
    }))
    .groupby("Sport")["movingDuration"]
    .sum()
    .div(60*60)  # omzetten naar uren
    )

    fig_pie = px.pie(
        names=tijd_per_sport.index,
        values=tijd_per_sport.values,
        title="Tijdsverdeling per sport (in uren)",
        hole=0.4
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        fig_Sleep = px.scatter(
            df_health,
            x="date",
            y="sleep_score",
            title="Slaap score")
        st.plotly_chart(fig_Sleep, use_container_width=True)
    with col2:
        fig_RestingHR = px.scatter(
            df_health,
            x="date",
            y="resting_hr",
            title="Hartslag in rust")
        st.plotly_chart(fig_RestingHR, use_container_width=True)
    with col3:
        fig_avg_hrv = px.scatter(
            df_health,
            x="date",
            y="avg_hrv",
            title="Gemiddelde nachtelijke HRV")
        st.plotly_chart(fig_avg_hrv, use_container_width= True)

    cutoff_date = datetime.now() - timedelta(days=7)
    df_lopen_7d = df_lopen[df_lopen["startTimeLocal"] >= cutoff_date]
    df_fiets_7d = df_fiets[df_fiets["startTimeLocal"] >= cutoff_date]
    df_swim_7d = df_swim[df_swim["startTimeLocal"] >= cutoff_date]

    col4, col5, col6 = st.columns(3)
    with col4:
        lopen_zone_data = df_lopen_7d.groupby("Training zone")["Tijd bewogen_num"].sum().reset_index()
        fig_lopen = px.pie(lopen_zone_data, names="Training zone", values="Tijd bewogen_num", 
            title="🏃‍♂️ Verdeling tijd per zone (Lopen)", color_discrete_map=color_map)
        st.plotly_chart(fig_lopen, use_container_width=True)

    with col5:
        fiets_zone_data = df_fiets_7d.groupby("Training zone")["Tijd bewogen_num"].sum().reset_index()
        fig_fiets = px.pie(fiets_zone_data, names="Training zone", values="Tijd bewogen_num", 
            title="🚴‍♂️ Verdeling tijd per zone (Fietsen)", color_discrete_map=color_map)
        st.plotly_chart(fig_fiets, use_container_width=True)

    with col6:
        zwem_zone_data = df_swim_7d.groupby("Training zone")["Tijd bewogen_num"].sum().reset_index()
        fig_zwem = px.pie(zwem_zone_data, names="Training zone", values="Tijd bewogen_num", 
            title="🏊‍♂️ Verdeling tijd per zone (Zwemmen)", color_discrete_map=color_map)
        st.plotly_chart(fig_zwem, use_container_width=True)
# === Loopactiviteiten Tab ===
with run_tab:
    st.title("🏃‍♂️ Persoonlijk Loopdashboard")

    min_week = df_lopen['Week_Index'].min()
    max_week = df_lopen['Week_Index'].max()
    week_range = pd.DataFrame({"Week_Index": range(min_week, max_week + 1)})
    week_range["Jaar"] = week_range["Week_Index"] // 53
    week_range["Week"] = week_range["Week_Index"] % 53
    week_range["Week+Jaar"] = week_range["Jaar"].astype(str) + "-W" + week_range["Week"].astype(str).str.zfill(2)

    all_zones = df_lopen["Training zone"].unique()
    zone_df = pd.DataFrame({"Training zone": all_zones})
    full_weeks_zones = week_range.assign(key=1).merge(zone_df.assign(key=1), on="key").drop("key", axis=1)

    WeekTotaal = (
        df_lopen.groupby(["Week+Jaar", "Training zone"])
        .agg({"Tijd bewogen_num": "sum", "distance": "sum"})
        .reset_index()
    )
    WeekTotaal = pd.merge(full_weeks_zones, WeekTotaal, on=["Week+Jaar", "Training zone"], how="left")
    WeekTotaal = WeekTotaal.sort_values("Week_Index")
    for col in ["Tijd bewogen_num", "distance"]:
        WeekTotaal[col] = WeekTotaal[col].fillna(0)

    filtered_totaal_weeks = WeekTotaal.copy()
    zone_order = sorted(df_lopen["Training zone"].dropna().unique())
    filtered_totaal_weeks["Tijd_uren"] = filtered_totaal_weeks["Tijd bewogen_num"] / 60

    col1, col2 = st.columns(2)
    with col1:
        fig_Tijd = px.bar(
            filtered_totaal_weeks,
            x="Week+Jaar",
            y="Tijd_uren",
            color="Training zone",
            title="🕒 Totale tijd per week",
            color_discrete_map=color_map,
            category_orders={"Training zone": zone_order},
        )
        st.plotly_chart(fig_Tijd, use_container_width=True)
    with col2:
        fig_Afstand = px.bar(
            filtered_totaal_weeks,
            x="Week+Jaar",
            y="distance",
            color="Training zone",
            title="📏 Totale afstand per week",
            color_discrete_map=color_map,
            category_orders={"Training zone": zone_order},
        )
        st.plotly_chart(fig_Afstand, use_container_width=True)

    st.markdown("🎯 **Kies trainingszones:**")
    zone_options = sorted(df_lopen["Training zone"].dropna().unique())
    cols = st.columns(len(zone_options))
    selected_zones = []
    for col, zone in zip(cols, zone_options):
        if col.checkbox(zone, value=True):
            selected_zones.append(zone)

    df_filtered = df_lopen[df_lopen["Training zone"].isin(selected_zones)]
    weekGemiddelde = (
        df_filtered.groupby(["Week+Jaar"])
        .agg({"averageHR": "mean", "Tempo_from_speed": "mean", "AE": "mean", "averageRunningCadenceInStepsPerMinute": "mean"})
        .reset_index()
    )
    weekGemiddelde = pd.merge(week_range, weekGemiddelde, on="Week+Jaar", how="left").sort_values("Week_Index")

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        fig_hr = px.line(weekGemiddelde, x="Week+Jaar", y="averageHR", markers=True, title="❤️ Gemiddelde Hartslag")
        fig_hr.update_traces(line=dict(color="red"))
        fig_hr.update_layout(yaxis_title="BPM")
        st.plotly_chart(fig_hr, use_container_width=True)
    with col4:
        fig_tempo = px.line(weekGemiddelde, x="Week+Jaar", y="Tempo_from_speed", markers=True, title="⏱️ Gemiddeld Tempo")
        fig_tempo.update_traces(line=dict(color="blue"))
        fig_tempo.update_layout(yaxis_title="min/km")
        st.plotly_chart(fig_tempo, use_container_width=True)
    with col5:
        fig_AE = px.line(weekGemiddelde, x="Week+Jaar", y="AE", markers=True, title="⚡ AE per week")
        fig_AE.update_traces(line=dict(color="green"))
        fig_AE.update_layout(yaxis_title="m/HS")
        st.plotly_chart(fig_AE, use_container_width=True)
    with col6:
        fig_Cadans = px.line(weekGemiddelde, x="Week+Jaar", y="averageRunningCadenceInStepsPerMinute", markers=True, title="🏃‍♂️ Cadans per week")
        fig_Cadans.update_traces(line=dict(color="orange"))
        fig_Cadans.update_layout(yaxis_title="Steps/min")
        st.plotly_chart(fig_Cadans, use_container_width=True)

    st.subheader("📋 Overzicht van loopactiviteiten")
    columns_to_show = [
        "activityId", "activityName", "startTimeLocal", "typeKey", "distance",
        "movingDuration", "elevationGain", "averageSpeed", "maxSpeed",
        "averageHR", "maxHR", "averageRunningCadenceInStepsPerMinute", "maxRunningCadenceInStepsPerMinute"
    ]
    renamed_columns = {
        "activityId": "ID",
        "activityName": "Naam",
        "startTimeLocal": "Starttijd",
        "typeKey": "Type",
        "distance": "Afstand (km)",
        "movingDuration": "Beweegtijd (s)",
        "elevationGain": "Hoogtewinst (m)",
        "averageSpeed": "Gem. snelheid (m/s)",
        "maxSpeed": "Max snelheid (m/s)",
        "averageHR": "Gem. HS",
        "maxHR": "Max HS",
        "averageRunningCadenceInStepsPerMinute": "Gem. cadans",
        "maxRunningCadenceInStepsPerMinute": "Max cadans"
    }
    valid_columns = [col for col in columns_to_show if col in df_lopen.columns]
    df_table = df_lopen[valid_columns].rename(columns={k: v for k, v in renamed_columns.items() if k in valid_columns})
    st.dataframe(df_table.sort_values("Starttijd", ascending=False), use_container_width=True)

# === Fietsactiviteiten Tab ===
with bike_tab:
    st.title("🚴 Persoonlijk Fietsdashboard")

    WeekTotaal_fiets = (
        df_fiets.groupby(["Week+Jaar", "Training zone"])
        .agg({"Tijd bewogen_num": "sum", "distance": "sum"})
        .reset_index()
    )
    WeekTotaal_fiets = pd.merge(full_weeks_zones, WeekTotaal_fiets, on=["Week+Jaar", "Training zone"], how="left")
    WeekTotaal_fiets = WeekTotaal_fiets.sort_values("Week_Index")
    for col in ["Tijd bewogen_num", "distance"]:
        WeekTotaal_fiets[col] = WeekTotaal_fiets[col].fillna(0)
    WeekTotaal_fiets["Tijd_uren"] = WeekTotaal_fiets["Tijd bewogen_num"] / 60

    col1, col2 = st.columns(2)
    with col1:
        fig_Tijd_fiets = px.bar(
            WeekTotaal_fiets,
            x="Week+Jaar",
            y="Tijd_uren",
            color="Training zone",
            title="🕒 Totale fietstijd per week",
            color_discrete_map=color_map,
            category_orders={"Training zone": zone_order},
        )
        st.plotly_chart(fig_Tijd_fiets, use_container_width=True)
    with col2:
        fig_Afstand_fiets = px.bar(
            WeekTotaal_fiets,
            x="Week+Jaar",
            y="distance",
            color="Training zone",
            title="📏 Totale afstand per week",
            color_discrete_map=color_map,
            category_orders={"Training zone": zone_order},
        )
        st.plotly_chart(fig_Afstand_fiets, use_container_width=True)

        
    df_outdoor = df_fiets[df_fiets["typeKey"] == "cycling"]
    df_indoor = df_fiets[df_fiets["typeKey"] == "indoor_cycling"]

    st.markdown("### 📊 Outdoor Cycling (filter op trainingszones)")
    zone_options_outdoor = sorted(df_outdoor["Training zone"].dropna().unique())
    cols = st.columns(len(zone_options_outdoor))
    selected_zones_outdoor = []
    for col, zone in zip(cols, zone_options_outdoor):
        if col.checkbox(f"Outdoor - {zone}", value=True):
            selected_zones_outdoor.append(zone)

    df_outdoor_filtered = df_outdoor[df_outdoor["Training zone"].isin(selected_zones_outdoor)]
    weekGem_outdoor = (
        df_outdoor_filtered.groupby(["Week+Jaar"])
        .agg({"averageHR": "mean", "Tempo_from_speed": "mean", "AE": "mean", "averageBikingCadenceInRevPerMinute": "mean"})
        .reset_index()
    )
    weekGem_outdoor = pd.merge(week_range, weekGem_outdoor, on="Week+Jaar", how="left").sort_values("Week_Index")

    col3, col4, col5, col6 = st.columns(4)
    with col3:
        fig_hr = px.line(weekGem_outdoor, x="Week+Jaar", y="averageHR", markers=True, title="❤️ Gemiddelde Hartslag (Outdoor)")
        fig_hr.update_traces(line=dict(color="red"))
        fig_hr.update_layout(yaxis_title="BPM")
        st.plotly_chart(fig_hr, use_container_width=True)
    with col4:
        fig_tempo = px.line(weekGem_outdoor, x="Week+Jaar", y="Tempo_from_speed", markers=True, title="⏱️ Gemiddeld Tempo (Outdoor)")
        fig_tempo.update_traces(line=dict(color="blue"))
        fig_tempo.update_layout(yaxis_title="min/km")
        st.plotly_chart(fig_tempo, use_container_width=True)
    with col5:
        fig_AE = px.line(weekGem_outdoor, x="Week+Jaar", y="AE", markers=True, title="⚡ AE per week (Outdoor)")
        fig_AE.update_traces(line=dict(color="green"))
        fig_AE.update_layout(yaxis_title="m/HS")
        st.plotly_chart(fig_AE, use_container_width=True)
    with col6:
        fig_Cadans = px.line(weekGem_outdoor, x="Week+Jaar", y="averageBikingCadenceInRevPerMinute", markers=True, title="🚴 Cadans per week (Outdoor)")
        fig_Cadans.update_traces(line=dict(color="orange"))
        fig_Cadans.update_layout(yaxis_title="RPM")
        st.plotly_chart(fig_Cadans, use_container_width=True)

    st.markdown("### 🏠 Indoor Cycling (filter op trainingszones)")
    zone_options_indoor = sorted(df_indoor["Training zone"].dropna().unique())
    cols = st.columns(len(zone_options_indoor))
    selected_zones_indoor = []
    for col, zone in zip(cols, zone_options_indoor):
        if col.checkbox(f"Indoor - {zone}", value=True):
            selected_zones_indoor.append(zone)

    df_indoor_filtered = df_indoor[df_indoor["Training zone"].isin(selected_zones_indoor)]
    weekGem_indoor = (
        df_indoor_filtered.groupby(["Week+Jaar"])
        .agg({"averageHR": "mean", "Tempo_from_speed": "mean", "AE": "mean", "averageBikingCadenceInRevPerMinute": "mean"})
        .reset_index()
    )
    weekGem_indoor = pd.merge(week_range, weekGem_indoor, on="Week+Jaar", how="left").sort_values("Week_Index")

    col7, col8, col9, col10 = st.columns(4)
    with col7:
        fig_hr_in = px.line(weekGem_indoor, x="Week+Jaar", y="averageHR", markers=True, title="❤️ Gemiddelde Hartslag (Indoor)")
        fig_hr_in.update_traces(line=dict(color="red"))
        fig_hr_in.update_layout(yaxis_title="BPM")
        st.plotly_chart(fig_hr_in, use_container_width=True)
    with col8:
        fig_tempo_in = px.line(weekGem_indoor, x="Week+Jaar", y="Tempo_from_speed", markers=True, title="⏱️ Gemiddeld Tempo (Indoor)")
        fig_tempo_in.update_traces(line=dict(color="blue"))
        fig_tempo_in.update_layout(yaxis_title="min/km")
        st.plotly_chart(fig_tempo_in, use_container_width=True)
    with col9:
        fig_AE_in = px.line(weekGem_indoor, x="Week+Jaar", y="AE", markers=True, title="⚡ AE per week (Indoor)")
        fig_AE_in.update_traces(line=dict(color="green"))
        fig_AE_in.update_layout(yaxis_title="m/HS")
        st.plotly_chart(fig_AE_in, use_container_width=True)
    with col10:
        fig_Cadans_in = px.line(weekGem_indoor, x="Week+Jaar", y="averageBikingCadenceInRevPerMinute", markers=True, title="🚴 Cadans per week (Indoor)")
        fig_Cadans_in.update_traces(line=dict(color="orange"))
        fig_Cadans_in.update_layout(yaxis_title="RPM")
        st.plotly_chart(fig_Cadans_in, use_container_width=True)

    st.subheader("📄 Overzicht van fietsactiviteiten")
    columns_to_show_bike = [
        "activityId", "activityName", "startTimeLocal", "typeKey", "distance",
        "movingDuration", "elevationGain", "averageSpeed", "maxSpeed",
        "averageHR", "maxHR"
    ]
    renamed_columns_bike = {
        "activityId": "ID",
        "activityName": "Naam",
        "startTimeLocal": "Starttijd",
        "typeKey": "Type",
        "distance": "Afstand (km)",
        "movingDuration": "Beweegtijd (s)",
        "elevationGain": "Hoogtewinst (m)",
        "averageSpeed": "Gem. snelheid (m/s)",
        "maxSpeed": "Max snelheid (m/s)",
        "averageHR": "Gem. HS",
        "maxHR": "Max HS"
    }
    valid_cols_bike = [col for col in columns_to_show_bike if col in df_fiets.columns]
    df_table_bike = df_fiets[valid_cols_bike].rename(columns={k: v for k, v in renamed_columns_bike.items() if k in valid_cols_bike})
    st.dataframe(df_table_bike.sort_values("Starttijd", ascending=False), use_container_width=True)

# === Zwemactiviteiten Tab ===
with swim_tab:
    st.title("🏊 Persoonlijk Zwemdashboard")

    WeekTotaal_swim = (
        df_swim.groupby(["Week+Jaar", "Training zone"])
        .agg({"Tijd bewogen_num": "sum", "distance": "sum"})
        .reset_index()
    )
    WeekTotaal_swim = pd.merge(full_weeks_zones, WeekTotaal_swim, on=["Week+Jaar", "Training zone"], how="left")
    WeekTotaal_swim = WeekTotaal_swim.sort_values("Week_Index")
    for col in ["Tijd bewogen_num", "distance"]:
        WeekTotaal_swim[col] = WeekTotaal_swim[col].fillna(0)
    WeekTotaal_swim["Tijd_uren"] = WeekTotaal_swim["Tijd bewogen_num"] / 60

    col1, col2 = st.columns(2)
    with col1:
        fig_Tijd_swim = px.bar(
            WeekTotaal_swim,
            x="Week+Jaar",
            y="Tijd_uren",
            color="Training zone",
            title="🕒 Totale zwemtijd per week",
            color_discrete_map=color_map,
        )
        st.plotly_chart(fig_Tijd_swim, use_container_width=True)
    with col2:
        fig_Afstand_swim = px.bar(
            WeekTotaal_swim,
            x="Week+Jaar",
            y="distance",
            color="Training zone",
            title="📏 Totale afstand per week",
            color_discrete_map=color_map,
        )
        st.plotly_chart(fig_Afstand_swim, use_container_width=True)

    df_openwater = df_swim[df_swim["typeKey"] == "open_water_swimming"]
    df_lapswim = df_swim[df_swim["typeKey"] == "lap_swimming"]

    for swim_type, df_part, label in [
        ("Open Water", df_openwater, "Open Water"),
        ("Zwemmen (Baantjes)", df_lapswim, "Lap")
    ]:
        st.markdown(f"### 📊 {swim_type} (filter op trainingszones)")
        zones = sorted(df_part["Training zone"].dropna().unique())
        cols = st.columns(len(zones))
        selected = [zone for col, zone in zip(cols, zones) if col.checkbox(f"{label} - {zone}", value=True)]

        df_filtered = df_part[df_part["Training zone"].isin(selected)]
        weekGem = (
            df_filtered.groupby("Week+Jaar")
            .agg({"averageHR": "mean", "Tempo_from_speed": "mean", "AE": "mean"})
            .reset_index()
        )
        weekGem = pd.merge(week_range, weekGem, on="Week+Jaar", how="left").sort_values("Week_Index")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.plotly_chart(px.line(weekGem, x="Week+Jaar", y="averageHR", markers=True, title=f"❤️ Gem. Hartslag ({label})"), use_container_width=True)
        with c2:
            st.plotly_chart(px.line(weekGem, x="Week+Jaar", y="Tempo_from_speed", markers=True, title=f"⏱️ Gem. Tempo ({label})"), use_container_width=True)
        with c3:
            st.plotly_chart(px.line(weekGem, x="Week+Jaar", y="AE", markers=True, title=f"⚡ AE ({label})"), use_container_width=True)

    st.subheader("📄 Overzicht van zwemactiviteiten")
    columns_to_show_swim = [
        "activityId", "activityName", "startTimeLocal", "typeKey", "distance",
        "movingDuration", "averageSpeed", "maxSpeed", "averageHR", "maxHR"
    ]
    renamed_columns_swim = {
        "activityId": "ID",
        "activityName": "Naam",
        "startTimeLocal": "Starttijd",
        "typeKey": "Type",
        "distance": "Afstand (km)",
        "movingDuration": "Beweegtijd (s)",
        "averageSpeed": "Gem. snelheid (m/s)",
        "maxSpeed": "Max snelheid (m/s)",
        "averageHR": "Gem. HS",
        "maxHR": "Max HS"
    }
    valid_cols_swim = [col for col in columns_to_show_swim if col in df_swim.columns]
    df_table_swim = df_swim[valid_cols_swim].rename(columns={k: v for k, v in renamed_columns_swim.items() if k in valid_cols_swim})
    st.dataframe(df_table_swim.sort_values("Starttijd", ascending=False), use_container_width=True)
