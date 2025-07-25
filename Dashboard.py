# === Imports ===
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import ast
import numpy as np
import math

# === Functions ===
def classify_discipline(type_key): # Opdelen van activities per sport in nieuwe kolom
    if type_key in ["running", "trail_running", "treadmill_running"]:
        return "Run"
    elif type_key in ["cycling", "indoor_cycling"]:
        return "Bike"
    elif type_key in ["open_water_swimming", "lap_swimming"]:
        return "Swim"
    else:
        return "Other"  # optional for anything else
    
def convert_pace(row): # Omzetten van snelheid naar pace (sportspecifiek)
    speed = row['averageSpeed']
    if row['Discipline'] == 'Run':
        return 1000 / speed / 60  # min/km
    elif row['Discipline'] == 'Bike':
        return speed * 3.6        # km/h
    elif row['Discipline'] == 'Swim':
        return 100 / speed / 60  # min/100m
    else:
        return speed  # fallback: keep raw m/s

def convert_max_pace(row): # Omzetten van max. snelheid naar pace (sportspecifiek)
    speed = row['maxSpeed']
    if row['Discipline'] == 'Run':
        return 1000 / speed / 60
    elif row['Discipline'] == 'Bike':
        return speed * 3.6
    elif row['Discipline'] == 'Swim':
        return 100 / speed / 60
    else:
        return speed

def classify_workout(row):
    """Classify workout based on HR zones and training effect metrics."""
    
    z1 = row.get('hrTimeInZone_1', 0)
    z2 = row.get('hrTimeInZone_2', 0)
    z3 = row.get('hrTimeInZone_3', 0)
    z4 = row.get('hrTimeInZone_4', 0)
    z5 = row.get('hrTimeInZone_5', 0)
    # -- look at training effect
    ae = row.get('aerobicTrainingEffect', 0)
    an = row.get('anaerobicTrainingEffect', 0)
    # Priority 1: VO2max efforts (high anaerobic load + z5 time)
    if z5 > 60 and an > 2.5:
        return 'Z5 - VO²Max'
    # Priority 2: Threshold training (substantial z4 time and high TE)
    elif z4 > 180 and ae >= 2.5 and an > 1.0:
        return 'Z4 - Threshold'
    # Priority 3: Tempo (z3 dominant, moderate TE)
    elif z3 > max(z1, z2):
        return 'Z3 - Tempo'
    # Priority 4: Endurance (z2 dominant)
    elif z2 > max(z3, z1):
        return 'Z2 - Endurance'
    # Priority 5: Recovery (mostly z1)
    elif z1 > max(z2, z3, z4, z5):
        return 'Z1 - Recovery'
    else:
        return 'Unknown'

workout_zone_colors = {
    'Z5 - VO²Max': '#d62728',     # Fel rood (intensief)
    'Z4 - Threshold': '#ff7f0e',  # Oranje
    'Z3 - Tempo': '#2ca02c',      # Groen
    'Z2 - Endurance': '#1f77b4',  # Blauw
    'Z1 - Recovery': '#7f7f7f',   # Grijs
    'Unknown': '#bd22b8'          # Paars
    }

def summarize_timeDistance_discipline(df, days=None): # sommeren van afstand en tijd per discipline
    if days is not None:
        df = df[df["startTimeLocal"] >= datetime.now() - timedelta(days=days)]
    summary = (
        df.groupby("Discipline")[["movingDuration", "distance"]]
        .sum()
        .reset_index()
    )
    return summary

def summarize_timeDistance_zone(df, days=None): # sommeren van afstand en tijd per discipline
    if days is not None:
        df = df[df["startTimeLocal"] >= datetime.now() - timedelta(days=days)]
    summary = (
        df.groupby("classifiedZone")[["movingDuration", "distance"]]
        .sum()
        .reset_index()
    )
    return summary

timeframe_days = { # timeframe omzetten in nummeriek getal
    "Last week": 7,
    "Last month": 30,
    "Last year": 365,
    "All time": None}

discipline_colors = { # kleuren voor disciplines
    "Run": "#3FD200",
    "Bike": "#F30909",
    "Swim": "#1731C5",
    "Other": "#707070"
}



def format_hours_to_hhmm(hours):
    total_minutes = int(hours * 60)
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"

def format_pace_to_minsec(pace):
    total_seconds = round(pace * 60)
    return f"{total_seconds // 60:02d}:{total_seconds % 60:02d}"

# === Data managment ===
df_activities = pd.read_csv("garmin_full_activities.csv") # import activities data (locally)
df_health = pd.read_csv("garmin_metrics_log.csv") # import health data (locally)

# === Time columns ===
df_activities['startTimeLocal'] = pd.to_datetime(df_activities["startTimeLocal"], format='%Y-%m-%d %H:%M:%S') # kolom omzetten naar een python tijd
df_activities['Day'] = df_activities['startTimeLocal'].dt.isocalendar().day # toevoegen dag kolom
df_activities['Week'] = df_activities['startTimeLocal'].dt.isocalendar().week # toevoegen weeknummer
df_activities['Year'] = df_activities['startTimeLocal'].dt.isocalendar().year # toevoegen jaarnummer
df_activities["YearWeek"] = df_activities["Year"].astype(str) + "-W" + df_activities["Week"].astype(str)
df_activities['WeekIndex'] = df_activities['startTimeLocal'].dt.isocalendar().year * 53 + df_activities["startTimeLocal"].dt.isocalendar().week
df_health['date'] = pd.to_datetime(df_health['date'], format='%Y-%m-%d') # kolom omzetten naar een python tijd
df_health = df_health.sort_values("date") # sorteren van de data
df_health['Week'] = df_health['date'].dt.isocalendar().week
df_health['Year'] = df_health['date'].dt.isocalendar().year
# === Making of the Yearweek df
Min_date = df_activities["startTimeLocal"].min()
Max_date = df_activities["startTimeLocal"].max()
All_weeks = pd.date_range(start=Min_date, end=Max_date, freq="W-MON")
isocal_df = All_weeks.isocalendar()
df_week = pd.DataFrame({
    "Year": isocal_df["year"],
    "Week": isocal_df["week"]
}).drop_duplicates()

# === Unpacking the activity dictonary into columns
df_activities["activityType_dict"] = df_activities["activityType"].apply(lambda x: ast.literal_eval(x) if pd.notnull(x) else {} ) # kolom omzetten in een dictionary
activity_type_df = df_activities["activityType_dict"].apply(pd.Series)  # Zet elke dictionary om tot verschillende kolommen (in een nieuwe df)
df_activities = pd.concat([df_activities, activity_type_df], axis=1).drop(columns=["activityType", "activityType_dict"])  # mergen van originele df met nieuwe df

# === Removing and reforming df_activities
df_activities = df_activities.drop(columns=['activityId','startTimeGMT','eventType','duration',
                                            'elapsedDuration','startLatitude','startLongitude',
                                            'hasPolyline','hasImages','ownerId','ownerDisplayName',
                                            'ownerFullName','ownerProfileImageUrlSmall','ownerProfileImageUrlMedium',
                                            'ownerProfileImageUrlLarge','bmrCalories','steps','userRoles',
                                            'privacy','userPro','hasVideo','timeZoneId','beginTimestamp',
                                            'sportTypeId','deviceId','minElevation','maxElevation','maxDoubleCadence',
                                            'summarizedDiveInfo','maxVerticalSpeed','manufacturer','locationName',
                                            'lapCount','endLatitude','endLongitude','waterEstimated','minRespirationRate',
                                            'maxRespirationRate','avgRespirationRate','trainingEffectLabel','minActivityLapDuration',
                                            'splitSummaries','hasSplits','moderateIntensityMinutes','vigorousIntensityMinutes','avgGradeAdjustedSpeed',
                                            'differenceBodyBattery','hasHeatMap','fastestSplit_1000','fastestSplit_1609','fastestSplit_5000',
                                            'endTimeGMT','qualifyingDive','purposeful','manualActivity','autoCalcCalories','elevationCorrected',
                                            'atpActivity','favorite','decoDive','pr','parent','strokes','avgStrokeDistance',
                                            'minTemperature','maxTemperature','courseId','summarizedExerciseSets','totalSets','activeSets','totalReps',
                                            'vO2MaxValue','max20MinPower','trainingStressScore','intensityFactor','maxAvgPower_1','maxAvgPower_2','maxAvgPower_5','maxAvgPower_10','maxAvgPower_20','maxAvgPower_30','maxAvgPower_60','maxAvgPower_120','maxAvgPower_300','maxAvgPower_600','maxAvgPower_1200','maxAvgPower_1800','maxAvgPower_3600',
                                            'excludeFromPowerCurveReports','fastestSplit_40000','powerTimeInZone_6','powerTimeInZone_7',
                                            'fastestSplit_10000','workoutId','fastestSplit_21098','description','avgStress','startStress','endStress','differenceStress','maxStress',
                                            'activeLengths','poolLength','unitOfPoolLength','avgStrokes','fastestSplit_100','grit',
                                            'avgFlow','isHidden','restricted','trimmable'])
df_activities['Discipline'] = df_activities['typeKey'].apply(classify_discipline) # kolom toevoegen dat discipline geeft aan activiteiten
df_activities['sportPace'] = df_activities.apply(convert_pace, axis=1) # kolom met pace toevoegen
df_activities['sportMaxPace'] = df_activities.apply(convert_max_pace, axis=1) # kolom met max pace toevoegen
df_activities['classifiedZone'] = df_activities.apply(classify_workout, axis=1) # kolom met zone toevoegen
df_activities['distance'] = df_activities['distance']/1000 # meters omzetten naar kilometers
df_activities['movingDuration'] = df_activities['movingDuration']/3600 # seconden omzetten naar uur
df_activities["AerobicEfficiency"] = df_activities["sportPace"]*df_activities["averageHR"]
df_activities["AerobicEfficiencyBike"] = df_activities["averageHR"]/(df_activities["sportPace"]/60)
# === Setup Dashboard ===
st.set_page_config(layout="wide") # opstarten van een webpage
overview_tab, health_metric, swim_tab, bike_tab, run_tab = st.tabs(["📊 OVERVIEW", "🩺 HEALTH","🏊 SWIM","🚴 BIKE","🏃‍♂️ RUN"]) # invoegen van de tabs aan de bovenkant van page

st.sidebar.markdown('### **TIMEFRAME**') # sidebalk maken voor de optie timeframe
timeframe = st.sidebar.selectbox("Choose timeframe:",["Last week", "Last month", "Last year", "All time"]) # subselectie van data op basis van timeframe

st.sidebar.markdown("### 🏊 **SWIM**") # sidebalk maken voor de optie zwemmen
selected_swim = st.sidebar.radio( # subselectie op basis van keuzen
    "Choose type:",
    options=["Both", "lap_swimming", "open_water_swimming"],
    format_func=lambda x: (
        "Both types" if x == "Both"
        else "Pool Swimming" if x == "lap_swimming"
        else "Open Water Swimming"
    )
)

st.sidebar.markdown("### 🚴 **BIKE**")  # sidebalk maken voor de optie fietsen
selected_bike = st.sidebar.radio(  # subselectie op basis van keuze
    "Choose type:",
    options=["Both", "cycling", "indoor_cycling"],
    format_func=lambda x: (
        "Both types" if x == "Both"
        else "Outdoor Cycling" if x == "cycling"
        else "Indoor Cycling"
    )
)

# Sidebar selection for running type
st.sidebar.markdown("### 🏃 **RUN**")
selected_run = st.sidebar.radio(
    "Choose running type:",
    options=["Both", "outdoor_running", "indoor_running"],
    format_func=lambda x: (
        "Both types" if x == "Both"
        else "Outdoor Running" if x == "outdoor_running"
        else "Indoor Running"
    )
)


## === DF subselections ===
# OVERVIEW
# --- Pie charts
Summary_Overview = summarize_timeDistance_discipline(df_activities, timeframe_days[timeframe]) # aanpasbare timeframe voor totale som activiteiten
# --- Calender
today = datetime.today()
three_months_ago = today - pd.DateOffset(months=2)
start_of_month = pd.to_datetime(f"{three_months_ago.year}-{three_months_ago.month}-01")
df_Calender = df_activities[df_activities["startTimeLocal"] >= start_of_month]
# HEALTH
if timeframe_days[timeframe] is not None: # kiezen hoeveel dagen worden toegevoegd aan de grafiek
    cutoff = datetime.now() - timedelta(days=timeframe_days[timeframe])
    df_health_timeframed = df_health[df_health["date"] >= cutoff]
else:
    df_health_timeframed = df_health.copy()

# SWIM

# === Setup Overview ===
with overview_tab:
    ## === Calender charts ===
    fig_calender = px.scatter(
        df_Calender,
        x=df_Calender["startTimeLocal"].dt.day,
        y=df_Calender["startTimeLocal"].dt.month_name(),
        color="Discipline",
        color_discrete_map=discipline_colors,
        title="🏁 Triathlon Training Calendar (Last 3 Months)",
        custom_data=["Discipline", "startTimeLocal"])
    fig_calender.update_traces(
    hovertemplate="Date: %{customdata[1]|%Y-%m-%d}<extra></extra>")
    fig_calender.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
    st.plotly_chart(fig_calender, use_container_width=True)
    ## === Pie charts ===
    fig_time = go.Figure(go.Pie( # pie chart met gesommeerde tijd
        labels=Summary_Overview["Discipline"],
        values=Summary_Overview["movingDuration"],
        text=Summary_Overview["movingDuration"].apply(format_hours_to_hhmm),
        texttemplate="%{percent} - %{text}",
        marker=dict(colors=[discipline_colors.get(d, "#CCCCCC") for d in Summary_Overview["Discipline"]]),
        hole=0.4))
    fig_time.update_traces(hovertemplate="<extra></extra>")
    fig_time.update_layout(title_text=f"{timeframe} – Time spent (h)")
    fig_distance = go.Figure(go.Pie( # pie chart met gesommeerde afstand
        labels=Summary_Overview["Discipline"],
        values=Summary_Overview["distance"],
        texttemplate="%{percent} - %{value:.2f}",
        marker=dict(colors=[discipline_colors.get(d, "#CCCCCC") for d in Summary_Overview["Discipline"]]),
        hole=0.4))
    fig_distance.update_traces(hovertemplate="<extra></extra>")
    fig_distance.update_layout(title_text=f"{timeframe} – Distance covered (km)")
    st.markdown(f"<h2 style='text-align: center;'>Overview – {timeframe} training</h2>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig_time, use_container_width=True)
    with col2:
        st.plotly_chart(fig_distance, use_container_width=True)

# === Health Overview ===
with health_metric:
    # === Scatter plots ===
    st.markdown(f"<h2 style='text-align: center;'>Health Metrics - {timeframe} </h2>", unsafe_allow_html=True)
    
    
    fig_Sleep = px.line(
        df_health_timeframed,
        x="date",
        y="sleep_score",
        title="😴 Sleep score")
    fig_Sleep.update_layout(xaxis_title="Date",yaxis_title="Sleep score")
    fig_Sleep.update_traces(line=dict(color="#FFD700"), hovertemplate="<extra></extra>")
    st.plotly_chart(fig_Sleep, use_container_width=True)
    
    fig_RestingHR = px.line(
        df_health_timeframed,
        x="date",
        y="resting_hr",
        title="❤️‍🩹 Resting heart rate")
    fig_RestingHR.update_layout(xaxis_title="Date",yaxis_title="Resting heart rate (bpm)")
    fig_RestingHR.update_traces(line=dict(color="#FF00C8"), hovertemplate="<extra></extra>")
    st.plotly_chart(fig_RestingHR, use_container_width=True)
    
    fig_avg_hrv = px.line(
        df_health_timeframed,
        x="date",
        y="avg_hrv",
        title="🔄 Nightly HRV")
    fig_avg_hrv.update_layout(xaxis_title="Date",yaxis_title="HRV")
    fig_avg_hrv.update_traces(line=dict(color="#00FFDD"), hovertemplate="<extra></extra>")
    st.plotly_chart(fig_avg_hrv, use_container_width= True)
    
# === setup SWIM ===
with swim_tab:
    # --- calender zone training
    df_Swim_Calender = df_Calender[df_Calender['Discipline'] == "Swim"]
    df_Swim_Calender["day"] = df_Swim_Calender["startTimeLocal"].dt.day
    df_Swim_Calender["month"] = df_Swim_Calender["startTimeLocal"].dt.month_name()
    # --- pie chart
    if selected_swim == "Both":
        df_swim_selected = df_activities[df_activities["typeKey"].isin(["lap_swimming", "open_water_swimming"])]
    else:
        df_swim_selected = df_activities[df_activities["typeKey"] == selected_swim]

    if timeframe_days[timeframe] is not None: # kiezen hoeveel dagen worden toegevoegd aan de grafiek
        cutoff = datetime.now() - timedelta(days=timeframe_days[timeframe])
        df_swim_timeframed = df_swim_selected[df_swim_selected["startTimeLocal"] >= cutoff]
    else:
        df_swim_timeframed = df_swim_selected.copy()

    Summary_Swim_Overview = summarize_timeDistance_zone(df_swim_timeframed, timeframe_days[timeframe])
    # --- grouped bar chart
    Summary_Swim_Week = (
        df_swim_timeframed
        .groupby(["WeekIndex","YearWeek", "Year", "Week", "typeKey", "classifiedZone"])
        .agg({
            "movingDuration": "sum",
            "distance": "sum"})
        .reset_index())
    Summary_Swim_Week = pd.merge(df_week, Summary_Swim_Week, on=["Week", "Year"], how="left")
    Summary_Swim_Week = Summary_Swim_Week.sort_values("WeekIndex")
    for col in ["movingDuration", "distance"]:
        Summary_Swim_Week[col] = Summary_Swim_Week[col].fillna(0)
    # --- scatter plots
    max_val_time = Summary_Swim_Week.groupby(["Year", "Week"])["movingDuration"].sum().max()
    rounded_max_time = math.ceil(max_val_time / 0.083) * 0.083
    tickvals_time = np.linspace(0, rounded_max_time, 8)
    ticktext_time = [format_hours_to_hhmm(p) for p in tickvals_time]

    max_val_pace = df_swim_timeframed['sportPace'].max()
    rounded_max_pace = math.ceil(max_val_pace / 0.083) * 0.083
    tickvals_pace = np.linspace(0, rounded_max_pace, 8)
    ticktext_pace = [format_pace_to_minsec(p) for p in tickvals_pace]

    max_val_maxpace = df_swim_timeframed['sportMaxPace'].max()
    rounded_max_maxpace = math.ceil(max_val_maxpace / 0.083) * 0.083
    tickvals_maxpace = np.linspace(0, rounded_max_maxpace, 8)
    ticktext_maxpace = [format_pace_to_minsec(p) for p in tickvals_maxpace]
    # --- table
    columns_to_show_swim = [
        "activityName", "startTimeLocal","YearWeek", "typeKey", "distance",
        "classifiedZone","movingDuration", "sportPace", "sportMaxPace",
        "averageHR", "maxHR", "AerobicEfficiency"]
    df_swim_table = df_swim_timeframed[columns_to_show_swim]
    df_swim_table.rename(columns={
        "activityName": "Name",
        "startTimeLocal": "Time",
        "YearWeek": "Week",
        "distance": "Distance (km)",
        "typeKey": "Swim type",
        "sportPace": "Pace (min/100m)",
        "sportMaxPace": "Max pace (min/100m)",
        "movingDuration": "Duration (hr)",
        "averageHR": "Avg HR",
        "maxHR": "Max HR",
        "AerobicEfficiency": "AE (HB/100m)",
        "averageSwolf": "SWOLF"}, inplace=True)

    # Calender van swim met zone
    fig_calender = px.scatter(
        df_Swim_Calender,
        x=df_Swim_Calender["startTimeLocal"].dt.day,
        y=df_Swim_Calender["startTimeLocal"].dt.month_name(),
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        title="Swimming Training Calendar (Last 3 Months)",
        custom_data=["Discipline", "startTimeLocal"])
    fig_calender.update_traces(
    hovertemplate="Date: %{customdata[1]|%Y-%m-%d}<extra></extra>")
    fig_calender.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
    st.plotly_chart(fig_calender, use_container_width=True)
    # bar chart
    fig_Time_swim = px.bar(
        Summary_Swim_Week,
        x="WeekIndex",
        y="movingDuration",
        color="classifiedZone",
        title="Total swimming time per week",
        barmode="stack",
        pattern_shape="typeKey",
        color_discrete_map=workout_zone_colors,
        text=Summary_Swim_Week["movingDuration"].apply(format_hours_to_hhmm))
    fig_Time_swim.update_layout(
        xaxis_title="Week",
        yaxis_title="Duration (hours)",
        xaxis=dict(
            tickmode='array',
            tickvals=Summary_Swim_Week["WeekIndex"],  # actual values used
            ticktext=Summary_Swim_Week["YearWeek"]))   # how they should be shown
    fig_Time_swim.update_yaxes(
        tickvals=tickvals_time,
        ticktext=ticktext_time)
    fig_Time_swim.update_traces(
        textposition="inside", hovertemplate="<extra></extra>",
        textfont=dict(color="white"))
    st.plotly_chart(fig_Time_swim, use_container_width=True)
    fig_Distance_swim = px.bar(
        Summary_Swim_Week,
        x="WeekIndex",
        y="distance",
        color="classifiedZone",
        title="Total swimming distance per week",
        barmode="stack",
        pattern_shape="typeKey",
        text=round(Summary_Swim_Week["distance"],ndigits=2),
        color_discrete_map=workout_zone_colors)
    fig_Distance_swim.update_layout(
        xaxis_title="Week",
        yaxis_title="Distance (km)",
        xaxis=dict(
            tickmode='array',
            tickvals=Summary_Swim_Week["WeekIndex"],  # actual values used
            ticktext=Summary_Swim_Week["YearWeek"]))   # how they should be shown
    fig_Distance_swim.update_traces(
        textposition="inside", hovertemplate="<extra></extra>",
        textfont=dict(color="white"))
    fig_Distance_swim.add_shape(type="line",
        y=3.8, line=dict(color="blue", dash="dot"))
    st.plotly_chart(fig_Distance_swim, use_container_width=True) 
    # pie chart met zones
    fig_Swim_time = go.Figure(go.Pie( # pie chart met gesommeerde tijd
        labels=Summary_Swim_Overview["classifiedZone"],
        values=Summary_Swim_Overview["movingDuration"],
        marker=dict(colors=[workout_zone_colors.get(d, "#CCCCCC") for d in Summary_Swim_Overview["classifiedZone"]]),
        hole=0.4))
    fig_Swim_time.update_layout(title_text=f"{timeframe} – Time spent in each zone (h)", uniformtext_minsize=8, uniformtext_mode='hide')
    fig_Swim_time.update_traces(hovertemplate="<extra></extra>", text=Summary_Swim_Overview["movingDuration"].apply(format_hours_to_hhmm))
    st.plotly_chart(fig_Swim_time, use_container_width=True)  
    # line charts
    fig_swim_Pace = px.scatter(
        df_swim_timeframed,
        x="startTimeLocal",
        y="sportPace",
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        symbol="typeKey",
        hover_data=["distance", df_swim_timeframed["movingDuration"].apply(format_hours_to_hhmm), "averageHR", "classifiedZone"])
    fig_swim_Pace.update_layout(
        xaxis_title="Date",
        yaxis_title="Pace (min/100m)",
        legend_title="Zone",
        height=500)
    fig_swim_Pace.update_yaxes(
        tickvals=tickvals_pace,
        ticktext=ticktext_pace,
        range=[0, rounded_max_pace])
    fig_swim_Pace.update_traces(
        hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>" +
            "Pace: %{y:.2f} min/100m<br>" +
            "Distance: %{customdata[0]} km<br>" +
            "Duration: %{customdata[1]}<br>" +
            "HR: %{customdata[2]} bpm<br>" +
            "Zone: %{customdata[3]}<extra></extra>")
    st.plotly_chart(fig_swim_Pace, use_container_width=True)
    fig_swim_maxPace = px.scatter(
        df_swim_timeframed,
        x="startTimeLocal",
        y="sportMaxPace",
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        symbol="typeKey",
        hover_data=["distance", df_swim_timeframed["movingDuration"].apply(format_hours_to_hhmm), "averageHR", "classifiedZone"])
    fig_swim_maxPace.update_layout(
        xaxis_title="Date",
        yaxis_title="Max pace (min/100m)",
        legend_title="Zone",
        height=500)
    fig_swim_maxPace.update_yaxes(
        tickvals=tickvals_maxpace,
        ticktext=ticktext_maxpace,
        range=[0, rounded_max_maxpace])
    fig_swim_maxPace.update_traces(
        hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>" +
            "Pace: %{y:.2f} min/100m<br>" +
            "Distance: %{customdata[0]} km<br>" +
            "Duration: %{customdata[1]}<br>" +
            "HR: %{customdata[2]} bpm<br>" +
            "Zone: %{customdata[3]}<extra></extra>")
    st.plotly_chart(fig_swim_maxPace, use_container_width=True)
    fig_swim_AE = px.scatter(
        df_swim_timeframed,
        x="startTimeLocal",
        y="AerobicEfficiency",
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        symbol="typeKey",
        hover_data=["distance", df_swim_timeframed["movingDuration"].apply(format_hours_to_hhmm), "averageHR", "classifiedZone"])
    fig_swim_AE.update_layout(
        xaxis_title="Date",
        yaxis_title="Aerobic efficiency (Heartbeats/100m)",
        legend_title="Zone",
        height=500)
    fig_swim_AE.update_traces(
        hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>" +
            "Pace: %{y:.2f} min/100m<br>" +
            "Distance: %{customdata[0]} km<br>" +
            "Duration: %{customdata[1]}<br>" +
            "HR: %{customdata[2]} bpm<br>" +
            "Zone: %{customdata[3]}<extra></extra>")
    st.plotly_chart(fig_swim_AE, use_container_width=True)
    fig_swim_swolf = px.scatter(
        df_swim_timeframed,
        x="startTimeLocal",
        y="averageSwolf",
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        symbol="typeKey",
        hover_data=["distance", df_swim_timeframed["movingDuration"].apply(format_hours_to_hhmm), "averageHR", "classifiedZone"])
    fig_swim_swolf.update_layout(
        xaxis_title="Date",
        yaxis_title="Average Swolf",
        legend_title="Zone",
        height=500)
    fig_swim_swolf.update_traces(
        hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>" +
            "Pace: %{y:.2f} min/100m<br>" +
            "Distance: %{customdata[0]} km<br>" +
            "Duration: %{customdata[1]}<br>" +
            "HR: %{customdata[2]} bpm<br>" +
            "Zone: %{customdata[3]}<extra></extra>")
    st.plotly_chart(fig_swim_swolf, use_container_width=True)
    # --- table
    st.subheader("📄 Swimactivity")
    st.dataframe(df_swim_table)
# === Setup BIKE ===
with bike_tab:
    # --- calender zone training
    df_Bike_Calender = df_Calender[df_Calender['Discipline'] == "Bike"]
    df_Bike_Calender["day"] = df_Bike_Calender["startTimeLocal"].dt.day
    df_Bike_Calender["month"] = df_Bike_Calender["startTimeLocal"].dt.month_name()
    # --- pie chart
    if selected_bike == "Both":
        df_bike_selected = df_activities[df_activities["typeKey"].isin(["cycling", "indoor_cycling"])]
    else:
        df_bike_selected = df_activities[df_activities["typeKey"] == selected_bike]


    if timeframe_days[timeframe] is not None: # kiezen hoeveel dagen worden toegevoegd aan de grafiek
        cutoff = datetime.now() - timedelta(days=timeframe_days[timeframe])
        df_bike_timeframed = df_bike_selected[df_bike_selected["startTimeLocal"] >= cutoff]
    else:
        df_bike_timeframed = df_bike_selected.copy()

    Summary_Bike_Overview = summarize_timeDistance_zone(df_bike_timeframed, timeframe_days[timeframe])
    # --- grouped bar chart
    Summary_Bike_Week = (
        df_bike_timeframed
        .groupby(["WeekIndex","YearWeek", "Year", "Week", "typeKey", "classifiedZone"])
        .agg({
            "movingDuration": "sum",
            "distance": "sum"})
        .reset_index())
    Summary_Bike_Week = pd.merge(df_week, Summary_Bike_Week, on=["Week", "Year"], how="left")
    Summary_Bike_Week = Summary_Bike_Week.sort_values("WeekIndex")
    for col in ["movingDuration", "distance"]:
        Summary_Bike_Week[col] = Summary_Bike_Week[col].fillna(0)
    # --- scatter plots
    max_val_time = Summary_Bike_Week.groupby(["Year", "Week"])["movingDuration"].sum().max()
    rounded_max_time = math.ceil(max_val_time / 0.083) * 0.083
    tickvals_time = np.linspace(0, rounded_max_time, 8)
    ticktext_time = [format_hours_to_hhmm(p) for p in tickvals_time]

    # --- table
    columns_to_show_bike = [
        "activityName", "startTimeLocal","YearWeek", "typeKey", "distance",
        "classifiedZone","movingDuration", "sportPace", "sportMaxPace",
        "averageHR", "maxHR", "AerobicEfficiencyBike", "averageBikingCadenceInRevPerMinute", "maxBikingCadenceInRevPerMinute"]
    df_bike_table = df_bike_timeframed[columns_to_show_bike]
    df_bike_table.rename(columns={
        "activityName": "Name",
        "startTimeLocal": "Time",
        "YearWeek": "Week",
        "distance": "Distance (km)",
        "typeKey": "Swim type",
        "sportPace": "Speed (km/h)",
        "sportMaxPace": "Max pace (km/h)",
        "movingDuration": "Duration (h)",
        "averageHR": "Avg HR",
        "maxHR": "Max HR",
        "AerobicEfficiencyBike": "AE (HB/km)",
        "averageBikingCadenceInRevPerMinute": "Avg biking cadance (Rev/min)", 
        "maxBikingCadenceInRevPerMinute": "Max biking cadance (Rev/min)"
        }, inplace=True)

    # Calender van swim met zone
    fig_calender_Bike = px.scatter(
        df_Bike_Calender,
        x=df_Bike_Calender["startTimeLocal"].dt.day,
        y=df_Bike_Calender["startTimeLocal"].dt.month_name(),
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        title="Biking Training Calendar (Last 3 Months)",
        custom_data=["Discipline", "startTimeLocal"])
    fig_calender_Bike.update_traces(
    hovertemplate="Date: %{customdata[1]|%Y-%m-%d}<extra></extra>")
    fig_calender_Bike.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
    st.plotly_chart(fig_calender_Bike, use_container_width=True)
    # bar chart
    fig_Time_bike = px.bar(
        Summary_Bike_Week,
        x="WeekIndex",
        y="movingDuration",
        color="classifiedZone",
        title="Total biking time per week",
        barmode="stack",
        pattern_shape="typeKey",
        color_discrete_map=workout_zone_colors,
        text=Summary_Bike_Week["movingDuration"].apply(format_hours_to_hhmm))
    fig_Time_bike.update_layout(
        xaxis_title="Week",
        yaxis_title="Duration (hours)",
        xaxis=dict(
            tickmode='array',
            tickvals=Summary_Bike_Week["WeekIndex"],  # actual values used
            ticktext=Summary_Bike_Week["YearWeek"]))   # how they should be shown
    fig_Time_bike.update_yaxes(
        tickvals=tickvals_time,
        ticktext=ticktext_time)
    fig_Time_bike.update_traces(
        textposition="inside", hovertemplate="<extra></extra>",
        textfont=dict(color="white"))
    st.plotly_chart(fig_Time_bike, use_container_width=True)
    fig_Distance_bike = px.bar(
        Summary_Bike_Week,
        x="WeekIndex",
        y="distance",
        color="classifiedZone",
        title="Total biking distance per week",
        barmode="stack",
        pattern_shape="typeKey",
        text=round(Summary_Bike_Week["distance"],ndigits=2),
        color_discrete_map=workout_zone_colors)
    fig_Distance_bike.update_layout(
        xaxis_title="Week",
        yaxis_title="Distance (km)",
        xaxis=dict(
            tickmode='array',
            tickvals=Summary_Bike_Week["WeekIndex"],  # actual values used
            ticktext=Summary_Bike_Week["YearWeek"]))   # how they should be shown
    fig_Distance_bike.update_traces(
        textposition="inside", hovertemplate="<extra></extra>",
        textfont=dict(color="white"))
    fig_Distance_bike.add_shape(type="line",
        y=180, line=dict(color="red", dash="dot"))
    st.plotly_chart(fig_Distance_bike, use_container_width=True) 
    # pie chart met zones
    fig_Bike_time = go.Figure(go.Pie( # pie chart met gesommeerde tijd
        labels=Summary_Bike_Overview["classifiedZone"],
        values=Summary_Bike_Overview["movingDuration"],
        marker=dict(colors=[workout_zone_colors.get(d, "#CCCCCC") for d in Summary_Bike_Overview["classifiedZone"]]),
        hole=0.4))
    fig_Bike_time.update_layout(title_text=f"{timeframe} – Time spent in each zone (h)", uniformtext_minsize=8, uniformtext_mode='hide')
    fig_Bike_time.update_traces(hovertemplate="<extra></extra>", text=Summary_Bike_Overview["movingDuration"].apply(format_hours_to_hhmm))
    st.plotly_chart(fig_Bike_time, use_container_width=True)  
    # line charts
    fig_bike_Pace = px.scatter(
        df_bike_timeframed,
        x="startTimeLocal",
        y="sportPace",
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        symbol="typeKey",
        hover_data=["distance", df_bike_timeframed["movingDuration"].apply(format_hours_to_hhmm), "averageHR", "classifiedZone"])
    fig_bike_Pace.update_layout(
        xaxis_title="Date",
        yaxis_title="Speed (km/h)",
        legend_title="Zone",
        height=500)
    fig_bike_Pace.update_traces(
        hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>" +
            "Speed: %{y:.2f} km/h<br>" +
            "Distance: %{customdata[0]} km<br>" +
            "Duration: %{customdata[1]}<br>" +
            "HR: %{customdata[2]} bpm<br>" +
            "Zone: %{customdata[3]}<extra></extra>")
    st.plotly_chart(fig_bike_Pace, use_container_width=True)
    fig_bike_maxPace = px.scatter(
        df_bike_timeframed,
        x="startTimeLocal",
        y="sportMaxPace",
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        symbol="typeKey",
        hover_data=["distance", df_bike_timeframed["movingDuration"].apply(format_hours_to_hhmm), "averageHR", "classifiedZone"])
    fig_bike_maxPace.update_layout(
        xaxis_title="Date",
        yaxis_title="Max pace (km/h)",
        legend_title="Zone",
        height=500)
    fig_bike_maxPace.update_traces(
        hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>" +
            "Speed: %{y:.2f} km/h<br>" +
            "Distance: %{customdata[0]} km<br>" +
            "Duration: %{customdata[1]}<br>" +
            "HR: %{customdata[2]} bpm<br>" +
            "Zone: %{customdata[3]}<extra></extra>")
    st.plotly_chart(fig_bike_maxPace, use_container_width=True)
    fig_bike_AE = px.scatter(
        df_bike_timeframed,
        x="startTimeLocal",
        y="AerobicEfficiencyBike",
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        symbol="typeKey",
        hover_data=["distance", df_bike_timeframed["movingDuration"].apply(format_hours_to_hhmm), "averageHR", "classifiedZone"])
    fig_bike_AE.update_layout(
        xaxis_title="Date",
        yaxis_title="Aerobic efficiency (Heartbeats/km)",
        legend_title="Zone",
        height=500)
    fig_bike_AE.update_traces(
        hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>" +
            "AE: %{y:.2f} hb/km<br>" +
            "Distance: %{customdata[0]} km<br>" +
            "Duration: %{customdata[1]}<br>" +
            "HR: %{customdata[2]} bpm<br>" +
            "Zone: %{customdata[3]}<extra></extra>")
    st.plotly_chart(fig_bike_AE, use_container_width=True)
    fig_bike_candance = px.scatter(
        df_bike_timeframed,
        x="startTimeLocal",
        y="averageBikingCadenceInRevPerMinute",
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        symbol="typeKey",
        hover_data=["distance", df_bike_timeframed["movingDuration"].apply(format_hours_to_hhmm), "averageHR", "classifiedZone"])
    fig_bike_candance.update_layout(
        xaxis_title="Date",
        yaxis_title="Average cadance (Rev/min)",
        legend_title="Zone",
        height=500)
    fig_bike_candance.update_traces(
        hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>" +
            "Cadance: %{y:.2f} Rev/min<br>" +
            "Distance: %{customdata[0]} km<br>" +
            "Duration: %{customdata[1]}<br>" +
            "HR: %{customdata[2]} bpm<br>" +
            "Zone: %{customdata[3]}<extra></extra>")
    st.plotly_chart(fig_bike_candance, use_container_width=True)
    # --- table
    st.subheader("📄 Bikeactivity")
    st.dataframe(df_bike_table)
# === Setup RUN ===
with run_tab:
    # --- calender zone training
    df_Run_Calender = df_Calender[df_Calender['Discipline'] == "Run"]
    df_Run_Calender["day"] = df_Run_Calender["startTimeLocal"].dt.day
    df_Run_Calender["month"] = df_Run_Calender["startTimeLocal"].dt.month_name()
    # --- pie chart
    run_mapping = {
        "Both": ["running", "trail_running", "treadmill_running"],
        "outdoor_running": ["running", "trail_running"],
        "indoor_running": ["treadmill_running"]}
    df_run_selected = df_activities[df_activities["typeKey"].isin(run_mapping[selected_run])]

    if timeframe_days[timeframe] is not None: # kiezen hoeveel dagen worden toegevoegd aan de grafiek
        cutoff = datetime.now() - timedelta(days=timeframe_days[timeframe])
        df_run_timeframed = df_run_selected[df_run_selected["startTimeLocal"] >= cutoff]
    else:
        df_run_timeframed = df_run_selected.copy()

    Summary_Run_Overview = summarize_timeDistance_zone(df_run_timeframed, timeframe_days[timeframe])
    # --- grouped bar chart
    Summary_Run_Week = (
        df_run_timeframed
        .groupby(["WeekIndex","YearWeek", "Year", "Week", "typeKey", "classifiedZone"])
        .agg({
            "movingDuration": "sum",
            "distance": "sum"})
        .reset_index())
    Summary_Run_Week = pd.merge(df_week, Summary_Run_Week, on=["Week", "Year"], how="left")
    Summary_Run_Week = Summary_Run_Week.sort_values("WeekIndex")
    for col in ["movingDuration", "distance"]:
        Summary_Run_Week[col] = Summary_Run_Week[col].fillna(0)
    # --- scatter plots
    max_val_time = Summary_Run_Week.groupby(["Year", "Week"])["movingDuration"].sum().max()
    rounded_max_time = math.ceil(max_val_time / 0.083) * 0.083
    tickvals_time = np.linspace(0, rounded_max_time, 8)
    ticktext_time = [format_hours_to_hhmm(p) for p in tickvals_time]

    max_val_pace = df_run_timeframed['sportPace'].max()
    rounded_max_pace = math.ceil(max_val_pace / 0.083) * 0.083
    tickvals_pace = np.linspace(0, rounded_max_pace, 8)
    ticktext_pace = [format_pace_to_minsec(p) for p in tickvals_pace]

    max_val_maxpace = df_run_timeframed['sportMaxPace'].max()
    rounded_max_maxpace = math.ceil(max_val_maxpace / 0.083) * 0.083
    tickvals_maxpace = np.linspace(0, rounded_max_maxpace, 8)
    ticktext_maxpace = [format_pace_to_minsec(p) for p in tickvals_maxpace]
    # --- table
    columns_to_show_Run = [
        "activityName", "startTimeLocal","YearWeek", "typeKey", "distance",
        "classifiedZone","movingDuration", "sportPace", "sportMaxPace",
        "averageHR", "maxHR", "AerobicEfficiency", "averageRunningCadenceInStepsPerMinute", "maxRunningCadenceInStepsPerMinute"]
    df_run_table = df_run_timeframed[columns_to_show_Run]
    df_run_table.rename(columns={
        "activityName": "Name",
        "startTimeLocal": "Time",
        "YearWeek": "Week",
        "distance": "Distance (km)",
        "typeKey": "Run type",
        "sportPace": "Pace (min/km)",
        "sportMaxPace": "Max pace (min/km)",
        "movingDuration": "Duration (h)",
        "averageHR": "Avg HR",
        "maxHR": "Max HR",
        "AerobicEfficiency": "AE (HB/km)",
        "averageRunningCadenceInStepsPerMinute": "Avg running cadance (Steps/min)", 
        "maxRunningCadenceInStepsPerMinute": "Max running cadance (Steps/min)"
        }, inplace=True)

    # Calender van swim met zone
    fig_calender_Run = px.scatter(
        df_Run_Calender,
        x=df_Run_Calender["startTimeLocal"].dt.day,
        y=df_Run_Calender["startTimeLocal"].dt.month_name(),
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        title="Running Training Calendar (Last 3 Months)",
        custom_data=["Discipline", "startTimeLocal"])
    fig_calender_Run.update_traces(
    hovertemplate="Date: %{customdata[1]|%Y-%m-%d}<extra></extra>")
    fig_calender_Run.update_layout(xaxis_title="Day of the Month", yaxis_title="Month")
    st.plotly_chart(fig_calender_Run, use_container_width=True)
    # bar chart
    fig_Time_run = px.bar(
        Summary_Run_Week,
        x="WeekIndex",
        y="movingDuration",
        color="classifiedZone",
        title="Total running time per week",
        barmode="stack",
        pattern_shape="typeKey",
        color_discrete_map=workout_zone_colors,
        text=Summary_Run_Week["movingDuration"].apply(format_hours_to_hhmm))
    fig_Time_run.update_layout(
        xaxis_title="Week",
        yaxis_title="Duration (hours)",
        xaxis=dict(
            tickmode='array',
            tickvals=Summary_Run_Week["WeekIndex"],  # actual values used
            ticktext=Summary_Run_Week["YearWeek"]))   # how they should be shown
    fig_Time_run.update_yaxes(
        tickvals=tickvals_time,
        ticktext=ticktext_time)
    fig_Time_run.update_traces(
        textposition="inside", hovertemplate="<extra></extra>",
        textfont=dict(color="white"))
    st.plotly_chart(fig_Time_run, use_container_width=True)
    fig_Distance_run = px.bar(
        Summary_Run_Week,
        x="WeekIndex",
        y="distance",
        color="classifiedZone",
        title="Total running distance per week",
        barmode="stack",
        pattern_shape="typeKey",
        text=round(Summary_Run_Week["distance"],ndigits=2),
        color_discrete_map=workout_zone_colors)
    fig_Distance_run.update_layout(
        xaxis_title="Week",
        yaxis_title="Distance (km)",
        xaxis=dict(
            tickmode='array',
            tickvals=Summary_Run_Week["WeekIndex"],  # actual values used
            ticktext=Summary_Run_Week["YearWeek"]))   # how they should be shown
    fig_Distance_run.update_traces(
        textposition="inside", hovertemplate="<extra></extra>",
        textfont=dict(color="white"))
    fig_Distance_run.add_shape(type="line",
        y=42.2, line=dict(color="green", dash="dot"))
    st.plotly_chart(fig_Distance_run, use_container_width=True) 
    # pie chart met zones
    fig_Run_time = go.Figure(go.Pie( # pie chart met gesommeerde tijd
        labels=Summary_Run_Overview["classifiedZone"],
        values=Summary_Run_Overview["movingDuration"],
        marker=dict(colors=[workout_zone_colors.get(d, "#CCCCCC") for d in Summary_Run_Overview["classifiedZone"]]),
        hole=0.4))
    fig_Run_time.update_layout(title_text=f"{timeframe} – Time spent in each zone (h)", uniformtext_minsize=8, uniformtext_mode='hide')
    fig_Run_time.update_traces(hovertemplate="<extra></extra>", text=Summary_Run_Overview["movingDuration"].apply(format_hours_to_hhmm))
    st.plotly_chart(fig_Run_time, use_container_width=True)  
    # line charts
    fig_Run_Pace = px.scatter(
        df_run_timeframed,
        x="startTimeLocal",
        y="sportPace",
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        symbol="typeKey",
        hover_data=["distance", df_run_timeframed["movingDuration"].apply(format_hours_to_hhmm), "averageHR", "classifiedZone"])
    fig_Run_Pace.update_layout(
        xaxis_title="Date",
        yaxis_title="Speed (km/h)",
        legend_title="Zone",
        height=500)
    fig_Run_Pace.update_yaxes(
        tickvals=tickvals_pace,
        ticktext=ticktext_pace,
        range=[0, rounded_max_pace])
    fig_Run_Pace.update_traces(
        hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>" +
            "Speed: %{y:.2f} km/h<br>" +
            "Distance: %{customdata[0]} km<br>" +
            "Duration: %{customdata[1]}<br>" +
            "HR: %{customdata[2]} bpm<br>" +
            "Zone: %{customdata[3]}<extra></extra>")
    st.plotly_chart(fig_Run_Pace, use_container_width=True)
    fig_run_maxPace = px.scatter(
        df_run_timeframed,
        x="startTimeLocal",
        y="sportMaxPace",
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        symbol="typeKey",
        hover_data=["distance", df_run_timeframed["movingDuration"].apply(format_hours_to_hhmm), "averageHR", "classifiedZone"])
    fig_run_maxPace.update_layout(
        xaxis_title="Date",
        yaxis_title="Max pace (km/h)",
        legend_title="Zone",
        height=500)
    fig_run_maxPace.update_yaxes(
        tickvals=tickvals_maxpace,
        ticktext=ticktext_maxpace,
        range=[0, rounded_max_maxpace])
    fig_run_maxPace.update_traces(
        hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>" +
            "Speed: %{y:.2f} km/h<br>" +
            "Distance: %{customdata[0]} km<br>" +
            "Duration: %{customdata[1]}<br>" +
            "HR: %{customdata[2]} bpm<br>" +
            "Zone: %{customdata[3]}<extra></extra>")
    st.plotly_chart(fig_run_maxPace, use_container_width=True)
    fig_run_AE = px.scatter(
        df_run_timeframed,
        x="startTimeLocal",
        y="AerobicEfficiency",
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        symbol="typeKey",
        hover_data=["distance", df_run_timeframed["movingDuration"].apply(format_hours_to_hhmm), "averageHR", "classifiedZone"])
    fig_run_AE.update_layout(
        xaxis_title="Date",
        yaxis_title="Aerobic efficiency (Heartbeats/km)",
        legend_title="Zone",
        height=500)
    fig_run_AE.update_traces(
        hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>" +
            "AE: %{y:.2f} hb/km<br>" +
            "Distance: %{customdata[0]} km<br>" +
            "Duration: %{customdata[1]}<br>" +
            "HR: %{customdata[2]} bpm<br>" +
            "Zone: %{customdata[3]}<extra></extra>")
    st.plotly_chart(fig_run_AE, use_container_width=True)
    fig_run_candance = px.scatter(
        df_run_timeframed,
        x="startTimeLocal",
        y="averageRunningCadenceInStepsPerMinute",
        color="classifiedZone",
        color_discrete_map=workout_zone_colors,
        symbol="typeKey",
        hover_data=["distance", df_run_timeframed["movingDuration"].apply(format_hours_to_hhmm), "averageHR", "classifiedZone"])
    fig_run_candance.update_layout(
        xaxis_title="Date",
        yaxis_title="Average cadance (Step/min)",
        legend_title="Zone",
        height=500)
    fig_run_candance.update_traces(
        hovertemplate=
            "Date: %{x|%Y-%m-%d}<br>" +
            "Cadance: %{y:.2f} Step/min<br>" +
            "Distance: %{customdata[0]} km<br>" +
            "Duration: %{customdata[1]}<br>" +
            "HR: %{customdata[2]} bpm<br>" +
            "Zone: %{customdata[3]}<extra></extra>")
    st.plotly_chart(fig_run_candance, use_container_width=True)
    # --- table
    st.subheader("📄 Runactivity")
    st.dataframe(df_run_table)


