import os
import pandas as pd
from datetime import datetime, date, timedelta
from garminconnect import Garmin

# --- credentials ---
email = os.environ["GARMIN_EMAIL"]
password = os.environ["GARMIN_PASSWORD"]

# --- Login ---
try:
    api = Garmin(email, password)
    api.login()
    print("✅ Garmin login successful.")
except Exception as e:
    print(f"❌ Login failed: {e}")
    exit()

# --- Load existing file ---
file_path = "garmin_full_activities.csv"

try:
    df_existing = pd.read_csv(file_path)
    df_existing["startTimeLocal"] = pd.to_datetime(df_existing["startTimeLocal"], errors="coerce")
    latest_old_date = df_existing["startTimeLocal"].max()
    print(f"📅 Latest saved activity: {latest_old_date}")
except FileNotFoundError:
    df_existing = pd.DataFrame()
    latest_old_date = datetime(2000, 1, 1)  # Fallback: very old date
    print("📂 No existing file found. Starting from scratch.")

# --- Fetch new activities from Garmin ---
start_date = latest_old_date.date()
end_date = date.today()


# --- Fetch activities from Garmin ---
try:
    activities = api.get_activities_by_date(
        start_date.strftime("%Y-%m-%d"),
        end_date.strftime("%Y-%m-%d")
    )
    print(f"✅ Retrieved {len(activities)} activities.")
except Exception as e:
    print(f"❌ Error fetching activities: {e}")
    activities = []

# --- Save to CSV ---
df_activities = pd.DataFrame(activities)
df_activities.to_csv("garmin_activities.csv", index=False)
print("📄 Activities saved to 'garmin_activities.csv'")

# --- Convert to DataFrame and filter by activityId ---
new_ids = set(df_activities["activityId"]) - set(df_existing["activityId"])

df_new_only = df_activities[df_activities["activityId"].isin(new_ids)]

# --- Combine and save ---
df_combined = pd.concat([df_existing, df_new_only], ignore_index=True)
if "startTimeLocal" in df_combined.columns:
    df_combined["startTimeLocal"] = pd.to_datetime(df_combined["startTimeLocal"], errors="coerce")
    df_combined = df_combined.sort_values("activityId", ascending=False).reset_index(drop=True)
df_combined.to_csv("garmin_full_activities.csv", index=False)
print(f"💾 Saved {len(df_combined)} total activities to 'garmin_full_activities.csv'")

