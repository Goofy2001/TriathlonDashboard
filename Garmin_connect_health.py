import os
import json
import pandas as pd
from datetime import datetime, timedelta
from garminconnect import Garmin

# === Config ===
email = ""
password = ""
CSV_FILE = "garmin_metrics_log.csv"

# === Garmin login ===
client = Garmin(email, password)
client.login()

# === Datum ===
today = (datetime.now() - timedelta(days=0)).strftime("%Y-%m-%d")  # gebruik ISO-formaat
sleep_file_path = f"sleep_raw_{today}.json"

# === Stop als datum al in CSV staat ===
if os.path.exists(CSV_FILE):
    df_existing = pd.read_csv(CSV_FILE)
    df_existing["date"] = pd.to_datetime(df_existing["date"], errors="coerce").dt.strftime("%Y-%m-%d")

    if today in df_existing["date"].values:
        print(f"⏭️ Gegevens voor {today} bestaan al in {CSV_FILE}, overslaan.")
        exit()

# === Sleepdata ophalen ===
sleep_score = None
resting_hr = None
avg_hrv = None

try:
    sleep_data = client.get_sleep_data(today)

    with open(sleep_file_path, "w", encoding="utf-8") as f:
        json.dump(sleep_data, f, indent=2, ensure_ascii=False)
    print(f"📁 Sleep JSON opgeslagen: {sleep_file_path}")

    sleep_score = sleep_data.get("dailySleepDTO", {}).get("sleepScores", {}).get("overall", {}).get("value")
    resting_hr = sleep_data.get("restingHeartRate")
    avg_hrv = sleep_data.get("avgOvernightHrv")

    print("✅ Sleep score:", sleep_score)
    print("✅ Resting HR:", resting_hr)
    print("✅ Avg HRV:", avg_hrv)

except Exception as e:
    print("❌ Fout bij ophalen van slaapgegevens:", e)

# === Log opslaan in CSV ===
row = {
    "date": today,
    "sleep_score": sleep_score,
    "resting_hr": resting_hr,
    "avg_hrv": avg_hrv,
}
df = pd.DataFrame([row])
df.to_csv(CSV_FILE, mode="a", index=False, header=not os.path.exists(CSV_FILE))
print("✅ Data opgeslagen:", row)

# === Verwijder JSON-bestand na logging ===
if os.path.exists(sleep_file_path):
    os.remove(sleep_file_path)
    print(f"🗑️ Verwijderd: {sleep_file_path}")
