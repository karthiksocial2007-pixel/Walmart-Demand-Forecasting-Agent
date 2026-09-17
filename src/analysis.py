from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "m5_features.csv"
OUTPUT_DIR = ROOT / "outputs"

def trend_label(item_df):
    recent = float(item_df["sales"].tail(7).mean())
    previous = float(item_df["sales"].tail(14).head(7).mean())
    if previous == 0:
        return "Increasing" if recent > 0 else "Stable"
    change = (recent - previous) / previous * 100
    if change > 5:
        return "Increasing"
    if change < -5:
        return "Decreasing"
    return "Stable"

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Processed data not found: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df.sort_values(["item_id", "date"]).reset_index(drop=True)

    trend_rows = []
    anomaly_rows = []

    for item_id, g in df.groupby("item_id", sort=True):
        recent = float(g["sales"].tail(7).mean())
        previous = float(g["sales"].tail(14).head(7).mean())
        growth = 0.0 if previous == 0 else (recent - previous) / previous * 100

        trend_rows.append({
            "item_id": item_id,
            "recent_7_day_avg": recent,
            "previous_7_day_avg": previous,
            "growth_percent": growth,
            "trend": trend_label(g),
        })

        q1 = float(g["sales"].quantile(0.25))
        q3 = float(g["sales"].quantile(0.75))
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        temp = g[["date", "item_id", "sales"]].copy()
        temp["q1"] = q1
        temp["q3"] = q3
        temp["lower_threshold"] = lower
        temp["upper_threshold"] = upper
        temp["is_anomaly"] = (temp["sales"] < lower) | (temp["sales"] > upper)
        anomaly_rows.append(temp)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(trend_rows).to_csv(OUTPUT_DIR / "trend_analysis.csv", index=False)
    pd.concat(anomaly_rows, ignore_index=True).to_csv(
        OUTPUT_DIR / "anomaly_analysis.csv", index=False
    )

    print("Analysis complete.")
    print("Trend table:", OUTPUT_DIR / "trend_analysis.csv")
    print("Anomaly table:", OUTPUT_DIR / "anomaly_analysis.csv")

if __name__ == "__main__":
    main()
