import pandas as pd
import numpy as np


FEATURES = [
    "sell_price",
    "price_change",
    "day_of_week",
    "day",
    "week",
    "month",
    "year",
    "is_weekend",
    "has_event",
    "sales_lag_1",
    "sales_lag_7",
    "sales_lag_14",
    "sales_lag_28",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_mean_28",
    "rolling_std_7",
]


def _load_future_calendar(days_ahead, last_d_number, calendar_path="data/calendar.csv"):
    """
    Pulls the real calendar rows immediately after the last training day.
    M5's calendar.csv extends beyond sales_train_validation.csv, so these
    are genuine future weekday/month/event values, not guesses.
    """
    calendar = pd.read_csv(calendar_path)

    future_d_list = [f"d_{last_d_number + i}" for i in range(1, days_ahead + 1)]
    future_cal = calendar[calendar["d"].isin(future_d_list)].copy()

    if future_cal.empty:
        raise ValueError(
            "No future calendar rows found. This means the M5 calendar.csv "
            "does not extend far enough past the last training day, or the "
            "file path is wrong."
        )

    future_cal["date"] = pd.to_datetime(future_cal["date"])
    future_cal = future_cal.sort_values("date").reset_index(drop=True)

    future_cal["day_of_week"] = future_cal["date"].dt.dayofweek
    future_cal["day"] = future_cal["date"].dt.day
    future_cal["week"] = future_cal["date"].dt.isocalendar().week.astype(int)
    future_cal["month"] = future_cal["date"].dt.month
    future_cal["year"] = future_cal["date"].dt.year
    future_cal["is_weekend"] = (future_cal["day_of_week"] >= 5).astype(int)
    future_cal["has_event"] = (
        future_cal["event_name_1"].notna() | future_cal["event_name_2"].notna()
    ).astype(int)

    return future_cal


def generate_future_forecast(item_id, features_df, model_bundle, days_ahead=7,
                              calendar_path="data/calendar.csv"):
    """
    Recursively forecasts the next `days_ahead` days of demand for one item_id,
    using the same feature definitions as train_model.py.

    features_df: the loaded data/processed/m5_features.csv (with 'date' as datetime)
    model_bundle: the dict loaded from models/xgboost_demand_model.pkl
                  -> {"model": ..., "features": [...]}
    """
    model = model_bundle["model"]
    trained_features = model_bundle["features"]

    item_df = (
        features_df[features_df["item_id"] == item_id]
        .sort_values("date")
        .reset_index(drop=True)
    )

    if item_df.empty:
        raise ValueError(f"No rows found for item_id={item_id} in features_df.")

    last_row = item_df.iloc[-1]
    last_d_number = int(last_row["d"].split("_")[1])
    last_sell_price = last_row["sell_price"]

    # Running buffer of sales history: real observed sales, then predictions
    # get appended one at a time as we move forward.
    sales_history = item_df["sales"].tolist()

    future_cal = _load_future_calendar(days_ahead, last_d_number, calendar_path)

    future_rows = []

    for i in range(min(days_ahead, len(future_cal))):
        cal_row = future_cal.iloc[i]

        sales_lag_1 = sales_history[-1]
        sales_lag_7 = sales_history[-7] if len(sales_history) >= 7 else np.nan
        sales_lag_14 = sales_history[-14] if len(sales_history) >= 14 else np.nan
        sales_lag_28 = sales_history[-28] if len(sales_history) >= 28 else np.nan

        rolling_mean_7 = np.mean(sales_history[-7:]) if len(sales_history) >= 7 else np.nan
        rolling_mean_14 = np.mean(sales_history[-14:]) if len(sales_history) >= 14 else np.nan
        rolling_mean_28 = np.mean(sales_history[-28:]) if len(sales_history) >= 28 else np.nan
        rolling_std_7 = (
            np.std(sales_history[-7:], ddof=1) if len(sales_history) >= 7 else np.nan
        )

        row = {
            "sell_price": last_sell_price,      # no future price in M5 -> carried forward
            "price_change": 0.0,                # no future price movement to compute
            "day_of_week": cal_row["day_of_week"],
            "day": cal_row["day"],
            "week": cal_row["week"],
            "month": cal_row["month"],
            "year": cal_row["year"],
            "is_weekend": cal_row["is_weekend"],
            "has_event": cal_row["has_event"],
            "sales_lag_1": sales_lag_1,
            "sales_lag_7": sales_lag_7,
            "sales_lag_14": sales_lag_14,
            "sales_lag_28": sales_lag_28,
            "rolling_mean_7": rolling_mean_7,
            "rolling_mean_14": rolling_mean_14,
            "rolling_mean_28": rolling_mean_28,
            "rolling_std_7": rolling_std_7,
        }

        X = pd.DataFrame([row])[trained_features]
        pred = float(model.predict(X)[0])
        pred = max(0.0, pred)  # demand cannot be negative

        sales_history.append(pred)

        future_rows.append(
            {
                "date": cal_row["date"],
                "predicted_sales": pred,
                "is_weekend": bool(cal_row["is_weekend"]),
                "has_event": bool(cal_row["has_event"]),
            }
        )

    return pd.DataFrame(future_rows)


if __name__ == "__main__":
    import joblib

    features_df = pd.read_csv("data/processed/m5_features.csv")
    features_df["date"] = pd.to_datetime(features_df["date"])

    model_bundle = joblib.load("models/xgboost_demand_model.pkl")

    sample_item = features_df["item_id"].iloc[0]
    forecast = generate_future_forecast(sample_item, features_df, model_bundle, days_ahead=7)

    print(f"7-day future forecast for {sample_item}:")
    print(forecast.to_string(index=False))