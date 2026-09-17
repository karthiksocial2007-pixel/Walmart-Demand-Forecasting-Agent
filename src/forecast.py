from pathlib import Path
import numpy as np
import pandas as pd

FEATURES = [
    "sell_price", "price_change", "day_of_week", "day", "week", "month",
    "year", "is_weekend", "has_event", "sales_lag_1", "sales_lag_7",
    "sales_lag_14", "sales_lag_28", "rolling_mean_7", "rolling_mean_14",
    "rolling_mean_28", "rolling_std_7",
]

def _calendar_features(calendar_row, date):
    """Return model calendar features for one future date."""
    return {
        "day_of_week": int(date.dayofweek),
        "day": int(date.day),
        "week": int(date.isocalendar().week),
        "month": int(date.month),
        "year": int(date.year),
        "is_weekend": int(date.dayofweek >= 5),
        "has_event": int(
            pd.notna(calendar_row.get("event_name_1"))
            or pd.notna(calendar_row.get("event_name_2"))
        ),
    }

def forecast_product(model, history_df, calendar_df, prices_df=None, item_id=None, horizon=7):
    """
    Recursive future forecast. The model predicts day t, then that prediction
    becomes the next day's lagged history. This is a true forecast beyond the
    final observed M5 sales date.
    """
    if horizon < 1:
        return pd.DataFrame(columns=["date", "item_id", "predicted_sales"])

    hist = history_df.copy()
    hist["date"] = pd.to_datetime(hist["date"])
    hist = hist.sort_values("date").reset_index(drop=True)

    if hist.empty:
        raise ValueError("No historical rows available for this product.")

    required = {"sales", "date", "item_id"}
    missing = required - set(hist.columns)
    if missing:
        raise ValueError(f"History is missing columns: {sorted(missing)}")

    cal = calendar_df.copy()
    cal["date"] = pd.to_datetime(cal["date"])
    cal = cal.sort_values("date").drop_duplicates("date", keep="first")

    price_lookup = None
    if prices_df is not None and not prices_df.empty and item_id is not None:
        p = prices_df.copy()
        if {"item_id", "wm_yr_wk", "sell_price"}.issubset(p.columns):
            p = p[p["item_id"] == item_id].copy()
            price_lookup = p.groupby("wm_yr_wk")["sell_price"].last().to_dict()

    last_price = float(hist["sell_price"].dropna().iloc[-1]) if "sell_price" in hist.columns and hist["sell_price"].notna().any() else 0.0
    last_date = hist["date"].max()
    history_values = hist["sales"].astype(float).tolist()
    prev_price = last_price

    rows = []
    for step in range(1, horizon + 1):
        future_date = last_date + pd.Timedelta(days=step)
        cal_match = cal[cal["date"] == future_date]

        if cal_match.empty:
            # Calendar normally extends beyond the M5 validation sales horizon.
            # If it does not, continue with date-derived features and no event.
            cal_row = {}
            wm_yr_wk = None
        else:
            cal_row = cal_match.iloc[0].to_dict()
            wm_yr_wk = cal_row.get("wm_yr_wk")

        future_price = None
        if price_lookup is not None and pd.notna(wm_yr_wk):
            future_price = price_lookup.get(wm_yr_wk)
        if future_price is None or pd.isna(future_price):
            future_price = last_price

        future_price = float(future_price)
        if prev_price == 0:
            price_change = 0.0
        else:
            price_change = (future_price - prev_price) / prev_price
        price_change = float(np.nan_to_num(price_change, nan=0.0, posinf=0.0, neginf=0.0))

        if len(history_values) < 28:
            raise ValueError("At least 28 historical sales values are required for recursive forecasting.")

        row = _calendar_features(cal_row, future_date)
        row.update({
            "sell_price": future_price,
            "price_change": price_change,
            "sales_lag_1": history_values[-1],
            "sales_lag_7": history_values[-7],
            "sales_lag_14": history_values[-14],
            "sales_lag_28": history_values[-28],
            "rolling_mean_7": float(np.mean(history_values[-7:])),
            "rolling_mean_14": float(np.mean(history_values[-14:])),
            "rolling_mean_28": float(np.mean(history_values[-28:])),
            "rolling_std_7": float(np.std(history_values[-7:], ddof=1)),
        })

        X = pd.DataFrame([row])[FEATURES]
        prediction = float(max(0.0, model.predict(X)[0]))
        history_values.append(prediction)

        rows.append({
            "date": future_date,
            "item_id": item_id if item_id is not None else hist["item_id"].iloc[0],
            "predicted_sales": prediction,
            "sell_price": future_price,
        })

        prev_price = future_price

    return pd.DataFrame(rows)

def forecast_all_products(model, history_df, calendar_df, prices_df=None, horizon=7):
    outputs = []
    for item_id, group in history_df.groupby("item_id", sort=True):
        outputs.append(
            forecast_product(model, group, calendar_df, prices_df, item_id, horizon)
        )
    return pd.concat(outputs, ignore_index=True) if outputs else pd.DataFrame()
