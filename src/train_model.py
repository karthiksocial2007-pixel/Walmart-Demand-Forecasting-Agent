from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = ROOT / "data" / "processed" / "m5_features.csv"
MODEL_FILE = ROOT / "models" / "xgboost_demand_model.pkl"
RESULT_FILE = ROOT / "outputs" / "forecast_results.csv"
METRICS_FILE = ROOT / "outputs" / "metrics.json"

FEATURES = [
    "sell_price", "price_change", "day_of_week", "day", "week", "month",
    "year", "is_weekend", "has_event", "sales_lag_1", "sales_lag_7",
    "sales_lag_14", "sales_lag_28", "rolling_mean_7", "rolling_mean_14",
    "rolling_mean_28", "rolling_std_7",
]

def main():
    if not DATA_FILE.exists():
        raise FileNotFoundError(f"Processed data not found: {DATA_FILE}. Run prepare_data.py first.")

    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    df = df.sort_values(["date", "item_id"]).reset_index(drop=True)

    test_days = 28
    max_date = df["date"].max()
    cutoff = max_date - pd.Timedelta(days=test_days - 1)

    train_df = df[df["date"] < cutoff].copy()
    test_df = df[df["date"] >= cutoff].copy()

    X_train, y_train = train_df[FEATURES], train_df["sales"]
    X_test, y_test = test_df[FEATURES], test_df["sales"]

    if train_df.empty or test_df.empty:
        raise ValueError("Time-based split produced an empty train or test set.")

    print(f"Training rows: {len(train_df):,}")
    print(f"Testing rows: {len(test_df):,}")
    print("Training XGBoost...")

    model = XGBRegressor(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    pred = np.maximum(model.predict(X_test), 0)
    actual = y_test.to_numpy(dtype=float)

    mae = float(mean_absolute_error(actual, pred))
    rmse = float(np.sqrt(mean_squared_error(actual, pred)))
    denominator = max(float(np.abs(actual).sum()), 1.0)
    wape = float(np.abs(actual - pred).sum() / denominator * 100)

    results = test_df[
        ["date", "item_id", "store_id", "dept_id", "cat_id", "sales"]
    ].copy()
    results["predicted_sales"] = pred
    results["absolute_error"] = np.abs(results["sales"] - results["predicted_sales"])

    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    RESULT_FILE.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump({"model": model, "features": FEATURES}, MODEL_FILE)
    results.to_csv(RESULT_FILE, index=False)

    metrics = {
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "WAPE_percent": round(wape, 4),
        "test_days": test_days,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "test_start": str(test_df["date"].min().date()),
        "test_end": str(test_df["date"].max().date()),
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("\nMODEL TRAINING COMPLETE")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"WAPE: {wape:.2f}%")
    print("Saved model:", MODEL_FILE)
    print("Saved results:", RESULT_FILE)
    print("Saved metrics:", METRICS_FILE)

if __name__ == "__main__":
    main()
