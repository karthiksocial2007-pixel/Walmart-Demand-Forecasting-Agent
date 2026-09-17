from pathlib import Path
import os
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
OUTPUT_DIR = DATA_DIR / "processed"

SALES_FILE = DATA_DIR / "sales_train_validation.csv"
CALENDAR_FILE = DATA_DIR / "calendar.csv"
PRICES_FILE = DATA_DIR / "sell_prices.csv"
OUTPUT_FILE = OUTPUT_DIR / "m5_features.csv"

STORE = "CA_1"
CATEGORY = "FOODS"
N_PRODUCTS = 100

def main():
    print("Loading Walmart M5 historical data...")
    for path in (SALES_FILE, CALENDAR_FILE, PRICES_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Required dataset file not found: {path}")

    sales = pd.read_csv(SALES_FILE)
    calendar = pd.read_csv(CALENDAR_FILE)
    prices = pd.read_csv(PRICES_FILE)

    sales = sales[(sales["store_id"] == STORE) & (sales["cat_id"] == CATEGORY)].copy()
    if sales.empty:
        raise ValueError(f"No rows found for store={STORE}, category={CATEGORY}.")

    # Keep the same hackathon scope: first 100 products after filtering.
    sales = sales.head(N_PRODUCTS).copy()

    id_columns = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    day_columns = [c for c in sales.columns if c.startswith("d_")]
    if not day_columns:
        raise ValueError("No d_ sales columns were found in sales_train_validation.csv.")

    print(f"Selected products: {sales['item_id'].nunique()}")
    print("Converting sales from wide to long format...")

    sales_long = sales.melt(
        id_vars=id_columns,
        value_vars=day_columns,
        var_name="d",
        value_name="sales",
    )

    calendar_cols = [
        "d", "date", "wm_yr_wk", "weekday", "wday", "month", "year",
        "event_name_1", "event_type_1", "event_name_2", "event_type_2",
        "snap_CA", "snap_TX", "snap_WI"
    ]
    calendar_cols = [c for c in calendar_cols if c in calendar.columns]
    sales_long = sales_long.merge(calendar[calendar_cols], on="d", how="left")

    prices = prices[
        (prices["store_id"] == STORE) &
        (prices["item_id"].isin(sales["item_id"]))
    ].copy()

    sales_long = sales_long.merge(
        prices[["store_id", "item_id", "wm_yr_wk", "sell_price"]],
        on=["store_id", "item_id", "wm_yr_wk"],
        how="left",
    )

    sales_long["date"] = pd.to_datetime(sales_long["date"], errors="coerce")
    if sales_long["date"].isna().any():
        raise ValueError("Some sales rows could not be matched to calendar dates.")

    sales_long = sales_long.sort_values(["item_id", "date"]).reset_index(drop=True)

    # Prices are product-level historical features. Fill gaps safely.
    sales_long["sell_price"] = (
        sales_long.groupby("item_id")["sell_price"]
        .transform(lambda s: s.ffill().bfill())
        .fillna(0)
    )

    sales_long["day_of_week"] = sales_long["date"].dt.dayofweek.astype(int)
    sales_long["day"] = sales_long["date"].dt.day.astype(int)
    sales_long["week"] = sales_long["date"].dt.isocalendar().week.astype(int)
    sales_long["month"] = sales_long["date"].dt.month.astype(int)
    sales_long["year"] = sales_long["date"].dt.year.astype(int)
    sales_long["is_weekend"] = (sales_long["day_of_week"] >= 5).astype(int)

    event_1 = sales_long["event_name_1"].fillna("").astype(str).str.strip()
    event_2 = sales_long["event_name_2"].fillna("").astype(str).str.strip()
    sales_long["has_event"] = ((event_1 != "") | (event_2 != "")).astype(int)

    sales_long["price_change"] = (
        sales_long.groupby("item_id")["sell_price"]
        .pct_change()
        .replace([float("inf"), -float("inf")], 0)
        .fillna(0)
    )

    grouped = sales_long.groupby("item_id", sort=False)["sales"]
    for lag in (1, 7, 14, 28):
        sales_long[f"sales_lag_{lag}"] = grouped.shift(lag)

    shifted = grouped.shift(1)
    sales_long["rolling_mean_7"] = (
        shifted.groupby(sales_long["item_id"]).transform(
            lambda s: s.rolling(7, min_periods=7).mean()
        )
    )
    sales_long["rolling_mean_14"] = (
        shifted.groupby(sales_long["item_id"]).transform(
            lambda s: s.rolling(14, min_periods=14).mean()
        )
    )
    sales_long["rolling_mean_28"] = (
        shifted.groupby(sales_long["item_id"]).transform(
            lambda s: s.rolling(28, min_periods=28).mean()
        )
    )
    sales_long["rolling_std_7"] = (
        shifted.groupby(sales_long["item_id"]).transform(
            lambda s: s.rolling(7, min_periods=7).std()
        )
    )

    feature_cols = [
        "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
        "rolling_mean_7", "rolling_mean_14", "rolling_mean_28",
        "rolling_std_7"
    ]
    sales_long = sales_long.dropna(subset=feature_cols).copy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sales_long.to_csv(OUTPUT_FILE, index=False)

    print("\nPREPROCESSING COMPLETE")
    print("Shape:", sales_long.shape)
    print("Products:", sales_long["item_id"].nunique())
    print(
        "Date range:",
        sales_long["date"].min().date(),
        "to",
        sales_long["date"].max().date(),
    )
    print("Saved:", OUTPUT_FILE)

if __name__ == "__main__":
    main()
