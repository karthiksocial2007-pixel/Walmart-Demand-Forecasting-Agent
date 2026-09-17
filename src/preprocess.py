import pandas as pd
import os

print("Loading M5 data...")

sales = pd.read_csv("data/sales_train_validation.csv")
calendar = pd.read_csv("data/calendar.csv")
prices = pd.read_csv("data/sell_prices.csv")

print("All files loaded.")

print("Converting sales data to long format...")

id_columns = [
    "id",
    "item_id",
    "dept_id",
    "cat_id",
    "store_id",
    "state_id"
]

day_columns = [col for col in sales.columns if col.startswith("d_")]

sales_long = sales.melt(
    id_vars=id_columns,
    value_vars=day_columns,
    var_name="d",
    value_name="sales"
)

print("Sales converted.")

print("Merging calendar data...")

sales_long = sales_long.merge(
    calendar,
    on="d",
    how="left"
)

print("Calendar merged.")

print("Merging price data...")

sales_long = sales_long.merge(
    prices,
    on=["store_id", "item_id", "wm_yr_wk"],
    how="left"
)

print("Prices merged.")

sales_long["date"] = pd.to_datetime(sales_long["date"])

os.makedirs("data/processed", exist_ok=True)

output_file = "data/processed/sales_processed.csv"

sales_long.to_csv(output_file, index=False)

print("\nProcessing completed.")
print("Shape:", sales_long.shape)
print("Saved to:", output_file)

print("\nColumns:")
print(sales_long.columns.tolist())

print("\nSample:")
print(sales_long.head())