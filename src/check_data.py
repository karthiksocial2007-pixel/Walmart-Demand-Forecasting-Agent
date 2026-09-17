import pandas as pd

sales = pd.read_csv("data/sales_train_validation.csv")
calendar = pd.read_csv("data/calendar.csv")
prices = pd.read_csv("data/sell_prices.csv")

print("\n========== SALES DATA ==========")
print("Shape:", sales.shape)
print("\nColumns:")
print(sales.columns.tolist())
print("\nFirst 5 rows:")
print(sales.head())

print("\n========== CALENDAR DATA ==========")
print("Shape:", calendar.shape)
print("\nColumns:")
print(calendar.columns.tolist())
print("\nFirst 5 rows:")
print(calendar.head())

print("\n========== PRICE DATA ==========")
print("Shape:", prices.shape)
print("\nColumns:")
print(prices.columns.tolist())
print("\nFirst 5 rows:")
print(prices.head())