# Walmart AI — Intelligent Demand Forecasting & Inventory Optimization

Hackathon-ready Streamlit application based on the **Walmart M5 Forecasting Accuracy** dataset.

## What the project does

**Historical sales → feature engineering → XGBoost → historical evaluation → true 7-day future forecast → trend/seasonality/anomaly → inventory recommendation → business explanation**

The project intentionally stays simple enough for a 5–6 hour hackathon.

### Dataset scope

- Store: `CA_1`
- Category: `FOODS`
- Products: first 100 products after filtering
- Dataset: Walmart M5 Forecasting Accuracy
- Data is historical, not live Walmart data.

The M5 dataset does **not** provide current inventory quantities. Current inventory, lead time and safety-stock factor in the dashboard are user/demo inputs.

## Folder structure

```text
Walmart_Demand_Forecasting/
├── data/
│   ├── sales_train_validation.csv
│   ├── calendar.csv
│   ├── sell_prices.csv
│   └── processed/
├── models/
├── outputs/
├── src/
│   ├── prepare_data.py
│   ├── train_model.py
│   ├── forecast.py
│   └── analysis.py
├── app.py
├── requirements.txt
├── run_project.bat
└── README.md
```

## Setup on Windows

Open VS Code in the project folder.

### 1. Create the virtual environment

```powershell
python -m venv venv
```

### 2. Activate it

```powershell
.\venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
.\venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Put the three M5 files in `data/`

```text
data/sales_train_validation.csv
data/calendar.csv
data/sell_prices.csv
```

### 5. Prepare the data

```powershell
python src/prepare_data.py
```

Expected output:

```text
data/processed/m5_features.csv
```

### 6. Train the model

```powershell
python src/train_model.py
```

Expected outputs:

```text
models/xgboost_demand_model.pkl
outputs/forecast_results.csv
outputs/metrics.json
```

The printed MAE, RMSE and WAPE are the actual metrics generated from your run. Do not replace them with invented values.

### 7. Generate product analysis files

```powershell
python src/analysis.py
```

Outputs:

```text
outputs/trend_analysis.csv
outputs/anomaly_analysis.csv
```

### 8. Start the dashboard

```powershell
streamlit run app.py
```

The browser will open the local Streamlit dashboard.

## One-click run

After creating the environment and installing dependencies, double-click:

```text
run_project.bat
```

It prepares data, trains the model and starts Streamlit.

## Model

The model is an `XGBRegressor` using:

- sell_price
- price_change
- day_of_week
- day
- week
- month
- year
- is_weekend
- has_event
- sales_lag_1
- sales_lag_7
- sales_lag_14
- sales_lag_28
- rolling_mean_7
- rolling_mean_14
- rolling_mean_28
- rolling_std_7

The final 28 historical days are held out for evaluation.

Metrics:

- MAE
- RMSE
- WAPE

MAPE is not used because zero-sales observations can make MAPE problematic.

## True future forecasting

The dashboard now separates two concepts:

1. **Historical evaluation:** predictions for the final 28 observed M5 days.
2. **Future forecast:** a recursive 7-day forecast starting after the final observed M5 sales date.

For recursive forecasting, each predicted day is fed back into the next day's lag and rolling features.

Future prices are not known. The dashboard therefore carries forward the latest available product price rather than pretending future price data exists.

## Inventory engine

The dashboard calculates:

```text
Demand during lead time
= forecast/day × lead time

Safety stock
= recent 7-day demand standard deviation
  × safety factor × √lead time

Reorder point
= demand during lead time + safety stock

Recommended order
= max(0, reorder point − current inventory)
```

Inventory, lead time and safety factor are explicitly demo/user inputs.

## Demo flow

1. Select a product.
2. Show historical demand.
3. Show the 7-day future forecast.
4. Explain the demand trend.
5. Show the strongest weekday pattern.
6. Show anomaly status.
7. Enter current inventory and lead time.
8. Show safety stock and reorder point.
9. Show recommended order quantity.
10. Explain that the system converts prediction into an inventory decision.

## Important wording for the presentation

Say:

> “We use historical Walmart M5 retail data to learn demand patterns. The current inventory and lead time are user-provided business inputs. The agent converts the forecast into an explainable replenishment recommendation.”

Do **not** say that the project has live Walmart inventory data.
