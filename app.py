from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from forecast import forecast_product

DATA_FILE = ROOT / "data" / "processed" / "m5_features.csv"
CALENDAR_FILE = ROOT / "data" / "calendar.csv"
MODEL_FILE = ROOT / "models" / "xgboost_demand_model.pkl"
RESULT_FILE = ROOT / "outputs" / "forecast_results.csv"
METRICS_FILE = ROOT / "outputs" / "metrics.json"

st.set_page_config(
    page_title="Walmart AI | Demand Intelligence",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container {padding-top: 1.5rem; padding-bottom: 2rem;}
    .hero {
        padding: 1.4rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #334155 100%);
        color: white;
        margin-bottom: 1.2rem;
    }
    .hero h1 {margin: 0; font-size: 2.2rem;}
    .hero p {margin: .35rem 0 0; color: #dbeafe;}
    .section-label {
        font-size: .78rem; font-weight: 700; letter-spacing: .12em;
        text-transform: uppercase; margin-top: .8rem; margin-bottom: .3rem;
        opacity: .7;
    }
    .insight {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 14px; padding: 1rem; min-height: 145px;
        background: rgba(128,128,128,.05);
    }
    .small-note {font-size: .82rem; opacity: .7;}
    </style>
    """,
    unsafe_allow_html=True,
)

@st.cache_data(show_spinner=False)
def load_data():
    df = pd.read_csv(DATA_FILE, parse_dates=["date"])
    calendar = pd.read_csv(CALENDAR_FILE, parse_dates=["date"])
    return df, calendar

@st.cache_resource(show_spinner=False)
def load_model():
    bundle = joblib.load(MODEL_FILE)
    return bundle["model"], bundle.get("features", [])

@st.cache_data(show_spinner=False)
def load_results():
    if not RESULT_FILE.exists():
        return pd.DataFrame()
    return pd.read_csv(RESULT_FILE, parse_dates=["date"])

@st.cache_data(show_spinner=False)
def load_metrics():
    if not METRICS_FILE.exists():
        return {}
    with open(METRICS_FILE, encoding="utf-8") as f:
        return json.load(f)

required_files = [DATA_FILE, CALENDAR_FILE, MODEL_FILE]
missing = [str(p.relative_to(ROOT)) for p in required_files if not p.exists()]
if missing:
    st.error("Project files are missing: " + ", ".join(missing))
    st.info("Run these commands first: python src/prepare_data.py  →  python src/train_model.py")
    st.stop()

df, calendar = load_data()
model, model_features = load_model()
results = load_results()
metrics = load_metrics()

if df.empty:
    st.error("Processed dataset is empty.")
    st.stop()

products = sorted(df["item_id"].dropna().unique().tolist())

with st.sidebar:
    st.markdown("## ⚙️ Controls")
    selected_item = st.selectbox("Product", products, index=0)

    history_days = st.slider(
        "Historical days shown",
        min_value=30,
        max_value=min(365, max(30, int(df["date"].nunique()))),
        value=min(90, max(30, int(df["date"].nunique()))),
        step=30,
    )

    st.divider()
    st.markdown("### Inventory inputs")
    current_stock = st.number_input(
        "Current Inventory (demo input)",
        min_value=0.0,
        value=100.0,
        step=10.0,
    )
    lead_time = st.number_input(
        "Lead Time (days)",
        min_value=1,
        max_value=30,
        value=7,
        step=1,
    )
    safety_factor = st.slider(
        "Safety Stock Factor",
        min_value=0.0,
        max_value=2.0,
        value=0.5,
        step=0.1,
    )

    st.divider()
    st.caption("Dataset: Walmart M5 Forecasting Accuracy")
    st.caption("Scope: CA_1 • FOODS • first 100 products")
    st.caption("Inventory and lead time are user/demo inputs.")

item_df = df[df["item_id"] == selected_item].sort_values("date").copy()
if item_df.empty:
    st.error("No data found for the selected product.")
    st.stop()

latest_date = item_df["date"].max()
latest_actual = float(item_df["sales"].iloc[-1])

recent = float(item_df["sales"].tail(7).mean())
previous = float(item_df["sales"].tail(14).head(7).mean())
growth = 0.0 if previous == 0 else (recent - previous) / previous * 100.0

if growth > 5:
    trend = "Increasing"
elif growth < -5:
    trend = "Decreasing"
else:
    trend = "Stable"

weekday_avg = item_df.groupby("day_of_week")["sales"].mean()
peak_weekday = int(weekday_avg.idxmax())
weekday_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][peak_weekday]

q1 = float(item_df["sales"].quantile(0.25))
q3 = float(item_df["sales"].quantile(0.75))
iqr = q3 - q1
lower = q1 - 1.5 * iqr
upper = q3 + 1.5 * iqr
latest_is_anomaly = latest_actual < lower or latest_actual > upper

# Historical holdout predictions — only for evaluation context.
product_results = (
    results[results["item_id"] == selected_item].sort_values("date")
    if not results.empty and "item_id" in results.columns
    else pd.DataFrame()
)

# True future forecast: starts after the final observed M5 sales date.
future = forecast_product(
    model=model,
    history_df=item_df,
    calendar_df=calendar,
    prices_df=None,
    item_id=selected_item,
    horizon=7,
)
future_mean = float(future["predicted_sales"].mean())
daily_forecast = max(future_mean, 0.0)

recent_std = float(item_df["rolling_std_7"].iloc[-1])
if not np.isfinite(recent_std):
    recent_std = 0.0

demand_during_lead = daily_forecast * lead_time
safety_stock = max(0.0, recent_std) * safety_factor * np.sqrt(lead_time)
reorder_point = demand_during_lead + safety_stock
recommended_order = max(0.0, reorder_point - current_stock)

# ---------------- HERO ----------------
st.markdown(
    f"""
    <div class="hero">
        <h1>📦 Walmart AI</h1>
        <p>Intelligent Demand Forecasting & Inventory Optimization Agent</p>
        <div class="small-note">Historical M5 retail data → forecast → pattern detection → inventory decision</div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="section-label">Executive Overview</div>', unsafe_allow_html=True)
k1, k2, k3, k4 = st.columns(4)
k1.metric("Latest Daily Demand", f"{latest_actual:,.0f} units")
k2.metric("7-Day Future Forecast", f"{future_mean:,.1f} units/day")
k3.metric("7-Day Growth", f"{growth:+.1f}%")
k4.metric("Demand Trend", trend)

# ---------------- FORECAST ----------------
st.markdown('<div class="section-label">Demand Intelligence</div>', unsafe_allow_html=True)
st.subheader("📈 Historical Demand → AI Future Forecast")

history = item_df.tail(history_days)

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history["date"],
        y=history["sales"],
        mode="lines",
        name="Historical Demand",
        line=dict(width=2),
    )
)
fig.add_trace(
    go.Scatter(
        x=future["date"],
        y=future["predicted_sales"],
        mode="lines+markers",
        name="7-Day Future Forecast",
        line=dict(width=3, dash="dash"),
    )
)
fig.add_vline(
    x=latest_date,
    line_dash="dot",
    annotation_text="Last observed day",
    annotation_position="top left",
)
fig.update_layout(
    height=460,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis_title="Date",
    yaxis_title="Units Sold",
    hovermode="x unified",
    legend=dict(orientation="h", y=1.05, x=0),
)
st.plotly_chart(fig, use_container_width=True)

fc1, fc2, fc3 = st.columns(3)
fc1.metric("Forecast Start", future["date"].min().strftime("%d %b %Y"))
fc2.metric("Forecast End", future["date"].max().strftime("%d %b %Y"))
fc3.metric("7-Day Forecast Total", f"{future['predicted_sales'].sum():,.0f} units")

st.caption(
    "The dashed line is a true recursive future forecast beyond the final observed M5 sales date. "
    "Future price is carried forward from the latest available product price because future prices are not known."
)

# ---------------- INSIGHTS ----------------
st.markdown('<div class="section-label">Explainable AI</div>', unsafe_allow_html=True)
st.subheader("🤖 AI Business Insights")

i1, i2, i3 = st.columns(3)

with i1:
    st.markdown('<div class="insight">', unsafe_allow_html=True)
    st.markdown("### 📊 Demand Trend")
    if trend == "Increasing":
        st.success(f"Recent demand is **{growth:+.1f}%** versus the previous 7-day period.")
    elif trend == "Decreasing":
        st.warning(f"Recent demand is **{growth:+.1f}%** versus the previous 7-day period.")
    else:
        st.info(f"Recent demand is relatively stable at **{growth:+.1f}%**.")
    st.markdown("</div>", unsafe_allow_html=True)

with i2:
    st.markdown('<div class="insight">', unsafe_allow_html=True)
    st.markdown("### 📅 Weekly Seasonality")
    st.info(f"Highest historical average demand occurs on **{weekday_name}**.")
    st.markdown("</div>", unsafe_allow_html=True)

with i3:
    st.markdown('<div class="insight">', unsafe_allow_html=True)
    st.markdown("### 🚨 Anomaly Detection")
    if latest_is_anomaly:
        st.warning("Latest observed demand is outside the product's IQR range.")
    else:
        st.success("Latest observed demand is inside the normal IQR range.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- SEASONALITY ----------------
st.subheader("📅 Weekly Seasonality")
seasonality = weekday_avg.reindex(range(7), fill_value=0)

season_fig = go.Figure(
    go.Bar(
        x=["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"],
        y=seasonality.values,
        text=[f"{v:.0f}" for v in seasonality.values],
        textposition="auto",
        name="Average Demand",
    )
)
season_fig.update_layout(
    height=330,
    margin=dict(l=10, r=10, t=20, b=10),
    xaxis_title="Day of Week",
    yaxis_title="Average Units Sold",
)
st.plotly_chart(season_fig, use_container_width=True)

# ---------------- INVENTORY ----------------
st.markdown('<div class="section-label">Decision Engine</div>', unsafe_allow_html=True)
st.subheader("📦 Inventory Intelligence")

a, b, c, d = st.columns(4)
a.metric("Forecast / Day", f"{daily_forecast:,.1f}")
b.metric("Safety Stock", f"{safety_stock:,.1f}")
c.metric("Reorder Point", f"{reorder_point:,.1f}")
d.metric("Recommended Order", f"{recommended_order:,.0f}")

if current_stock < reorder_point:
    st.error(
        f"⚠️ Inventory is below the estimated reorder point. "
        f"Replenishment of approximately **{recommended_order:,.0f} units** is recommended."
    )
else:
    excess = current_stock - reorder_point
    st.success(
        f"✅ Inventory is above the estimated reorder point by approximately **{excess:,.0f} units**."
    )

with st.expander("How the inventory decision is calculated"):
    st.write("Expected demand during lead time = forecast/day × lead time")
    st.write("Safety stock = recent 7-day demand variability × safety factor × √lead time")
    st.write("Reorder point = expected lead-time demand + safety stock")
    st.write("Recommended order = max(0, reorder point − current inventory)")
    st.caption("Current inventory, lead time and safety factor are demo/user inputs, not M5 fields.")

# ---------------- MODEL PERFORMANCE ----------------
st.markdown('<div class="section-label">Model Evaluation</div>', unsafe_allow_html=True)
st.subheader("🎯 XGBoost Model Performance")

if metrics:
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"{metrics.get('MAE', 0):.2f}")
    m2.metric("RMSE", f"{metrics.get('RMSE', 0):.2f}")
    m3.metric("WAPE", f"{metrics.get('WAPE_percent', 0):.2f}%")
    st.caption(
        f"Time-based evaluation: final {metrics.get('test_days', 28)} historical days "
        f"({metrics.get('test_start', 'N/A')} to {metrics.get('test_end', 'N/A')}). "
        "These metrics evaluate historical holdout predictions; they are separate from the future forecast."
    )
else:
    st.info("Run train_model.py to generate model evaluation metrics.")

# ---------------- DATA DETAILS ----------------
with st.expander("📚 Dataset & Model Details"):
    d1, d2 = st.columns(2)
    with d1:
        st.write("**Dataset:** Walmart M5 Forecasting Accuracy")
        st.write("**Selected Store:** CA_1")
        st.write("**Selected Category:** FOODS")
        st.write("**Products in scope:** first 100 after filtering")
    with d2:
        st.write("**Model:** XGBoost Regressor")
        st.write("**Forecast horizon:** 7 days")
        st.write("**Historical evaluation:** final 28 days")
        st.write("**Features:** lags, rolling statistics, calendar and price features")

st.divider()
st.caption(
    "Hackathon demo • Historical Walmart M5 data • Explainable retail demand and inventory decision support"
)
