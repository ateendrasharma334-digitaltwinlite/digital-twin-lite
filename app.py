import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from simulator import forecast_energy, detect_anomalies

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score

# -------------------------------
# Sidebar Controls (Always Visible)
# -------------------------------
st.sidebar.header("Simulation Settings")
forecast_days = st.sidebar.slider("Select number of forecast days", 7, 60, 30)
building = st.sidebar.selectbox(
    "Select Building",
    ["Building A", "Building B", "Building C"]
)
role = st.sidebar.radio("Role", ["User", "Admin"])

# Building multipliers (for demo purposes)
multiplier = {"Building A": 1, "Building B": 1.2, "Building C": 0.8}

# -------------------------------------------------
# App Title
# -------------------------------------------------
st.title("🏢 Digital Twin Lite - Energy Dashboard")
st.markdown("---")

# -------------------------------
# CSV Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload Building Energy CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    df = clean_data(df)
    st.subheader(f"Data Preview - {building}")
    st.dataframe(df.head())

# Now other Streamlit code

# -------------------------------
# Generate Forecast
# -------------------------------
st.subheader("📈 Energy Forecast")

forecast_df = forecast_energy(forecast_days)
forecast_df = detect_anomalies(forecast_df)

# -------------------------------
# Plot Forecast
# -------------------------------
fig = px.line(forecast_df, x="date", y="forecast", title="Energy Forecast")
anomalies = forecast_df[forecast_df["anomaly"] == True]
fig.add_scatter(x=anomalies["date"], y=anomalies["forecast"],
                mode='markers', marker=dict(color='red', size=10),
                name="Anomaly")
st.plotly_chart(fig)

# -------------------------------
# Anomaly Table
# -------------------------------
st.subheader("⚠️ Anomaly Detection")
if len(anomalies) > 0:
    st.warning("Energy spike detected!")
    st.dataframe(anomalies)
else:
    st.success("No anomalies detected")

    # -------------------------------
# 🔥 Anomaly Heatmap
# -------------------------------
st.subheader("🔥 Anomaly Heatmap")

# Convert boolean anomaly to numeric (0/1)
forecast_df["anomaly_flag"] = forecast_df["anomaly"].astype(int)

heatmap_fig = px.density_heatmap(
    forecast_df,
    x="date",
    y="forecast",
    z="anomaly_flag",
    title="Energy Spike Intensity Heatmap"
)

st.plotly_chart(heatmap_fig)

# -------------------------------
# System Health Indicator
# -------------------------------
st.subheader("🖥 System Health Status")
health_score = max(0, 100 - len(anomalies) / len(forecast_df) * 100)
if len(anomalies) > 0:
    st.metric("System Health (%)", round(health_score, 2), delta=f"-{len(anomalies)} anomalies")
else:
    st.metric("System Health (%)", 100)

# -------------------------------
# Machine Learning Model Accuracy (R²)
# -------------------------------
st.subheader("📊 Model Accuracy & Feature Importance")

# Re-train model on past 200 days to calculate R²
historical_days = 200
dates = pd.date_range(end=pd.Timestamp.today(), periods=historical_days)
energy = 100 + np.sin(np.linspace(0, 20, historical_days))*10 + np.random.normal(0,5,historical_days)
df_hist = pd.DataFrame({"date": dates, "energy": energy})
df_hist["day"] = df_hist["date"].dt.day
df_hist["month"] = df_hist["date"].dt.month
df_hist["weekday"] = df_hist["date"].dt.weekday

X = df_hist[["day","month","weekday"]]
y = df_hist["energy"]

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X,y)
y_pred = model.predict(X)
r2 = r2_score(y, y_pred)

st.metric("Model Accuracy (R²)", round(r2,3))

# -------------------------------
# Feature Importance
# -------------------------------
importance = pd.DataFrame({
    "feature": ["day","month","weekday"],
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

fig2 = px.bar(importance, x="feature", y="importance", title="Feature Importance")
st.plotly_chart(fig2)

# -------------------------------
# Business KPI: Energy Cost
# -------------------------------
st.subheader("💰 Estimated Energy Cost")

cost_per_unit = 0.12  # £ per kWh
total_cost = forecast_df["forecast"].sum() * cost_per_unit
st.metric("Estimated Cost (£)", round(total_cost,2))

# -------------------------------
# 🌍 CO₂ Emissions KPI
# -------------------------------
st.subheader("🌍 Environmental Impact")

co2_factor = 0.233  # kg CO₂ per kWh (UK average grid factor)

total_energy = forecast_df["forecast"].sum()
total_co2 = total_energy * co2_factor

st.metric("Estimated CO₂ Emissions (kg)", round(total_co2, 2))

# ===============================
# Executive Summary Calculations
# ===============================

# Average efficiency proxy (based on health score)
avg_efficiency = health_score

# Sustainability Score (you already calculated earlier)
# sustainability_score already exists in your code

# Optimization Status
if avg_efficiency > 90:
    optimization_status = "Optimized ✅"
elif avg_efficiency > 75:
    optimization_status = "Moderate ⚠️"
else:
    optimization_status = "Needs Improvement ❌"

# -------------------------------
# 💡 Optimization Suggestions
# -------------------------------
st.subheader("💡 AI Optimization Suggestions")

if len(anomalies) > 0:

    anomaly_percentage = (len(anomalies) / len(forecast_df)) * 100

    st.warning(f"{len(anomalies)} energy spikes detected ({round(anomaly_percentage,2)}%).")

    st.markdown("""
    ### Recommended Actions:
    - Adjust HVAC temperature setpoints
    - Reschedule heavy equipment during off-peak hours
    - Inspect high-energy consuming machinery
    - Check sensor calibration
    - Consider predictive maintenance scheduling
    """)

else:
    st.success("Energy usage is stable. No optimization required.")

    # -------------------------------
# 🌱 Sustainability Score
# -------------------------------
st.subheader("🌱 Sustainability Score")

# Normalize CO2 impact (simple scaling for demo)
max_expected_co2 = 5000  # adjust depending on forecast size
co2_impact_score = max(0, 100 - (total_co2 / max_expected_co2) * 100)

sustainability_score = (health_score * 0.6) + (co2_impact_score * 0.4)

st.metric("Overall Sustainability Score (%)", round(sustainability_score, 2))

# -------------------------------
# 🌍 CO₂ Trend Graph
# -------------------------------
st.subheader("📉 CO₂ Emission Trend")

forecast_df["co2_emission"] = forecast_df["forecast"] * co2_factor

fig_co2 = px.line(
    forecast_df,
    x="date",
    y="co2_emission",
    title="Projected CO₂ Emissions Over Time"
)

# -------------------------------
# 📊 Executive Summary
# -------------------------------
st.markdown("## 📊 Executive Summary")

# Optimization Status Logic
if health_score > 90:
    optimization_status = "Optimized ✅"
elif health_score > 75:
    optimization_status = "Moderate ⚠️"
else:
    optimization_status = "Needs Improvement ❌"

col1, col2, col3, col4, col5 = st.columns(5)

col1.metric("⚡ Total Energy (kWh)", round(total_energy, 2))
col2.metric("💰 Estimated Cost (£)", round(total_cost, 2))
col3.metric("🌍 CO₂ Emissions (kg)", round(total_co2, 2))
col4.metric("🌱 Sustainability Score (%)", round(sustainability_score, 2))
col5.metric("🖥 Optimization Status", optimization_status)

st.markdown("---")
st.plotly_chart(fig_co2)

# -------------------------------
# ⚡ Live IoT Energy Simulation
# -------------------------------
import time

st.subheader("⚡ Live IoT Sensor Simulation")

placeholder = st.empty()

simulate = st.button("Start Live Simulation")

if simulate:
    for i in range(20):  # simulate 20 readings
        live_energy = 100 + np.random.normal(0, 5)
        placeholder.metric("Current Live Energy (kWh)", round(live_energy, 2))
        time.sleep(1)

# -------------------------------
# 📥 Download Forecast Report
# -------------------------------
st.subheader("📥 Download Energy Report")

csv = forecast_df.to_csv(index=False).encode("utf-8")

st.download_button(
    label="Download Forecast Report (CSV)",
    data=csv,
    file_name="energy_forecast_report.csv",
    mime="text/csv",
)

# -------------------------------
# 📄 Generate PDF Report
# -------------------------------
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import Table
import io

st.subheader("📄 Download Executive PDF Report")

def generate_pdf():
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph("Digital Twin Energy Report", styles["Title"]))
    elements.append(Spacer(1, 0.3 * inch))

    data = [
        ["Total Energy (kWh)", round(total_energy,2)],
        ["Estimated Cost (£)", round(total_cost,2)],
        ["CO₂ Emissions (kg)", round(total_co2,2)],
        ["Sustainability Score (%)", round(sustainability_score,2)],
        ["System Health (%)", round(health_score,2)]
    ]

    table = Table(data)
    elements.append(table)

    doc.build(elements)
    buffer.seek(0)
    return buffer

pdf = generate_pdf()

st.download_button(
    label="Download PDF Report",
    data=pdf,
    file_name="Digital_Twin_Report.pdf",
    mime="application/pdf"
)
