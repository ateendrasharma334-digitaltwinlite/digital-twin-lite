import numpy as np
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
from simulator import forecast_energy, detect_anomalies
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import streamlit_authenticator as stauth
import time
import io
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
import requests
import sqlite3
import random

# -------------------------------
# Database Connection
# -------------------------------
conn = sqlite3.connect("digital_twin.db", check_same_thread=False)
cursor = conn.cursor()

# -------------------------------
# Create Tables (Run once automatically)
# -------------------------------
cursor.execute("""
CREATE TABLE IF NOT EXISTS sensor_data (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    temperature REAL,
    vibration REAL,
    pressure REAL,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS maintenance_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    health_score REAL,
    status TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# -------------------------------
# Weather Function
# -------------------------------
def get_weather(city="London"):
    api_key = "cb3641e75a4ebf00f22d873a0474d7a0"

    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    data = response.json()

    if response.status_code == 200:
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "description": data["weather"][0]["description"]
        }
    else:
        return None

# -------------------------------
# Predictive Maintenance Function
# -------------------------------
def predictive_maintenance(temp, vibration, pressure):
    health_score = 100 - (temp * 0.2 + vibration * 5 + pressure * 0.1)

    if health_score > 70:
        status = "Healthy"
    elif health_score > 40:
        status = "Warning"
    else:
        status = "Critical"

    return health_score, status

# -------------------------------
# User Credentials (NEW STYLE)
# -------------------------------
credentials = {
    "usernames": {
        "admin": {
            "name": "Administrator",
            "password": "admin123"
        },
        "user1": {
            "name": "Standard User",
            "password": "user123"
        }
    }
}

# -------------------------------
# Authentication Setup
# -------------------------------
authenticator = stauth.Authenticate(
    credentials,
    cookie_name="digital_twin_lite",
    key="auth",
    cookie_expiry_days=1
)

# -------------------------------
# Login
# -------------------------------
authenticator.login(location="main")

if st.session_state.get("authentication_status"):
    st.success(f"Welcome {st.session_state.get('name')} 👋")

elif st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")

else:
    st.warning("Please enter your username and password")

# -------------------------------
# Wrap entire dashboard
# -------------------------------
if st.session_state.get("authentication_status"):

    # -------------------------------
    # Page config
    # -------------------------------
    st.set_page_config(
        page_title="Digital Twin Lite",
        page_icon="assets/logo.png",
        layout="wide"
    )
    # -------------------------------
    # Logout Button
    # -------------------------------
    authenticator.logout("Logout", "sidebar")

    # -------------------------------
    # Generate Sensor Data
    # -------------------------------
    temp_value = random.uniform(20, 80)
    vibration_value = random.uniform(0.1, 5.0)
    pressure_value = random.uniform(10, 50)

    # -------------------------------
    # Insert Sensor Data into DB
    # -------------------------------
    cursor.execute("""
    INSERT INTO sensor_data (temperature, vibration, pressure)
    VALUES (?, ?, ?)
    """, (temp_value, vibration_value, pressure_value))

    conn.commit()

    # -------------------------------
    # Predictive Maintenance Call
    # -------------------------------
    health_score, status = predictive_maintenance(
        temp_value, vibration_value, pressure_value
    )

    st.metric("Machine Health Score", f"{health_score:.2f}")

    if status == "Healthy":
        st.success("✅ Equipment Healthy")
    elif status == "Warning":
        st.warning("⚠ Maintenance Recommended")
    else:
        st.error("🚨 Immediate Maintenance Required")

    # -------------------------------
    # Store Maintenance Result
    # -------------------------------
    cursor.execute("""
    INSERT INTO maintenance_logs (health_score, status)
    VALUES (?, ?)
    """, (health_score, status))

    conn.commit()

    # -------------------------------
    # 🌦 Weather Section (ADD HERE)
    # -------------------------------
    st.sidebar.markdown("## 🌦 Live Weather")

    city = st.sidebar.text_input("Enter City", "London")

    weather = get_weather(city)

    if weather:
        st.sidebar.write(f"🌡 Temp: {weather['temperature']} °C")
        st.sidebar.write(f"💧 Humidity: {weather['humidity']}%")
        st.sidebar.write(f"🌬 Wind: {weather['wind_speed']} m/s")
        st.sidebar.write(f"🌤 {weather['description']}")
    else:
        st.sidebar.error("City not found")


    # -------------------------------
    # Cache Data Loading
    # -------------------------------
    @st.cache_data
    def load_data(file):
        return pd.read_csv(file)

    # -------------------------------
    # Data Cleaning Function
    # -------------------------------
    def clean_data(df):
        df.columns = df.columns.str.strip().str.lower()
        return df

    # -------------------------------
    # Sidebar Controls
    # -------------------------------
    st.sidebar.header("Simulation Settings")
    forecast_days = st.sidebar.slider("Select number of forecast days", 7, 60, 30)
    building = st.sidebar.selectbox(
        "Select Building",
        ["Building A", "Building B", "Building C"]
    )
    role = st.sidebar.radio("Role", ["User", "Admin"])
    if role == "Admin":
        st.subheader("Admin Controls")
        st.write("Advanced analytics visible only to admin.")

    multiplier = {"Building A": 1, "Building B": 1.2, "Building C": 0.8}

    # -------------------------------
    # App Title
    # -------------------------------
    st.title("🏢 Digital Twin Lite - Energy Dashboard")
    st.markdown("---")

    # -------------------------------
    # CSV Upload
    # -------------------------------
    uploaded_file = st.file_uploader("Upload Building Energy CSV", type=["csv"])
    if uploaded_file is not None:
        try:
            df = load_data(uploaded_file)
            df = clean_data(df)
        except Exception as e:
            st.error(f"Error reading file: {e}")
            st.stop()

        required_columns = ["energy_kwh"]
        if not all(col in df.columns for col in required_columns):
            st.error("CSV must contain 'energy_kwh' column.")
            st.stop()

        st.success("CSV uploaded successfully!")
        st.write(df.head())

        if df.isnull().sum().sum() > 0:
            st.warning("Missing values detected. Filling with forward fill method.")
            df.fillna(method="ffill", inplace=True)

        df["energy_kwh"] = pd.to_numeric(df["energy_kwh"], errors="coerce")
        if df["energy_kwh"].isnull().sum() > 0:
            st.error("Energy column contains invalid values.")
            st.stop()
        if (df["energy_kwh"] < 0).any():
            st.warning("Negative energy values detected. Converting to absolute values.")
            df["energy_kwh"] = df["energy_kwh"].abs()

        st.subheader(f"Data Preview - {building}")
        st.dataframe(df.head())

        # -------------------------------
        # Feature Engineering
        # -------------------------------
        if "date" in df.columns:
            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            df["day"] = df["date"].dt.day
            df["month"] = df["date"].dt.month
            df["weekday"] = df["date"].dt.weekday
        else:
            st.warning("No date column found. Forecast will use synthetic dates.")

        # -------------------------------
        # Building Comparison
        # -------------------------------
        st.markdown("---")
        st.subheader("🏢 Building Comparison")
        if "building" in df.columns:
            selected_buildings = st.multiselect(
                "Select buildings to compare",
                options=df["building"].unique()
            )
            if selected_buildings:
                comparison_df = df[df["building"].isin(selected_buildings)]
                building_kpis = comparison_df.groupby("building").agg(
                    total_energy=("energy_kwh", "sum"),
                    avg_energy=("energy_kwh", "mean"),
                    max_energy=("energy_kwh", "max")
                ).reset_index()
                st.dataframe(building_kpis)
                fig = px.bar(building_kpis, x="building", y="total_energy", title="Total Energy Comparison")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No 'building' column found in dataset.")

        # -------------------------------
        # Energy Forecast & Anomalies
        # -------------------------------
        st.subheader("📈 Energy Forecast")
        forecast_df = forecast_energy(forecast_days)
        forecast_df = detect_anomalies(forecast_df)

        fig = px.line(forecast_df, x="date", y="forecast", title="Energy Forecast")
        anomalies = forecast_df[forecast_df["anomaly"] == True]
        fig.add_scatter(x=anomalies["date"], y=anomalies["forecast"],
                        mode='markers', marker=dict(color='red', size=10),
                        name="Anomaly")
        st.plotly_chart(fig)

        st.subheader("⚠️ Anomaly Detection")
        if len(anomalies) > 0:
            st.warning("Energy spike detected!")
            st.dataframe(anomalies)
        else:
            st.success("No anomalies detected")

        # Heatmap
        st.subheader("🔥 Anomaly Heatmap")
        forecast_df["anomaly_flag"] = forecast_df["anomaly"].astype(int)
        heatmap_fig = px.density_heatmap(forecast_df, x="date", y="forecast", z="anomaly_flag",
                                         title="Energy Spike Intensity Heatmap")
        st.plotly_chart(heatmap_fig)

        # -------------------------------
        # System Health
        # -------------------------------
        st.subheader("🖥 System Health Status")
        health_score = max(0, 100 - len(anomalies) / len(forecast_df) * 100)
        if len(anomalies) > 0:
            st.metric("System Health (%)", round(health_score, 2), delta=f"-{len(anomalies)} anomalies")
        else:
            st.metric("System Health (%)", 100)

        # -------------------------------
        # ML Model Accuracy
        # -------------------------------
        st.subheader("📊 Model Accuracy & Feature Importance")
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
        importance = pd.DataFrame({"feature": ["day","month","weekday"], "importance": model.feature_importances_}).sort_values("importance", ascending=False)
        fig2 = px.bar(importance, x="feature", y="importance", title="Feature Importance")
        st.plotly_chart(fig2)

        # -------------------------------
        # KPI: Cost, CO2, Sustainability
        # -------------------------------
        st.subheader("💰 Estimated Energy Cost")
        cost_per_unit = 0.12
        total_cost = forecast_df["forecast"].sum() * cost_per_unit
        st.metric("Estimated Cost (£)", round(total_cost,2))

        st.subheader("🌍 Environmental Impact")
        co2_factor = 0.233
        total_energy = forecast_df["forecast"].sum()
        total_co2 = total_energy * co2_factor
        st.metric("Estimated CO₂ Emissions (kg)", round(total_co2, 2))

        max_expected_co2 = 5000
        co2_impact_score = max(0, 100 - (total_co2 / max_expected_co2) * 100)
        sustainability_score = (health_score * 0.6) + (co2_impact_score * 0.4)
        st.metric("Overall Sustainability Score (%)", round(sustainability_score, 2))

        # -------------------------------
        # Live Simulation
        # -------------------------------
        st.subheader("⚡ Live IoT Sensor Simulation")
        placeholder = st.empty()
        simulate = st.button("Start Live Simulation")
        if simulate:
            for i in range(20):
                live_energy = 100 + np.random.normal(0, 5)
                placeholder.metric("Current Live Energy (kWh)", round(live_energy, 2))
                time.sleep(1)

        # -------------------------------
        # Download CSV & PDF
        # -------------------------------
        st.subheader("📥 Download Energy Report")
        csv = forecast_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Forecast Report (CSV)",
            data=csv,
            file_name="energy_forecast_report.csv",
            mime="text/csv",
        )

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

elif st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")

else:
    st.warning("Please enter your username and password")


