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
from database import init_db, get_connection

st.set_page_config(
        page_title="Digital Twin Lite",
        page_icon="assets/logo.png",
        layout="wide"
    )


# -------------------------------
# Sensor Simulator
# -------------------------------
def generate_sensor_data():
    data = {
        "Temperature (°C)": round(random.uniform(20, 35), 2),
        "Humidity (%)": round(random.uniform(30, 80), 2),
        "Energy (kWh)": round(random.uniform(100, 500), 2),
        "Vibration": round(random.uniform(0.1, 2.5), 2)
    }
    return data


# -------------------------------
# Database Utilities
# -------------------------------
DB_PATH = "digital_twin.db"

def get_connection():
    return sqlite3.connect(DB_PATH, check_same_thread=False, timeout=30)

def init_db():
    with get_connection() as conn:
        conn.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL,
            vibration REAL,
            pressure REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)
        conn.execute("""
        CREATE TABLE IF NOT EXISTS maintenance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            health_score REAL,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

def insert_sensor_data(temp, vibration, pressure):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO sensor_data (temperature, vibration, pressure)
            VALUES (?, ?, ?)
        """, (temp, vibration, pressure))

def insert_maintenance_log(health_score, status):
    with get_connection() as conn:
        conn.execute("""
            INSERT INTO maintenance_logs (health_score, status)
            VALUES (?, ?)
        """, (health_score, status))

def fetch_maintenance_history(limit=10):
    with get_connection() as conn:
        return conn.execute("""
            SELECT timestamp, health_score, status
            FROM maintenance_logs
            ORDER BY timestamp DESC
            LIMIT ?
        """, (limit,)).fetchall()

def fetch_sensor_data():
    with get_connection() as conn:
        return pd.read_sql_query("SELECT * FROM sensor_data", conn)

# -------------------------------
# Initialize Database
# -------------------------------
init_db()

# -------------------------------
# App Title
# -------------------------------
st.title("🏢 Digital Twin Lite - Energy Dashboard")


# -------------------------------
# Weather Function
# -------------------------------
def get_weather(city="London"):
    api_key = "cb3641e75a4ebf00f22d873a0474d7a0"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
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
# AI Fault Detection
# -------------------------------
def detect_faults(temp, vibration, energy):

    alerts = []

    if temp > 70:
        alerts.append("🔥 Overheating detected")

    if vibration > 4:
        alerts.append("⚠ High vibration detected")

    if energy > 450:
        alerts.append("⚡ Energy spike detected")

    return alerts

# -------------------------------
# Digital Twin AI Copilot Brain
# -------------------------------
def ai_copilot(question, df, forecast_df):

    question = question.lower()

    # KPI calculations
    avg_energy = df["energy_kwh"].mean() if df is not None else 0
    max_energy = df["energy_kwh"].max() if df is not None else 0
    total_energy = forecast_df["forecast"].sum() if forecast_df is not None else 0

    # Smart responses
    if "average" in question:
        return f"⚡ Average energy consumption is {avg_energy:.2f} kWh"

    elif "peak" in question or "highest" in question:
        return f"📈 Peak energy usage reached {max_energy:.2f} kWh"

    elif "cost" in question:
        cost = total_energy * 0.12
        return f"💰 Estimated total energy cost is £{cost:.2f}"

    elif "co2" in question or "carbon" in question:
        co2 = total_energy * 0.233
        return f"🌍 Estimated CO₂ emissions are {co2:.2f} kg"

    elif "optimize" in question:
        return "⚙️ Recommendation: Reduce peak load hours and improve equipment efficiency to save energy."

    elif "fault" in question:
        return "🔧 System shows potential faults if vibration > 4 or temperature > 70°C."

    else:
        return "🤖 I can help with energy, cost, CO₂, faults, and optimization insights!"

# -------------------------------
# ✨ Typing Effect (AI feel)
# -------------------------------
def typewriter_effect(text):
    placeholder = st.empty()
    displayed_text = ""

    for char in text:
        displayed_text += char
        placeholder.markdown(f"**🤖 AI:** {displayed_text}")
        time.sleep(0.02)

    return placeholder

# -------------------------------
# Train Failure Prediction Model
# -------------------------------
def train_failure_model():

    # Generate synthetic training data
    np.random.seed(42)

    temperature = np.random.uniform(20, 90, 500)
    vibration = np.random.uniform(0.1, 6, 500)
    energy = np.random.uniform(100, 500, 500)

    # Create labels (0 = healthy, 1 = failure risk)
    failure = (temperature > 70) | (vibration > 4) | (energy > 450)
    failure = failure.astype(int)

    df_train = pd.DataFrame({
        "temperature": temperature,
        "vibration": vibration,
        "energy": energy,
        "failure": failure
    })

    X = df_train[["temperature", "vibration", "energy"]]
    y = df_train["failure"]

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)

    return model

# -------------------------------
# Train AI Failure Model
# -------------------------------
failure_model = train_failure_model()

# -------------------------------
# User Credentials 
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

authenticator.login(location="main")

if st.session_state.get("authentication_status") is False:
    st.error("Username/password is incorrect")
    st.stop()
elif st.session_state.get("authentication_status") is None:
    st.warning("Please enter your username and password")
    st.stop()
else:
    st.success(f"Welcome {st.session_state.get('name')} 👋")

# -------------------------------
# Wrap entire dashboard
# -------------------------------
if st.session_state.get("authentication_status"):

    authenticator.logout("Logout", "sidebar")

    # -------------------------------
    # AI Copilot Memory
    # -------------------------------
    if "copilot_history" not in st.session_state:
        st.session_state.copilot_history = []

    # -------------------------------
    # Generate Sensor Data
    # -------------------------------
    temp_value = random.uniform(20, 80)
    vibration_value = random.uniform(0.1, 5.0)
    pressure_value = random.uniform(10, 50)

    # -------------------------------
    # 🔧 Turbine Sensor Dashboard
    # -------------------------------
    st.subheader("🔧 Turbine Sensor Data")

    col1, col2, col3 = st.columns(3)

    col1.metric("Temperature", f"{temp_value:.2f} °C")
    col2.metric("Vibration", f"{vibration_value:.2f} mm/s")
    col3.metric("Pressure", f"{pressure_value:.2f} bar")

    # Insert into DB safely
    insert_sensor_data(temp_value, vibration_value, pressure_value)

    # -------------------------------
    # Predictive Maintenance Call
    # -------------------------------
    health_score, status = predictive_maintenance(
        temp_value, vibration_value, pressure_value
    )

    st.metric("Machine Health Score", f"{health_score:.2f}")

    # -------------------------------
    # Turbine Health Gauge
    # -------------------------------
    import plotly.graph_objects as go

    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=health_score,
        title={"text": "Turbine Health Score"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": "green"},
            "steps": [
                {"range": [0, 40], "color": "red"},
                {"range": [40, 70], "color": "yellow"},
                {"range": [70, 100], "color": "lightgreen"}
            ]
        }
    ))

    st.plotly_chart(gauge, use_container_width=True)

    if status == "Healthy":
        st.success("✅ Equipment Healthy")
    elif status == "Warning":
        st.warning("⚠ Maintenance Recommended")
    else:
        st.error("🚨 Immediate Maintenance Required")

    # -------------------------------
    # AI Predictive Maintenance Timeline
    # -------------------------------

    import random
    from datetime import datetime, timedelta

    # Simple failure estimation logic
    if health_score > 80:
        days_to_failure = random.randint(25, 40)
    elif health_score > 60:
        days_to_failure = random.randint(15, 25)
    elif health_score > 40:
        days_to_failure = random.randint(7, 15)
    else:
        days_to_failure = random.randint(1, 7)

    predicted_failure_date = datetime.now() + timedelta(days=days_to_failure)

    st.subheader("🧠 AI Predictive Maintenance Timeline")

    col1, col2 = st.columns(2)

    col1.metric("Estimated Days to Failure", f"{days_to_failure} Days")

    col2.metric(
        "Predicted Failure Date",
        predicted_failure_date.strftime("%Y-%m-%d")
    )

    # Maintenance recommendation
    if days_to_failure < 7:
        st.error("🚨 Immediate Maintenance Required")
    elif days_to_failure < 15:
        st.warning("⚠ Maintenance Recommended Soon")
    else:
        st.success("✅ Equipment Operating Normally")

    # -------------------------------
    # Maintenance History
    # -------------------------------
    st.subheader("🛠 Maintenance History")
    history = fetch_maintenance_history()
    for row in history:
        st.write(f"{row[0]} | Health Score: {row[1]:.2f} | Status: {row[2]}")

    # -------------------------------
    # Historical Dashboard
    # -------------------------------
    # Sensor Data History
    st.subheader("📊 Sensor Data History")
    df = fetch_sensor_data()
    st.dataframe(df)
    fig = px.line(df, x="timestamp", y="temperature", title="Temperature Trend")
    st.plotly_chart(fig, use_container_width=True)


    # -------------------------------
    # Weather Section
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
        # 🏢 Building Energy Ranking System
        # -------------------------------
        st.subheader("🏆 Building Energy Efficiency Ranking")

        # Check if building column exists
        if "building" in df.columns:

            # Calculate total energy per building
            building_energy = df.groupby("building")["energy_kwh"].sum().reset_index()

            # Sort from lowest to highest energy consumption
            ranking = building_energy.sort_values(by="energy_kwh")

            st.dataframe(ranking)

            # Bar chart ranking
            st.bar_chart(ranking.set_index("building"))

            # Efficiency labels
            for index, row in ranking.iterrows():

                building_name = row["building"]
                energy_value = row["energy_kwh"]

                if energy_value < 5000:
                    st.success(f"✅ {building_name} — Efficient Energy Usage")
                elif energy_value < 8000:
                    st.warning(f"⚠ {building_name} — Moderate Energy Usage")
                else:
                    st.error(f"🚨 {building_name} — High Energy Consumption")

        else:
            st.info("Add a 'building' column in the CSV to enable building ranking.")

    # -------------------------------
    # 🧠 AI Energy Optimization Advisor
    # -------------------------------
    st.subheader("🧠 AI Energy Optimization Advisor")

    if "building" in df.columns:

        # Calculate average energy per building
        building_avg = df.groupby("building")["energy_kwh"].mean().reset_index()

        for index, row in building_avg.iterrows():

            building_name = row["building"]
            avg_energy = row["energy_kwh"]

            # AI Logic for recommendations
            if avg_energy > 400:

                reduction = round((avg_energy - 300) / avg_energy * 100, 1)

                st.error(f"🚨 {building_name}")
                st.write(f"High energy usage detected: {round(avg_energy,2)} kWh")

                st.write(f"💡 Recommendation: Reduce consumption by approx **{reduction}%**")

                st.info("""
                Suggested Actions:
                - Optimize HVAC scheduling
                - Reduce peak-hour load
                - Upgrade inefficient equipment
                """)

            elif avg_energy > 250:

                st.warning(f"⚠ {building_name}")
                st.write(f"Moderate energy usage: {round(avg_energy,2)} kWh")

                st.info("""
                Suggested Actions:
                - Monitor peak usage times
                - Improve insulation efficiency
                """)

            else:

                st.success(f"✅ {building_name}")
                st.write(f"Efficient energy usage: {round(avg_energy,2)} kWh")

                st.info("No major optimization needed")

    else:
        st.info("Add a 'building' column in CSV to enable AI optimization insights.")
    
    # -------------------------------
    # 💰 AI Cost Savings Predictor
    # -------------------------------
    st.subheader("💰 AI Cost Savings Predictor")

    cost_per_kwh = 0.12  # £ per kWh

    if "building" in df.columns:

        building_stats = df.groupby("building")["energy_kwh"].mean().reset_index()

        for _, row in building_stats.iterrows():

            building_name = row["building"]
            avg_energy = row["energy_kwh"]

            # Assume optimal energy baseline
            optimal_energy = 250  

            if avg_energy > optimal_energy:

                excess_energy = avg_energy - optimal_energy

                # Monthly savings estimation (30 days)
                monthly_savings = excess_energy * cost_per_kwh * 30

                st.error(f"💸 {building_name}")
                st.write(f"Potential Monthly Savings: £{round(monthly_savings,2)}")

                st.info("💡 Reduce HVAC load, optimize schedules, and eliminate idle consumption")

            else:

                st.success(f"✅ {building_name}")
                st.write("No major cost savings needed — already optimized")

    else:
        st.info("Add 'building' column to enable cost predictions")


    # -------------------------------
    # 🌍 Carbon Reduction Advisor
    # -------------------------------
    st.subheader("🌍 Carbon Reduction Advisor")

    co2_factor = 0.233  # kg CO2 per kWh

    if "building" in df.columns:

        building_stats = df.groupby("building")["energy_kwh"].mean().reset_index()

        for _, row in building_stats.iterrows():

            building_name = row["building"]
            avg_energy = row["energy_kwh"]

            optimal_energy = 250

            if avg_energy > optimal_energy:

                excess_energy = avg_energy - optimal_energy

                # Monthly CO2 reduction potential
                co2_savings = excess_energy * co2_factor * 30

                reduction_percent = round((excess_energy / avg_energy) * 100, 1)

                st.warning(f"🌿 {building_name}")
                st.write(f"Potential CO₂ Reduction: {round(co2_savings,2)} kg/month")
                st.write(f"Reduction Opportunity: {reduction_percent}%")

                st.info("""
                Suggested Actions:
                - Shift to renewable energy sources
                - Improve equipment efficiency
                - Reduce peak-hour usage
                """)

            else:

                st.success(f"🌱 {building_name}")
                st.write("Low carbon footprint — operating efficiently")

    else:
        st.info("Add 'building' column to enable carbon insights")

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
        # 🧠 Initialize Copilot Memory
        # -------------------------------
        if "copilot_history" not in st.session_state:
            st.session_state.copilot_history = []

        # -------------------------------
        # 🤖 Digital Twin AI Copilot
        # -------------------------------
        st.markdown("---")

        # 🧠 Initialize memory
        if "copilot_history" not in st.session_state:
            st.session_state.copilot_history = []

        st.subheader("🤖 Digital Twin AI Copilot")

        user_input = st.text_input("Ask your building anything...")

        if user_input:
            response = ai_copilot(user_input, df, forecast_df)

            # Save chat history
            st.session_state.copilot_history.append(("You", user_input))
            st.session_state.copilot_history.append(("AI", response))

        # Display chat history
        for role, msg in st.session_state.copilot_history:
            if role == "You":
                st.markdown(f"**🧑 You:** {msg}")
            else:
                st.markdown(f"**🤖 AI:** {msg}")

        # -------------------------------
        # Live IoT Sensor Simulation + AI Failure Detection
        # -------------------------------
        st.subheader("⚡ Live IoT Sensor Simulation")

        # Button to start simulation
        start = st.button("Start Real-Time Simulation")

        # Placeholder container for dynamic updates
        placeholder = st.empty()

        # Store live sensor history
        live_data_records = []

        if start:
            for i in range(20):

                # -------------------------------
                # Generate Sensor Data
                # -------------------------------
                sensor_data = generate_sensor_data()
                live_data_records.append(sensor_data)

                # -------------------------------
                # Fault Detection (RULE-BASED)
                # -------------------------------
                alerts = detect_faults(
                    sensor_data["Temperature (°C)"],
                    sensor_data["Vibration"],
                    sensor_data["Energy (kWh)"]
                )

                # -------------------------------
                # AI Failure Prediction
                # -------------------------------
                failure_probability = failure_model.predict([[
                    sensor_data["Temperature (°C)"],
                    sensor_data["Vibration"],
                    sensor_data["Energy (kWh)"]
                ]])[0]

                failure_percent = round(failure_probability * 100, 2)

                # -------------------------------
                # Dashboard Display
                # -------------------------------
                df_live = pd.DataFrame(live_data_records)

                with placeholder.container():

                    col1, col2, col3, col4, col5 = st.columns(5)

                    col1.metric("Temperature (°C)", sensor_data["Temperature (°C)"])
                    col2.metric("Humidity (%)", sensor_data["Humidity (%)"])
                    col3.metric("Energy (kWh)", sensor_data["Energy (kWh)"])
                    col4.metric("Vibration", sensor_data["Vibration"])
                    col5.metric("Failure Risk", f"{failure_percent}%")

                    st.progress(int(failure_percent))

                    # Show alerts
                    for alert in alerts:
                        st.warning(alert)
                
                    st.line_chart(df_live)

                    # -------------------------------
                    # ⚡ Energy Control Panel
                    # -------------------------------
                    st.subheader("⚡ Energy Control Panel")

                    avg_energy = df_live["Energy (kWh)"].mean()
                    max_energy = df_live["Energy (kWh)"].max()
                    min_energy = df_live["Energy (kWh)"].min()

                    col1, col2, col3 = st.columns(3)

                    col1.metric("Average Energy", f"{avg_energy:.2f} kWh")
                    col2.metric("Peak Energy", f"{max_energy:.2f} kWh")
                    col3.metric("Minimum Energy", f"{min_energy:.2f} kWh")
                
                time.sleep(1)

        # -------------------------------
        # 🤖 AI Chatbot - Ask Your Building (Advanced)
        # -------------------------------
        st.markdown("---")
        st.subheader("🤖 Ask Your Building AI Assistant")

        # Initialize chat history
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        # User input
        user_query = st.text_input("Ask something about your building performance:")

        # Function to generate smart response
        def generate_ai_response(query, df, forecast_df):

            query = query.lower()

            avg_energy = df["energy_kwh"].mean()
            max_energy = df["energy_kwh"].max()
            total_energy = df["energy_kwh"].sum()

            forecast_avg = forecast_df["forecast"].mean()

            # Smart responses
            if "average" in query:
                return f"📊 Average energy consumption is {avg_energy:.2f} kWh."

            elif "maximum" in query or "peak" in query:
                return f"⚡ Peak energy usage reached {max_energy:.2f} kWh."

            elif "total" in query:
                return f"🔢 Total energy consumption is {total_energy:.2f} kWh."

            elif "forecast" in query:
                return f"🔮 Future average energy is predicted around {forecast_avg:.2f} kWh."

            elif "cost" in query:
                cost = total_energy * 0.12
                return f"💰 Estimated cost is £{cost:.2f}."

            elif "co2" in query or "carbon" in query:
                co2 = total_energy * 0.233
                return f"🌍 Estimated CO₂ emissions are {co2:.2f} kg."

            elif "optimize" in query or "reduce" in query:
                return "💡 Suggestion: Reduce peak loads during high usage hours and improve HVAC efficiency."

            elif "fault" in query or "failure" in query:
                return "⚠️ System monitoring shows potential anomalies. Check vibration and temperature trends."

            elif "ranking" in query:
                if "building" in df.columns:
                    ranking = df.groupby("building")["energy_kwh"].sum().sort_values()
                    best = ranking.index[0]
                    return f"🏆 Most efficient building is {best}."
                else:
                    return "ℹ️ No building-wise data available."

            else:
                return "🤖 I can help with energy, cost, CO₂, forecast, faults, and optimization insights."

        # Process user query
        if user_query:
            response = generate_ai_response(user_query, df, forecast_df)

            # Save chat
            st.session_state.chat_history.append(("You", user_query))
            st.session_state.chat_history.append(("AI", response))

        # Display chat history
        for sender, message in st.session_state.chat_history:
            if sender == "You":
                st.markdown(f"🧑‍💻 **You:** {message}")
            else:
                st.markdown(f"🤖 **AI:** {message}")

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
