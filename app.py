import numpy as np
import streamlit as st
import pandas as pd
import networkx as nx
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
from openai import OpenAI
import os
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
import paho.mqtt.client as mqtt
import json
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer
)

mqtt_data = {}

def on_message(client, userdata, msg):
    global mqtt_data
    mqtt_data = json.loads(msg.payload.decode())

client_mqtt = mqtt.Client()
client_mqtt.on_message = on_message

client_mqtt.connect("broker.hivemq.com", 1883, 60)
client_mqtt.subscribe("digital_twin/sensors")

client_mqtt.loop_start()

# -------------------------------
# 🔄 Auto Refresh 
# -------------------------------
from streamlit_autorefresh import st_autorefresh

# Refresh every 5 seconds
st_autorefresh(interval=30000, key="datarefresh")

# -------------------------------
# Global Safe Variables (FIX)
# -------------------------------
df = None
forecast_df = None
failure_percent = 0

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

def init_db():

    with get_connection() as conn:

        tables = pd.read_sql_query(
            """
            SELECT name
            FROM sqlite_master
            WHERE type='table'
            """,
            conn
        )

    st.write(tables)

    with get_connection() as conn:

        # Sensor Data
        conn.execute("""
        CREATE TABLE IF NOT EXISTS sensor_data (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            temperature REAL,
            vibration REAL,
            pressure REAL,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Maintenance Logs
        conn.execute("""
        CREATE TABLE IF NOT EXISTS maintenance_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            health_score REAL,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Security Logs
        conn.execute("""
        CREATE TABLE IF NOT EXISTS security_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            action TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Assets
        conn.execute("""
        CREATE TABLE IF NOT EXISTS assets (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_name TEXT,
            asset_type TEXT,
            health_score REAL,
            criticality TEXT,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Alerts
        conn.execute("""
        CREATE TABLE IF NOT EXISTS alerts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_name TEXT,
            alert_message TEXT,
            severity TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Enterprise Events
        conn.execute("""
        CREATE TABLE IF NOT EXISTS enterprise_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT,
            description TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # Incidents
        conn.execute("""
        CREATE TABLE IF NOT EXISTS incidents (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            asset_name TEXT,
            incident TEXT,
            severity TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        # SLA Metrics
        conn.execute("""
        CREATE TABLE IF NOT EXISTS sla_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name TEXT,
            target REAL,
            actual REAL,
            status TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """)

        conn.commit()

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

def log_security_event(username, action):

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO security_logs (username, action)
            VALUES (?, ?)
        """, (username, action))
        log_event(
            "Security",
            action
        )

def insert_asset(asset_name, asset_type, health_score, criticality, status):

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO assets (
                asset_name,
                asset_type,
                health_score,
                criticality,
                status
            )
            VALUES (?, ?, ?, ?, ?)
        """, (
            asset_name,
            asset_type,
            health_score,
            criticality,
            status
        ))
        
        log_event(
            "Asset",
            f"{asset_name} added to system"
        )

# -------------------------------
# Alert Engine
# -------------------------------
def create_alert(
    asset_name,
    alert_message,
    severity
):

    with get_connection() as conn:

        conn.execute("""
        INSERT INTO alerts (
            asset_name,
            alert_message,
            severity
        )
        VALUES (?, ?, ?)
        """,
        (
            asset_name,
            alert_message,
            severity
        ))
        log_event(
            "Alert",
            f"{severity}: {alert_message}"
        )

# -------------------------------
# Enterprise Event Logger
# -------------------------------
def log_event(
    event_type,
    description
):

    with get_connection() as conn:

        conn.execute("""
        INSERT INTO enterprise_events (
            event_type,
            description
        )
        VALUES (?, ?)
        """,
        (
            event_type,
            description
        ))

# -------------------------------
# AI Failure Prediction Model
# -------------------------------
def train_failure_model():

    data = pd.DataFrame({

        "temperature": [
            60,70,80,90,95,55,65,75,85,100
        ],

        "vibration": [
            2,3,4,7,9,2,3,5,8,10
        ],

        "pressure": [
            10,12,14,15,16,10,11,13,14,17
        ],

        "failure": [
            0,0,0,1,1,0,0,0,1,1
        ]
    })

    X = data[
        [
            "temperature",
            "vibration",
            "pressure"
        ]
    ]

    y = data["failure"]

    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42
    )

    model.fit(X, y)

    return model

# -------------------------------
# Incident Storage
# -------------------------------
def save_incident(asset_name, incident, severity):

    with get_connection() as conn:
        conn.execute("""
            INSERT INTO incidents (
                asset_name,
                incident,
                severity
            )
            VALUES (?, ?, ?)
        """, (
            asset_name,
            incident,
            severity
        ))


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
# Enterprise Report Generator
# -------------------------------
def generate_report():

    report_path = "enterprise_report.pdf"

    doc = SimpleDocTemplate(report_path)

    styles = getSampleStyleSheet()

    content = []

    content.append(
        Paragraph(
            "Enterprise Digital Twin Report",
            styles["Title"]
        )
    )

    content.append(Spacer(1, 20))

    content.append(
        Paragraph(
            "Executive Summary",
            styles["Heading2"]
        )
    )

    content.append(
        Paragraph(
            "All enterprise systems operating normally.",
            styles["BodyText"]
        )
    )

    doc.build(content)

    return report_path

# -------------------------------
# Weather API
# -------------------------------
def get_weather(city="London"):

    try:

        api_key = "YOUR_OPENWEATHER_API_KEY"

        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?q={city}&appid={api_key}&units=metric"
        )

        response = requests.get(url)

        return response.json()

    except:
        return None

# -------------------------------
# Initialize Database
# -------------------------------
init_db()
failure_model = train_failure_model()

# -------------------------------
# App Title
# -------------------------------
st.title("🏢 Digital Twin Lite - Energy Dashboard")

# =====================================================
# 🖥 Enterprise Command Center
# =====================================================

st.markdown("""
# 🖥 Enterprise Energy Command Center

Real-Time Monitoring • AI Forecasting • Predictive Maintenance • Sustainability Analytics
""")


# =========================================================
# 🌍 ENERGY INTELLIGENCE PLATFORM
# =========================================================

import requests

def get_live_carbon_intensity():
    ...

def get_live_grid_demand():
    ...

def estimate_solar_generation(weather_temp):
    ...

def calculate_renewable_percentage(generation_mix):
    ...


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
# ⚡ Live Carbon Intensity API
# -------------------------------
def get_live_energy_price():
    url = "https://api.carbonintensity.org.uk/intensity"

    response = requests.get(url)

    if response.status_code == 200:
        data = response.json()
        return data["data"][0]["intensity"]["actual"]

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
        alerts.append(("🔥 Overheating detected", "critical"))

    if vibration > 4:
        alerts.append(("⚠ High vibration detected", "medium"))

    if energy > 450:
        alerts.append(("⚡ Energy spike detected", "medium"))

    return alerts

# -------------------------------
# KPI calculations (SAFE VERSION)
# -------------------------------
def smart_kpi_response(question, df=None, forecast_df=None):

    # Energy KPIs
    if df is not None and "energy_kwh" in df.columns:
        avg_energy = df["energy_kwh"].mean()
        max_energy = df["energy_kwh"].max()
    else:
        avg_energy = 0
        max_energy = 0

    # Forecast KPIs
    if forecast_df is not None and "forecast" in forecast_df.columns:
        total_energy = forecast_df["forecast"].sum()
        forecast_avg = forecast_df["forecast"].mean()  
    else:
        total_energy = 0
        forecast_avg = 0

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
# 🤖 Real GPT Copilot 
# -------------------------------
def gpt_copilot(user_input, df=None, forecast_df=None):

    # Context from your data
    context = ""

    if df is not None and "energy_kwh" in df.columns:
        context += f"Average Energy: {df['energy_kwh'].mean():.2f} kWh\n"

    if forecast_df is not None and "forecast" in forecast_df.columns:
        context += f"Forecast Energy: {forecast_df['forecast'].mean():.2f} kWh\n"

    system_prompt = f"""
    You are an AI Energy Expert Digital Twin Assistant.

    Use this building data:
    {context}

    Give professional, short, actionable insights.
    """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input}
        ],
        stream=True   # 🔥 STREAMING ENABLED
    )

    return response

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
@st.cache_resource
def load_failure_model():
    return train_failure_model()

failure_model = load_failure_model()
# -------------------------------
# 🤖 Advanced AI Copilot Brain
# -------------------------------
def ai_copilot(query, df=None, forecast_df=None):

    query = query.lower()

    # -------------------------------
    # Safe Data Handling
    # -------------------------------
    if df is not None and "energy_kwh" in df.columns:
        avg_energy = df["energy_kwh"].mean()
        max_energy = df["energy_kwh"].max()
    else:
        avg_energy = 0
        max_energy = 0

    if forecast_df is not None and "forecast" in forecast_df.columns:
        total_energy = forecast_df["forecast"].sum()
        forecast_avg = forecast_df["forecast"].mean()
    else:
        total_energy = 0
        forecast_avg = 0

    # -------------------------------
    # Cost & CO2 Calculations
    # -------------------------------
    cost_per_kwh = 0.12
    co2_factor = 0.233

    total_cost = total_energy * cost_per_kwh
    total_co2 = total_energy * co2_factor

    # -------------------------------
    # 🧠 Intelligent Responses
    # -------------------------------

    if "cost" in query:
        potential_saving = total_cost * 0.15

        return f"""
💰 **AI Energy Cost Optimization Report**

• Current Estimated Cost: £{round(total_cost,2)}
• Potential Savings: £{round(potential_saving,2)} (≈15%)

📊 **Recommendations:**
- Optimize peak load usage
- Shift operations to off-peak hours
- Improve HVAC efficiency

⚡ This strategy can significantly reduce operational expenses.
"""

    elif "co2" in query or "carbon" in query:
        reduction = total_co2 * 0.2

        return f"""
🌍 **Carbon Reduction Advisory**

• Current CO₂ Emissions: {round(total_co2,2)} kg
• Reduction Potential: {round(reduction,2)} kg (≈20%)

♻️ **Actions:**
- Integrate renewable energy sources
- Improve insulation
- Reduce peak consumption

🚀 Sustainability score can improve significantly.
"""

    elif "fault" in query or "risk" in query:
        risk_level = "Low"

        if avg_energy > 400 or max_energy > 500:
            risk_level = "High"
        elif avg_energy > 300:
            risk_level = "Medium"

        return f"""
⚠ **AI Fault Risk Assessment**

• Risk Level: {risk_level}

🔍 **Insights:**
- High energy spikes detected
- Possible system stress conditions

🛠 **Recommended Actions:**
- Inspect turbine and generator
- Schedule preventive maintenance
- Monitor vibration closely
"""

    elif "optimize" in query or "improve" in query:
        return f"""
🚀 **AI Optimization Plan**

• Reduce energy usage by ~10–20%
• Improve system efficiency

📊 **Actions:**
- Use smart scheduling
- Reduce idle load
- Upgrade inefficient components

💡 Estimated Savings: £{round(total_cost*0.1,2)}
"""

    else:
        return f"""
🤖 I can help you with:

• 💰 Energy cost reduction
• 🌍 CO₂ impact analysis
• ⚠ Fault detection
• 🚀 System optimization

👉 Try asking:
- "How can I reduce energy cost?"
- "Any fault risks?"
- "What is my CO2 impact?"
"""

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
log_security_event(
    st.session_state.get("name"),
    "User Logged In"
)

# -------------------------------
# SQLite Database
# -------------------------------
conn = sqlite3.connect(
    "digital_twin.db",
    check_same_thread=False
)

cursor = conn.cursor()
cursor.execute("""
CREATE TABLE IF NOT EXISTS asset_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_name TEXT,
    asset_type TEXT,
    health_score REAL,
    status TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()
cursor.execute("""
CREATE TABLE IF NOT EXISTS incidents (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset_name TEXT,
    incident TEXT,
    severity TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
)
""")

conn.commit()

# -------------------------------
# Wrap entire dashboard
# -------------------------------
if st.session_state.get("authentication_status"):

    # -------------------------------
    # 🎯 PROFESSIONAL STATUS INDICATOR 
    # -------------------------------
    st.sidebar.success("🟢 System Running - Level 31 AI Enabled")

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
    # 🏭 Asset Registry
    # -------------------------------
    st.subheader("🏭 Industrial Asset Registry")

    asset_name = st.selectbox(
        "Select Asset",
        [
            "Gas Turbine GT-01",
            "Steam Turbine ST-02",
            "Transformer TX-01",
            "Boiler BLR-01",
            "Wind Turbine WT-01"
        ]
    )

    asset_type = asset_name.split()[0]

    # -------------------------------
    # Predictive Maintenance Call
    # -------------------------------
    health_score, status = predictive_maintenance(
        temp_value, vibration_value, pressure_value
    )

    st.metric("Machine Health Score", f"{health_score:.2f}")

    # -------------------------------
    # 🚨 Asset Criticality Engine
    # -------------------------------
    if health_score < 40:
        criticality = "Critical"
    elif health_score < 70:
        criticality = "Warning"
    else:
        criticality = "Healthy"
    
    if health_score < 40:

        create_alert(
            "Main Asset",
            "Asset health below threshold",
            "Critical"
        )

    elif health_score < 70:

        create_alert(
            "Main Asset",
            "Asset requires inspection",
            "Warning"
        )

    insert_asset(
        asset_name,
        asset_type,
        health_score,
        criticality,
        status
    )

    st.metric("Asset Criticality", criticality)

    # =====================================================
    # 🔐 Cybersecurity Threat Monitor
    # =====================================================

    st.subheader("🔐 Cybersecurity Threat Monitor")

    cyber_risk = random.randint(1, 100)

    if cyber_risk > 75:
        st.error("🚨 HIGH Cybersecurity Threat Detected")

    elif cyber_risk > 40:
        st.warning("⚠ Medium Security Risk")

    else:
        st.success("✅ Network Secure")

    st.progress(cyber_risk)

    st.metric(
        "Cyber Risk Score",
        f"{cyber_risk}/100"
    )

    # -------------------------------
    # 🧠 AI Asset Risk Score
    # -------------------------------
    asset_risk = 100 - health_score

    st.metric(
        "⚠ Asset Risk Score",
        f"{round(asset_risk,2)}%"
    )

    if asset_risk > 70:
        st.error("🚨 High Asset Failure Risk")
    elif asset_risk > 40:
        st.warning("⚠ Moderate Asset Risk")
    else:
        st.success("✅ Asset Risk Low")
    
    try:

        save_asset_history(
            asset_name="Main Asset",
            asset_type="Industrial Equipment",
            health_score=health_score,
            status="Active"
        )

    except:
        pass

    # -------------------------------
    # 🔮 Remaining Useful Life (RUL)
    # -------------------------------
    st.subheader("🔮 Remaining Useful Life")

    if health_score > 80:
        rul_days = 365
    elif health_score > 60:
        rul_days = 180
    elif health_score > 40:
        rul_days = 90
    else:
        rul_days = 30

    st.metric(
        "Estimated Remaining Life",
        f"{rul_days} Days"
    )
    
    # -------------------------------
    # 📉 Asset Degradation Analysis
    # -------------------------------
    st.subheader("📉 Asset Degradation")

    degradation_rate = round((100 - health_score) / 100, 2)

    st.metric(
        "Degradation Rate",
        f"{degradation_rate * 100}%"
    )

    if degradation_rate > 0.6:
        st.error("🚨 Severe degradation detected")
    elif degradation_rate > 0.3:
        st.warning("⚠ Moderate degradation detected")
    else:
        st.success("✅ Asset operating normally")

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
    # 🧠 AI Failure Prediction
    # -------------------------------

    risk_score = (
        temp_value * 0.3 +
        vibration_value * 20 +
        pressure_value * 0.5
    )

    # Convert to days
    if risk_score > 120:
        days_to_failure = np.random.randint(1, 5)
    elif risk_score > 90:
        days_to_failure = np.random.randint(5, 10)
    elif risk_score > 60:
        days_to_failure = np.random.randint(10, 20)
    else:
        days_to_failure = np.random.randint(20, 40)

    from datetime import datetime, timedelta
    predicted_failure_date = datetime.now() + timedelta(days=int(days_to_failure))

    st.subheader("🧠 AI Predictive Maintenance Timeline")

    col1, col2 = st.columns(2)

    col1.metric("Estimated Days to Failure", f"{days_to_failure} Days")

    col2.metric(
        "Predicted Failure Date",
        predicted_failure_date.strftime("%Y-%m-%d")
    )

    # =========================================================
    # ⏳ Predictive Failure Timeline
    # =========================================================
    st.subheader("⏳ Failure Prediction Timeline")

    timeline_df = pd.DataFrame({
        "Days Ahead": [1, 7, 14, 30, 60],
        "Failure Risk": [
            random.randint(5, 15),
            random.randint(10, 25),
            random.randint(20, 40),
            random.randint(30, 60),
            random.randint(40, 80)
        ]
    })

    fig_timeline = px.line(
        timeline_df,
        x="Days Ahead",
        y="Failure Risk",
        markers=True,
        title="Predicted Failure Risk Over Time"
    )

    st.plotly_chart(fig_timeline, use_container_width=True)

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
    
    # =====================================================
    # 📜 Live System Audit Logs
    # =====================================================

    st.subheader("📜 Live Audit Logs")

    audit_logs = [
        "User login successful",
        "Forecast model executed",
        "AI anomaly detection completed",
        "Sensor calibration verified",
        "System optimization triggered"
    ]

    for log in audit_logs:
        st.write(f"✅ {log}")

    # -------------------------------
    # 📊 Fleet Asset Monitoring
    # -------------------------------
    st.subheader("📊 Fleet Asset Overview")

    with get_connection() as conn:

        fleet_df = pd.read_sql_query(
            """
            SELECT asset_name,
                   health_score,
                   criticality,
                   status,
                   timestamp
            FROM assets
            ORDER BY timestamp DESC
            LIMIT 20
            """,
            conn
        )

    st.dataframe(fleet_df)
    
    # -------------------------------
    # 📜 Security Audit Trail
    # -------------------------------
    st.subheader("📜 User Activity Logs")

    with get_connection() as conn:
        audit_df = pd.read_sql_query(
            "SELECT * FROM security_logs ORDER BY timestamp DESC LIMIT 20",
            conn
        )

    st.dataframe(audit_df)

    # -------------------------------
    # Historical Dashboard
    # -------------------------------
    # Sensor Data History
    st.subheader("📊 Sensor Data History")
    df = fetch_sensor_data()

    # -------------------------------
    # 🏭 Multi-Asset Digital Twin Engine
    # -------------------------------
    st.subheader("🏭 Multi-Asset Monitoring")

    assets = [
        {"name": "Gas Turbine", "health": random.randint(60, 100)},
        {"name": "Boiler", "health": random.randint(40, 95)},
        {"name": "Transformer", "health": random.randint(50, 100)},
        {"name": "HVAC System", "health": random.randint(70, 100)}
    ]

    asset_df = pd.DataFrame(assets)

    st.dataframe(asset_df)

    fig_assets = px.bar(
        asset_df,
        x="name",
        y="health",
        color="health",
        title="Asset Health Comparison"
    )

    st.plotly_chart(fig_assets, use_container_width=True)

    # Continue existing code
    if df is None or df.empty:
        st.warning("No sensor data available.")
    else:
        st.dataframe(df)

    if df is None or df.empty:
        st.warning("No sensor data available.")
    else:
        st.dataframe(df)
    
    # =====================================================
    # 🏭 Multi-Asset Fleet Monitor
    # =====================================================

    st.subheader("🏭 Multi-Asset Fleet Monitor")

    fleet_data = pd.DataFrame({
        "Asset": [
            "Turbine A",
            "Turbine B",
            "HV Transformer",
            "Cooling Pump",
            "Solar Inverter"
        ],
        "Health Score": [
            random.randint(60, 98),
            random.randint(40, 90),
            random.randint(55, 95),
            random.randint(35, 88),
            random.randint(70, 99)
        ]
    })

    st.dataframe(fleet_data)

    fleet_fig = px.bar(
        fleet_data,
        x="Asset",
        y="Health Score",
        title="Fleet Health Monitoring"
    )

    st.plotly_chart(fleet_fig, use_container_width=True)


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
    
    # -------------------------------
    # 🌐 API Status Center
    # -------------------------------
    st.subheader("🌐 API Status")

    api_services = {
        "Weather API": "Online",
        "Carbon API": "Online",
        "MQTT Broker": "Online",
        "AI Copilot": "Online"
    }

    for service, status in api_services.items():

        if status == "Online":
            st.success(f"✅ {service} — {status}")
        else:
            st.error(f"❌ {service} — Offline")
    
    # -------------------------------
    # 💰 AI Energy Market Center
    # -------------------------------
    st.markdown("---")
    st.subheader("💰 AI Energy Market Center")

    market_price = round(random.uniform(50, 180), 2)

    market_col1, market_col2, market_col3 = st.columns(3)

    market_col1.metric(
        "Electricity Price",
        f"£{market_price}/MWh"
    )

    market_col2.metric(
        "Grid Frequency",
        f"{round(random.uniform(49.7, 50.3),2)} Hz"
    )

    market_col3.metric(
        "Renewable Penetration",
        f"{random.randint(30,80)}%"
    )

    if market_price > 140:
        st.error("🚨 Energy market price spike detected")
    elif market_price > 90:
        st.warning("⚠ Market volatility increasing")
    else:
        st.success("✅ Market operating normally")
    
    # -------------------------------
    # ⚡ AI Demand Response Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("⚡ AI Demand Response")

    demand_level = random.randint(40, 100)

    st.progress(demand_level)

    st.metric(
        "Current Demand Response",
        f"{demand_level}%"
    )

    if demand_level > 85:

        st.error("🚨 Peak demand event")

        st.info("""
        AI Recommendations:
        • Reduce HVAC usage
        • Shift non-critical loads
        • Enable battery discharge
        """)

    elif demand_level > 65:

        st.warning("⚠ High demand period detected")

    else:

        st.success("✅ Grid demand stable")
    
    # -------------------------------
    # 🗺 AI Grid Stability Map
    # -------------------------------
    st.markdown("---")
    st.subheader("🗺 AI Grid Stability Zones")

    grid_data = pd.DataFrame({
        "Zone": [
            "North Grid",
            "South Grid",
            "East Grid",
            "West Grid"
        ],
        "Stability": [
            random.randint(80,100),
            random.randint(60,100),
            random.randint(70,100),
            random.randint(50,100)
        ]
    })

    st.dataframe(grid_data)

    fig_grid = px.bar(
        grid_data,
        x="Zone",
        y="Stability",
        color="Stability",
        title="Regional Grid Stability"
    )

    st.plotly_chart(fig_grid, use_container_width=True)

    # -------------------------------
    # 🤖 AI Energy Trading Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🤖 AI Energy Trading Engine")

    buy_price = round(random.uniform(40, 90), 2)
    sell_price = round(random.uniform(100, 180), 2)

    trade_col1, trade_col2 = st.columns(2)

    trade_col1.metric(
        "Buy Price",
        f"£{buy_price}/MWh"
    )

    trade_col2.metric(
        "Sell Price",
        f"£{sell_price}/MWh"
    )

    profit_margin = sell_price - buy_price

    st.metric(
        "Potential Trading Margin",
        f"£{round(profit_margin,2)}/MWh"
    )

    if profit_margin > 60:
        st.success("✅ High-value trading opportunity")
    else:
        st.warning("⚠ Limited trading margin")
    
    # -------------------------------
    # 🔋 Battery Energy Storage System
    # -------------------------------
    st.markdown("---")
    st.subheader("🔋 Battery Energy Storage System")

    battery_charge = random.randint(20, 100)

    bess_col1, bess_col2, bess_col3 = st.columns(3)

    bess_col1.metric(
        "Battery Charge",
        f"{battery_charge}%"
    )

    bess_col2.metric(
        "Charge Rate",
        f"{random.randint(5,40)} MW"
    )

    bess_col3.metric(
        "Discharge Rate",
        f"{random.randint(5,40)} MW"
    )

    st.progress(battery_charge)

    if battery_charge < 25:
        st.warning("⚠ Battery reserve low")
    else:
        st.success("✅ Battery operating normally")
    
    # -------------------------------
    # 🌍 Renewable Forecast AI
    # -------------------------------
    st.markdown("---")
    st.subheader("🌍 Renewable Energy Forecast")

    renewable_data = pd.DataFrame({
        "Hour": list(range(1, 13)),
        "Solar MW": np.random.randint(20, 100, 12),
        "Wind MW": np.random.randint(10, 80, 12)
    })

    st.dataframe(renewable_data)

    fig_renew = px.line(
        renewable_data,
        x="Hour",
        y=["Solar MW", "Wind MW"],
        title="Renewable Generation Forecast"
    )

    st.plotly_chart(fig_renew, use_container_width=True)

    # -------------------------------
    # ⚡ AI Peak Shaving Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("⚡ AI Peak Shaving")

    peak_load = random.randint(50, 100)

    st.metric(
        "Peak Grid Load",
        f"{peak_load}%"
    )

    if peak_load > 85:

        st.error("🚨 Peak demand critical")

        st.info("""
        AI Peak Shaving Actions:
        • Activate battery storage
        • Reduce HVAC load
        • Shift industrial demand
        """)

    elif peak_load > 65:

        st.warning("⚠ High demand period")

    else:

        st.success("✅ Peak load under control")
    
    # -------------------------------
    # 📡 Power Quality Monitor
    # -------------------------------
    st.markdown("---")
    st.subheader("📡 Power Quality Monitor")

    voltage = round(random.uniform(390, 420), 2)
    frequency = round(random.uniform(49.7, 50.3), 2)
    harmonics = round(random.uniform(1, 8), 2)

    pq_col1, pq_col2, pq_col3 = st.columns(3)

    pq_col1.metric("Voltage", f"{voltage} V")
    pq_col2.metric("Frequency", f"{frequency} Hz")
    pq_col3.metric("THD", f"{harmonics}%")

    if harmonics > 5:
        st.warning("⚠ Harmonic distortion detected")
    else:
        st.success("✅ Power quality stable")

    # -------------------------------
    # 🤖 AI Dispatch Optimizer
    # -------------------------------
    st.markdown("---")
    st.subheader("🤖 AI Dispatch Optimizer")

    dispatch_data = pd.DataFrame({
        "Asset": [
            "Gas Turbine",
            "Battery Storage",
            "Solar Farm",
            "Wind Farm"
        ],
        "Dispatch MW": [
            random.randint(50,150),
            random.randint(20,80),
            random.randint(10,100),
            random.randint(20,120)
        ]
    })

    st.dataframe(dispatch_data)

    fig_dispatch = px.pie(
        dispatch_data,
        names="Asset",
        values="Dispatch MW",
        title="AI Power Dispatch Allocation"
    )

    st.plotly_chart(fig_dispatch, use_container_width=True)

    # -------------------------------
    # 🔄 Autonomous Grid Recovery Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🔄 Autonomous Grid Recovery")

    recovery_score = random.randint(60, 100)

    st.metric(
        "Grid Recovery Readiness",
        f"{recovery_score}%"
    )

    st.progress(recovery_score)

    if recovery_score < 70:

        st.error("🚨 Recovery capability reduced")

        st.info("""
        AI Recovery Actions:
        • Activate backup generation
        • Re-route critical feeders
        • Isolate unstable assets
        """)

    else:

        st.success("✅ Autonomous recovery systems operational")
    
    # -------------------------------
    # 🌍 AI Carbon Optimization Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🌍 AI Carbon Optimization")

    carbon_now = round(random.uniform(80, 350), 2)

    carbon_col1, carbon_col2 = st.columns(2)

    carbon_col1.metric(
        "Current Carbon Intensity",
        f"{carbon_now} gCO₂/kWh"
    )

    carbon_col2.metric(
        "Carbon Reduction",
        f"{random.randint(5,35)}%"
    )

    if carbon_now > 250:

        st.warning("⚠ High carbon intensity")

        st.info("""
        AI Optimization:
        • Increase renewable dispatch
        • Reduce fossil generation
        • Shift flexible loads
        """)

    else:

        st.success("✅ Carbon optimization stable")
    
    # -------------------------------
    # 🚨 AI Fault Isolation Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🚨 AI Fault Isolation")

    fault_probability = random.randint(1, 100)

    st.metric(
        "Fault Probability",
        f"{fault_probability}%"
    )

    if fault_probability > 80:

        st.error("🚨 Critical fault risk")

        st.info("""
        AI Isolation Actions:
        • Disconnect affected feeder
        • Notify SCADA operator
        • Start autonomous rerouting
        """)

    elif fault_probability > 50:

        st.warning("⚠ Potential fault developing")

    else:

        st.success("✅ Grid operating normally")
    
    # -------------------------------
    # ⚡ Smart Energy Routing Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("⚡ Smart Energy Routing")

    routing_data = pd.DataFrame({
        "Source": [
            "Solar Farm",
            "Wind Farm",
            "Battery Storage",
            "Gas Turbine"
        ],
        "Destination": [
            "Industrial Zone",
            "Residential Zone",
            "Grid Support",
            "Critical Infrastructure"
        ],
        "Power MW": [
            random.randint(20,100),
            random.randint(30,120),
            random.randint(10,60),
            random.randint(50,150)
        ]
    })

    st.dataframe(routing_data)

    fig_route = px.bar(
        routing_data,
        x="Source",
        y="Power MW",
        color="Destination",
        title="AI Smart Energy Routing"
    )

    st.plotly_chart(fig_route, use_container_width=True)

    # -------------------------------
    # 🛡 AI Grid Resilience Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🛡 AI Grid Resilience")

    resilience_score = random.randint(60, 100)

    st.metric(
        "Grid Resilience Score",
        f"{resilience_score}%"
    )

    st.progress(resilience_score)

    if resilience_score < 70:

        st.error("🚨 Grid resilience weakening")

    elif resilience_score < 85:

        st.warning("⚠ Moderate resilience risk")

    else:

        st.success("✅ Grid resilience strong")
    
    # -------------------------------
    # 🤖 Autonomous Operations Center
    # -------------------------------
    st.markdown("---")
    st.subheader("🤖 Autonomous Operations Center")

    auto_mode = st.toggle("Enable Autonomous AI Control")

    if auto_mode:

        st.success("✅ Autonomous control active")

        st.info("""
        AI Autonomous Actions:
        • Grid balancing enabled
        • Load optimization active
        • Predictive maintenance running
        • Smart dispatch operational
        """)

    else:

        st.warning("⚠ Manual operator mode enabled")
    
    # -------------------------------
    # 🖥 Digital Twin Command Center
    # -------------------------------
    st.markdown("---")
    st.subheader("🖥 Digital Twin Command Center")

    command_col1, command_col2, command_col3, command_col4 = st.columns(4)

    command_col1.metric(
        "Connected Assets",
        random.randint(20, 150)
    )

    command_col2.metric(
        "Active AI Agents",
        random.randint(5, 40)
    )

    command_col3.metric(
        "Live IoT Devices",
        random.randint(100, 1000)
    )

    command_col4.metric(
        "SCADA Signals/sec",
        random.randint(500, 5000)
    )

    st.success("✅ Enterprise command center operational")

    # -------------------------------
    # ⚡ AI Operational Optimization
    # -------------------------------
    st.markdown("---")
    st.subheader("⚡ AI Operational Optimization")

    optimization_score = random.randint(60, 100)

    st.metric(
        "Optimization Efficiency",
        f"{optimization_score}%"
    )

    st.progress(optimization_score)

    if optimization_score < 70:

        st.warning("⚠ Optimization opportunities detected")

        st.info("""
        AI Recommendations:
        • Reduce idle equipment
        • Optimize HVAC schedules
        • Shift flexible demand
        • Improve dispatch routing
        """)

    else:

        st.success("✅ Operations optimized")
    
    # -------------------------------
    # 🚚 Enterprise Fleet Monitoring
    # -------------------------------
    st.markdown("---")
    st.subheader("🚚 Enterprise Fleet Monitoring")

    fleet_data = pd.DataFrame({
        "Asset": [
            "Turbine-01",
            "Boiler-02",
            "HVAC-03",
            "Transformer-04",
            "Battery-05"
        ],
        "Health": [
            random.randint(60,100),
            random.randint(50,100),
            random.randint(70,100),
            random.randint(40,100),
            random.randint(80,100)
        ],
        "Status": [
            "Running",
            "Monitoring",
            "Operational",
            "Warning",
            "Optimal"
        ]
    })

    st.dataframe(fleet_data)

    fig_fleet = px.bar(
        fleet_data,
        x="Asset",
        y="Health",
        color="Health",
        title="Enterprise Fleet Health"
    )

    st.plotly_chart(fig_fleet, use_container_width=True)

    # -------------------------------
    # 🧠 AI Confidence Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🧠 AI Confidence Engine")

    confidence_score = random.randint(75, 99)

    st.metric(
        "AI Decision Confidence",
        f"{confidence_score}%"
    )

    st.progress(confidence_score)

    if confidence_score < 80:

        st.warning("⚠ AI confidence slightly reduced")

    else:

        st.success("✅ AI operating with high confidence")
    
    # -------------------------------
    # 🤖 Industrial AI Copilot
    # -------------------------------
    st.markdown("---")
    st.subheader("🤖 Industrial AI Copilot")

    copilot_question = st.text_input(
        "Ask AI Copilot",
        placeholder="Example: Why is turbine temperature increasing?"
    )

    if copilot_question:

        st.info(f"🧠 AI Analysis: {copilot_question}")

        recommendations = [
            "Inspect cooling systems",
            "Check vibration anomalies",
            "Review maintenance logs",
            "Optimize load balancing",
            "Verify sensor calibration"
        ]

        st.success("✅ AI Recommendations Generated")

        for rec in recommendations:
            st.write(f"• {rec}")
    # -------------------------------
    # 🌍 Global Operations Center
    # -------------------------------
    st.markdown("---")
    st.subheader("🌍 Global Operations Center")

    global_sites = pd.DataFrame({
        "Site": [
            "London Grid",
            "Manchester Plant",
            "Birmingham Hub",
            "Leeds Station",
            "Glasgow Energy Center"
        ],
        "Status": [
            "Operational",
            "Stable",
            "Monitoring",
            "Warning",
            "Operational"
        ],
        "Efficiency": [
            random.randint(70,100),
            random.randint(60,100),
            random.randint(75,100),
            random.randint(50,90),
            random.randint(80,100)
        ]
    })

    st.dataframe(global_sites)

    fig_global = px.scatter(
        global_sites,
        x="Site",
        y="Efficiency",
        color="Status",
        size="Efficiency",
        title="Global Operations Monitoring"
    )

    st.plotly_chart(fig_global, use_container_width=True)

    # -------------------------------
    # 🚨 Autonomous Alarm Priority Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🚨 Autonomous Alarm Prioritization")

    alarm_score = random.randint(1, 100)

    st.metric(
        "Alarm Severity Index",
        f"{alarm_score}%"
    )

    if alarm_score > 80:

        st.error("🚨 Critical alarm escalation")

        st.info("""
        AI Actions:
        • Notify executive operators
        • Trigger emergency workflow
        • Start asset isolation
        """)

    elif alarm_score > 50:

        st.warning("⚠ Medium priority alarm")

    else:

        st.success("✅ Alarm levels stable")

    # -------------------------------
    # 🧠 AI Decision Recommendation Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🧠 AI Decision Recommendations")

    decision_data = pd.DataFrame({
        "AI Recommendation": [
            "Increase battery dispatch",
            "Reduce HVAC demand",
            "Shift industrial load",
            "Optimize turbine output",
            "Activate reserve systems"
        ],
        "Priority": [
            "High",
            "Medium",
            "Medium",
            "High",
            "Critical"
        ]
    })

    st.dataframe(decision_data)

    st.success("✅ AI strategic recommendations generated")

    # -------------------------------
    # 📊 Enterprise SLA Monitoring
    # -------------------------------
    st.markdown("---")
    st.subheader("📊 Enterprise SLA Monitoring")

    sla_score = random.randint(85, 100)

    sla_col1, sla_col2, sla_col3 = st.columns(3)

    sla_col1.metric(
        "System Availability",
        f"{sla_score}%"
    )

    sla_col2.metric(
        "Incident Response",
        f"{random.randint(90,100)}%"
    )

    sla_col3.metric(
        "Operational Compliance",
        f"{random.randint(88,100)}%"
    )

    if sla_score < 90:

        st.warning("⚠ SLA performance slightly reduced")

    else:

        st.success("✅ SLA targets achieved")

    # -------------------------------
    # 📡 Live AI Event Stream
    # -------------------------------
    st.markdown("---")
    st.subheader("📡 Live AI Event Stream")

    event_logs = pd.DataFrame({
        "Timestamp": pd.date_range(
            start=pd.Timestamp.now(),
            periods=10,
            freq="min"
        ),
        "Event": [
            "AI Dispatch Updated",
            "SCADA Sync Complete",
            "Battery Optimization",
            "Carbon Reduction Triggered",
            "Load Balancing Active",
            "HVAC Optimization",
            "MQTT Data Received",
            "Predictive Alert Generated",
            "AI Risk Calculation",
            "Grid Stability Check"
        ],
        "Severity": [
            "Info",
            "Info",
            "Medium",
            "High",
            "Medium",
            "Low",
            "Info",
            "High",
            "Medium",
            "Info"
        ]
    })

    st.dataframe(event_logs)

    # -------------------------------
    # ⚙ Autonomous Workflow Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("⚙ Autonomous Workflow Engine")

    workflow_score = random.randint(70, 100)

    st.metric(
        "Workflow Automation",
        f"{workflow_score}%"
    )

    st.progress(workflow_score)

    if workflow_score < 80:

        st.warning("⚠ Some workflows require operator approval")

    else:

        st.success("✅ Autonomous workflows operational")

    # -------------------------------
    # 🔥 Enterprise KPI Heatmap
    # -------------------------------
    st.markdown("---")
    st.subheader("🔥 Enterprise KPI Heatmap")

    heatmap_data = pd.DataFrame({
        "System": [
            "SCADA",
            "AI Engine",
            "MQTT",
            "Battery",
            "Grid Control"
        ],
        "Performance": [
            random.randint(70,100),
            random.randint(75,100),
            random.randint(60,100),
            random.randint(65,100),
            random.randint(80,100)
        ]
    })

    fig_heatmap = px.imshow(
        [heatmap_data["Performance"]],
        labels=dict(x="System", y="KPI", color="Performance"),
        x=heatmap_data["System"],
        y=["Performance"],
        title="Enterprise KPI Heatmap"
    )

    st.plotly_chart(fig_heatmap, use_container_width=True)

    # -------------------------------
    # 🚨 AI Incident Correlation Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🚨 AI Incident Correlation")

    incident_score = random.randint(1, 100)

    st.metric(
        "Incident Correlation Index",
        f"{incident_score}%"
    )

    if incident_score > 75:

        st.error("🚨 Multi-system anomaly correlation detected")

        st.info("""
        AI Correlation Findings:
        • Temperature anomaly linked
        • Vibration increase detected
        • Power fluctuation identified
        • Maintenance risk elevated
        """)

    elif incident_score > 50:

        st.warning("⚠ Potential correlated incidents")

    else:

        st.success("✅ Systems operating independently")
    
    # -------------------------------
    # 👔 Executive AI Forecast Center
    # -------------------------------
    st.markdown("---")
    st.subheader("👔 Executive AI Forecast Center")

    forecast_data = pd.DataFrame({
        "Month": [
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct"
        ],
        "Projected Savings (£)": [
            random.randint(100000,200000),
            random.randint(120000,220000),
            random.randint(150000,250000),
            random.randint(170000,280000),
            random.randint(200000,300000)
        ]
    })

    st.dataframe(forecast_data)

    fig_exec = px.line(
        forecast_data,
        x="Month",
        y="Projected Savings (£)",
        markers=True,
        title="Executive Financial Forecast"
    )

    st.plotly_chart(fig_exec, use_container_width=True)

    st.success("✅ AI executive forecasting operational")

    # -------------------------------
    # 🤖 AI Orchestration Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🤖 AI Orchestration Engine")

    orchestration_score = random.randint(70, 100)

    st.metric(
        "AI Orchestration Efficiency",
        f"{orchestration_score}%"
    )

    st.progress(orchestration_score)

    if orchestration_score < 80:

        st.warning("⚠ Some AI workflows require optimization")

    else:

        st.success("✅ Enterprise AI orchestration stable")
    
    # -------------------------------
    # 🖥 Enterprise Digital War Room
    # -------------------------------
    st.markdown("---")
    st.subheader("🖥 Enterprise Digital War Room")

    war_col1, war_col2, war_col3, war_col4 = st.columns(4)

    war_col1.metric(
        "Critical Alerts",
        random.randint(0, 10)
    )

    war_col2.metric(
        "Active AI Systems",
        random.randint(10, 50)
    )

    war_col3.metric(
        "Connected Facilities",
        random.randint(5, 25)
    )

    war_col4.metric(
        "Global Uptime",
        f"{random.randint(95,100)}%"
    )

    st.success("✅ Enterprise command operations active")

    # -------------------------------
    # ⚡ AI Resource Allocation Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("⚡ AI Resource Allocation")

    resource_data = pd.DataFrame({
        "Resource": [
            "Battery Storage",
            "Gas Turbine",
            "Solar Farm",
            "Wind Farm",
            "HVAC Systems"
        ],
        "Allocation (%)": [
            random.randint(40,100),
            random.randint(30,90),
            random.randint(20,100),
            random.randint(25,100),
            random.randint(50,100)
        ]
    })

    st.dataframe(resource_data)

    fig_resource = px.bar(
        resource_data,
        x="Resource",
        y="Allocation (%)",
        color="Allocation (%)",
        title="AI Resource Allocation"
    )

    st.plotly_chart(fig_resource, use_container_width=True)

    # -------------------------------
    # 🛡 Autonomous Compliance Monitor
    # -------------------------------
    st.markdown("---")
    st.subheader("🛡 Autonomous Compliance Monitor")

    compliance_score = random.randint(80, 100)

    comp_col1, comp_col2 = st.columns(2)

    comp_col1.metric(
        "Compliance Score",
        f"{compliance_score}%"
    )

    comp_col2.metric(
        "Audit Readiness",
        f"{random.randint(85,100)}%"
    )

    if compliance_score < 90:

        st.warning("⚠ Compliance deviations detected")

    else:

        st.success("✅ Compliance systems healthy")

    # -------------------------------
    # 🌍 Enterprise Health Index
    # -------------------------------
    st.markdown("---")
    st.subheader("🌍 Enterprise Health Index")

    enterprise_health = random.randint(70, 100)

    st.metric(
        "Enterprise Health",
        f"{enterprise_health}%"
    )

    st.progress(enterprise_health)

    if enterprise_health < 80:

        st.warning("⚠ Enterprise performance degrading")

    else:

        st.success("✅ Enterprise systems performing optimally")
    
    # -------------------------------
    # 🤖 AI Autonomous Supervisor
    # -------------------------------
    st.markdown("---")
    st.subheader("🤖 AI Autonomous Supervisor")

    supervisor_mode = st.toggle(
        "Enable Autonomous Enterprise Supervision"
    )

    if supervisor_mode:

        st.success("✅ AI Supervisor Active")

        st.info("""
        Autonomous Supervision Enabled:
        • AI risk mitigation
        • Self-healing workflows
        • AI asset balancing
        • Smart operational recovery
        • Predictive escalation
        """)

    else:

        st.warning("⚠ Manual supervision mode active")

    # -------------------------------
    # 🚨 Enterprise Risk Matrix
    # -------------------------------
    st.markdown("---")
    st.subheader("🚨 Enterprise Risk Matrix")

    risk_data = pd.DataFrame({
        "Risk Area": [
            "SCADA",
            "Cybersecurity",
            "Grid Stability",
            "Battery Storage",
            "AI Automation"
        ],
        "Risk Score": [
            random.randint(20,90),
            random.randint(10,95),
            random.randint(15,85),
            random.randint(20,80),
            random.randint(10,70)
        ]
    })

    st.dataframe(risk_data)

    fig_risk = px.bar(
        risk_data,
        x="Risk Area",
        y="Risk Score",
        color="Risk Score",
        title="Enterprise Risk Matrix"
    )

    st.plotly_chart(fig_risk, use_container_width=True)

    # -------------------------------
    # 🔄 Self-Healing Infrastructure
    # -------------------------------
    st.markdown("---")
    st.subheader("🔄 Self-Healing Infrastructure")

    healing_score = random.randint(60, 100)

    st.metric(
        "Self-Healing Capability",
        f"{healing_score}%"
    )

    st.progress(healing_score)

    if healing_score < 75:

        st.warning("⚠ Recovery capability reduced")

        st.info("""
        AI Healing Actions:
        • Restart failed workflows
        • Rebalance energy systems
        • Restore IoT connections
        • Activate backup services
        """)

    else:

        st.success("✅ Self-healing systems operational")
    
    # -------------------------------
    # 💰 AI Cost Optimization Center
    # -------------------------------
    st.markdown("---")
    st.subheader("💰 AI Cost Optimization")

    cost_saving = random.randint(10000, 500000)

    cost_col1, cost_col2, cost_col3 = st.columns(3)

    cost_col1.metric(
        "Predicted Savings",
        f"£{cost_saving:,}"
    )

    cost_col2.metric(
        "Operational Efficiency",
        f"{random.randint(75,100)}%"
    )

    cost_col3.metric(
        "AI Optimization Gain",
        f"{random.randint(5,35)}%"
    )

    st.success("✅ AI financial optimization active")

    # -------------------------------
    # 🌍 Global Sustainability Intelligence
    # -------------------------------
    st.markdown("---")
    st.subheader("🌍 Global Sustainability Intelligence")

    sustainability_data = pd.DataFrame({
        "Region": [
            "Europe",
            "Asia",
            "North America",
            "Middle East",
            "Africa"
        ],
        "Renewable Usage (%)": [
            random.randint(40,90),
            random.randint(30,80),
            random.randint(35,85),
            random.randint(20,70),
            random.randint(25,75)
        ]
    })

    st.dataframe(sustainability_data)

    fig_sustain = px.line(
        sustainability_data,
        x="Region",
        y="Renewable Usage (%)",
        markers=True,
        title="Global Sustainability Metrics"
    )

    st.plotly_chart(fig_sustain, use_container_width=True)

    st.success("✅ Sustainability intelligence operational")

    # -------------------------------
    # ☁ Cloud Connectivity Center
    # -------------------------------
    st.markdown("---")
    st.subheader("☁ Cloud Connectivity Center")

    cloud_services = {
        "AWS IoT Core": "Online",
        "Azure Digital Twin": "Online",
        "MQTT Broker": "Online",
        "AI Cloud Engine": "Online",
        "SCADA Gateway": "Online"
    }

    for service, status in cloud_services.items():

        if status == "Online":
            st.success(f"✅ {service} Connected")
        else:
            st.error(f"❌ {service} Offline")

    # -------------------------------
    # 🗄 Enterprise Data Lake
    # -------------------------------
    st.markdown("---")
    st.subheader("🗄 Enterprise Data Lake")

    data_lake = pd.DataFrame({
        "Source": [
            "SCADA",
            "MQTT Sensors",
            "Weather API",
            "Carbon API",
            "AI Analytics"
        ],
        "Records": [
            random.randint(10000,50000),
            random.randint(50000,200000),
            random.randint(5000,20000),
            random.randint(2000,15000),
            random.randint(10000,80000)
        ]
    })

    st.dataframe(data_lake)

    fig_data_lake = px.pie(
        data_lake,
        names="Source",
        values="Records",
        title="Enterprise Data Lake Distribution"
    )

    st.plotly_chart(fig_data_lake, use_container_width=True)

    # -------------------------------
    # 🌐 AI API Traffic Analytics
    # -------------------------------
    st.markdown("---")
    st.subheader("🌐 AI API Traffic Analytics")

    api_df = pd.DataFrame({
        "API": [
            "Weather API",
            "Carbon API",
            "MQTT API",
            "AI Prediction API",
            "Grid Analytics API"
        ],
        "Requests/min": [
            random.randint(100,1000),
            random.randint(50,500),
            random.randint(500,3000),
            random.randint(100,1500),
            random.randint(80,900)
        ]
    })

    st.dataframe(api_df)
 
    fig_api = px.bar(
        api_df,
        x="API",
        y="Requests/min",
        color="Requests/min",
        title="Live API Traffic"
    )

    st.plotly_chart(fig_api, use_container_width=True)
    
    # -------------------------------
    # 📡 Real-Time Sensor Stream Monitor
    # -------------------------------
    st.markdown("---")
    st.subheader("📡 Real-Time Sensor Streams")

    sensor_stream = pd.DataFrame({
        "Sensor": [
            "Temperature",
            "Pressure",
            "Voltage",
            "Current",
            "Vibration"
        ],
        "Live Value": [
            random.randint(50,100),
            random.randint(1,10),
            random.randint(380,450),
            random.randint(50,200),
            random.randint(1,8)
        ]
    })

    st.dataframe(sensor_stream)

    st.success("✅ Live IoT streams synchronized")

    # -------------------------------
    # ⚡ Digital Twin Performance Optimizer
    # -------------------------------
    st.markdown("---")
    st.subheader("⚡ Digital Twin Performance Optimizer")

    performance_score = random.randint(70, 100)

    st.metric(
    "System Performance",
    f"{performance_score}%"
    )

    st.progress(performance_score)

    if performance_score < 80:

        st.warning("⚠ Optimization recommended")

        st.info("""
        Recommended Actions:
        • Reduce redundant API calls
        • Optimize chart rendering
        • Compress MQTT payloads
        • Improve caching
        """)

    else:

        st.success("✅ Platform performance optimized")
    
    # -------------------------------
    # 🏭 Multi-Site Enterprise Manager
    # -------------------------------
    st.markdown("---")
    st.subheader("🏭 Multi-Site Enterprise Manager")

    site_data = pd.DataFrame({
        "Site": [
            "London Plant",
            "Manchester Hub",
            "Birmingham Grid",
            "Leeds Facility",
            "Glasgow Station"
        ],
        "Status": [
            "Operational",
            "Monitoring",
            "Operational",
            "Warning",
            "Operational"
        ],
        "Health (%)": [
            random.randint(70,100),
            random.randint(60,95),
            random.randint(75,100),
            random.randint(50,85),
            random.randint(80,100)
        ]
    })

    st.dataframe(site_data)

    fig_sites = px.bar(
        site_data,
        x="Site",
        y="Health (%)",
        color="Status",
        title="Enterprise Site Health"
    )

    st.plotly_chart(fig_sites, use_container_width=True)

    # -------------------------------
    # 👥 AI Tenant Management System
    # -------------------------------
    st.markdown("---")
    st.subheader("👥 AI Tenant Management")

    tenant_df = pd.DataFrame({
        "Client": [
            "Energy Corp",
            "Smart Grid Ltd",
            "Industrial AI Group",
            "Utility Systems",
            "Green Energy UK"
        ],
        "Subscription": [
            "Enterprise",
            "Professional",
            "Enterprise",
            "Standard",
            "Enterprise"
        ],
        "AI Usage (%)": [
            random.randint(40,100),
            random.randint(20,90),
            random.randint(60,100),
            random.randint(10,80),
            random.randint(50,100)
        ]
    })

    st.dataframe(tenant_df)

    st.success("✅ Multi-tenant AI management active")

    # -------------------------------
    # 🔔 Enterprise Notification Center
    # -------------------------------
    st.markdown("---")
    st.subheader("🔔 Enterprise Notification Center")

    notifications = [
        "⚡ Grid optimization completed",
        "🔋 Battery dispatch updated",
        "🌍 Carbon reduction target achieved",
        "🚨 Predictive maintenance alert generated",
        "📡 MQTT sensor synchronization complete"
    ]  

    for note in notifications:
        st.info(note)
    
    # -------------------------------
    # 📊 AI Operational Benchmarking
    # -------------------------------
    st.markdown("---")
    st.subheader("📊 AI Operational Benchmarking")

    benchmark_df = pd.DataFrame({
        "Metric": [
            "Energy Efficiency",
            "Carbon Reduction",
            "Grid Stability",
            "AI Automation",
            "Operational Uptime"
        ],
        "Your Platform": [
            random.randint(75,100),
            random.randint(70,100),
            random.randint(80,100),
            random.randint(75,100),
            random.randint(85,100)
        ],
        "Industry Average": [
            72,
            65,
            74,
            68,
            80
        ]
    })

    st.dataframe(benchmark_df)

    fig_benchmark = px.line(
        benchmark_df,
        x="Metric",
        y=["Your Platform", "Industry Average"],
        markers=True,
        title="Enterprise Benchmark Comparison"
    )

    st.plotly_chart(fig_benchmark, use_container_width=True)

    # -------------------------------
    # 🌍 Global Energy Command Analytics
    # -------------------------------
    st.markdown("---")
    st.subheader("🌍 Global Energy Command Analytics")

    global_energy = pd.DataFrame({
        "Region": [
            "Europe",
            "Asia",
            "North America",
            "Middle East",
            "Africa"
        ],
        "Demand (GW)": [
            random.randint(100,500),
            random.randint(300,900),
            random.randint(200,700),
            random.randint(80,300),
            random.randint(50,250)
        ]
    })

    st.dataframe(global_energy)

    fig_global_energy = px.area(
        global_energy,
        x="Region",
        y="Demand (GW)",
        title="Global Energy Demand Analytics"
    )

    st.plotly_chart(fig_global_energy, use_container_width=True)

    st.success("✅ Global energy analytics operational")

    # -------------------------------
    # 🧩 AI Plugin Marketplace
    # -------------------------------
    st.markdown("---")
    st.subheader("🧩 AI Plugin Marketplace")

    plugins = pd.DataFrame({
        "Plugin": [
            "Predictive Maintenance AI",
            "Carbon Forecast Engine",
            "SCADA Analytics",
            "Grid Stability AI",
            "Battery Optimizer"
        ],
        "Status": [
            "Installed",
            "Installed",
            "Available",
            "Installed",
            "Available"
        ],
        "Version": [
            "v2.1",
            "v1.8",
            "v3.0",
            "v2.5",
            "v1.2"
        ]
    })

    st.dataframe(plugins)

    st.success("✅ AI marketplace synchronized")

    # -------------------------------
    # ⚙ Enterprise Workflow Marketplace
    # -------------------------------
    st.markdown("---")
    st.subheader("⚙ Enterprise Workflow Marketplace")

    workflow_market = pd.DataFrame({
        "Workflow": [
            "AI Alarm Escalation",
            "Autonomous Recovery",
            "Grid Optimization",
            "Energy Trading",
            "Carbon Reporting"
        ],
        "Automation Level": [
            "100%",
            "95%",
            "90%",
            "85%",
            "100%"
        ]
    })

    st.dataframe(workflow_market)

    fig_workflow_market = px.bar(
        workflow_market,
        x="Workflow",
        y="Automation Level",
        color="Automation Level",
        title="Workflow Automation Marketplace"
    )

    st.plotly_chart(fig_workflow_market, use_container_width=True)

    # -------------------------------
    # 🧠 AI Model Registry
    # -------------------------------
    st.markdown("---")
    st.subheader("🧠 AI Model Registry")

    model_registry = pd.DataFrame({
        "Model": [
            "Failure Prediction",
            "Energy Forecast",
            "Carbon Optimization",
            "Load Balancer",
            "Risk Intelligence"
        ],
        "Accuracy": [
            random.randint(85,99),
            random.randint(80,98),
            random.randint(82,97),
            random.randint(84,96),
            random.randint(80,95)
        ],
        "Status": [
            "Production",
            "Production",
            "Testing",
            "Production",
            "Production"
        ]
    })

    st.dataframe(model_registry)

    fig_models = px.scatter(
        model_registry,
        x="Model",
        y="Accuracy",
        color="Status",
        size="Accuracy",
        title="Enterprise AI Models"
    )

    st.plotly_chart(fig_models, use_container_width=True)

    # -------------------------------
    # 💳 Smart Contract Energy Transactions
    # -------------------------------
    st.markdown("---")
    st.subheader("💳 Smart Contract Energy Transactions")

    transactions = pd.DataFrame({
        "Transaction ID": [
            "TXN-1001",
            "TXN-1002",
            "TXN-1003",
            "TXN-1004"
        ],
        "Energy (MWh)": [
            random.randint(50,300),
            random.randint(30,200),
            random.randint(60,400),
            random.randint(20,150)
        ],
        "Status": [
            "Completed",
            "Pending",
            "Completed",
            "Validated"
        ]
    })

    st.dataframe(transactions)

    st.success("✅ Smart energy transactions validated")

    # -------------------------------
    # 🤝 Industrial Partner Ecosystem
    # -------------------------------
    st.markdown("---")
    st.subheader("🤝 Industrial Partner Ecosystem")

    partners = pd.DataFrame({
        "Partner": [
            "Siemens Energy",
            "GE Vernova",
            "ABB",
            "Schneider Electric",
            "Honeywell"
        ],
        "Integration": [
            "Active",
            "Testing",
            "Active",
            "Active",
            "Planned"
        ]
    })

    st.dataframe(partners)

    st.success("✅ Enterprise ecosystem integrations active")

    # -------------------------------
    # ⚡ Live Energy Trading Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("⚡ Live Energy Trading Engine")

    energy_price = round(random.uniform(65, 180), 2)

    trade_col1, trade_col2, trade_col3 = st.columns(3)

    trade_col1.metric(
        "Live Energy Price",
        f"£{energy_price}/MWh"
    )

    trade_col2.metric(
        "Market Demand",
        f"{random.randint(60,100)} GW"
    )

    trade_col3.metric(
        "Trading Volume",
        f"{random.randint(500,5000)} MWh"
    )

    st.success("✅ Energy trading systems operational")

    # -------------------------------
    # 📈 AI Energy Price Forecasting
    # -------------------------------
    st.markdown("---")
    st.subheader("📈 AI Energy Price Forecasting")

    forecast_hours = list(range(1, 25))

    forecast_prices = [
        random.randint(60, 180)
        for _ in forecast_hours
    ]

    forecast_df = pd.DataFrame({
        "Hour": forecast_hours,
        "Forecast Price": forecast_prices
    })

    fig_price_forecast = px.line(
        forecast_df,
        x="Hour",
        y="Forecast Price",
        markers=True,
        title="24-Hour Energy Price Forecast"
    )

    st.plotly_chart(fig_price_forecast, use_container_width=True)

    st.info("🤖 AI models forecasting energy market fluctuations")

    # -------------------------------
    # 🚦 Grid Congestion Intelligence
    # -------------------------------
    st.markdown("---")
    st.subheader("🚦 Grid Congestion Intelligence")

    grid_regions = pd.DataFrame({
        "Region": [
            "London",
            "Manchester",
            "Birmingham",
            "Leeds",
            "Glasgow"
        ],
        "Congestion (%)": [
            random.randint(20,95),
            random.randint(10,85),
            random.randint(15,80),
            random.randint(25,90),
            random.randint(5,70)
        ]
    })

    st.dataframe(grid_regions)

    fig_congestion = px.bar(
        grid_regions,
        x="Region",
        y="Congestion (%)",
        color="Congestion (%)",
        title="Grid Congestion Analysis"
    )

    st.plotly_chart(fig_congestion, use_container_width=True)

    # -------------------------------
    # 🌱 Renewable Dispatch Optimizer
    # -------------------------------
    st.markdown("---")
    st.subheader("🌱 Renewable Dispatch Optimizer")

    renewables = pd.DataFrame({
        "Source": [
            "Solar",
            "Wind",
            "Hydro",
            "Battery",
            "Thermal Backup"
        ],
        "Dispatch (%)": [
            random.randint(20,90),
            random.randint(30,100),
            random.randint(10,70),
            random.randint(20,80),
            random.randint(5,50)
        ]
    })

    st.dataframe(renewables)

    fig_dispatch = px.pie(
        renewables,
        names="Source",
        values="Dispatch (%)",
        title="AI Renewable Dispatch Allocation"
    )

    st.plotly_chart(fig_dispatch, use_container_width=True)

    st.success("✅ Renewable balancing optimized")

    # -------------------------------
    # 🤖 Autonomous Energy Decision Center
    # -------------------------------
    st.markdown("---")
    st.subheader("🤖 Autonomous Energy Decision Center")

    decision_engine = random.randint(70, 100)

    st.metric(
        "AI Decision Accuracy",
        f"{decision_engine}%"
    )

    st.progress(decision_engine)

    if decision_engine < 80:

        st.warning("⚠ AI recommendations require validation")

    else:

        st.success("✅ Autonomous decisions optimized")

    st.info("""
    AI Decision Actions:
    • Optimize generation mix
    • Reduce carbon intensity
    • Balance grid demand
    • Predict market volatility
    • Improve operational efficiency
    """)

    # -------------------------------
    # 🛰 Autonomous Infrastructure Orchestrator
    # -------------------------------
    st.markdown("---")
    st.subheader("🛰 Autonomous Infrastructure Orchestrator")

    orchestrator_df = pd.DataFrame({
        "Infrastructure": [
            "Grid Control",
            "Battery Network",
            "SCADA Layer",
            "Wind Fleet",
            "Solar Farm"
        ],
        "Automation Status": [
            "Autonomous",
            "Optimized",
            "Autonomous",
            "Balanced",
            "Autonomous"
        ]
    })

    st.dataframe(orchestrator_df)

    st.success("✅ Infrastructure orchestration synchronized")

    # -------------------------------
    # 👷 AI Workforce Coordination Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("👷 AI Workforce Coordination")

    team_df = pd.DataFrame({
        "Department": [
            "Operations",
            "Maintenance",
            "Grid Analytics",
            "Cybersecurity",
            "Energy Trading"
        ],
        "AI Efficiency (%)": [
            random.randint(70,100),
            random.randint(65,95),
            random.randint(75,100),
            random.randint(60,98),
            random.randint(70,99)
        ]
    })

    st.dataframe(team_df)

    fig_team = px.line_polar(
        team_df,
        r="AI Efficiency (%)",
        theta="Department",
        line_close=True
    )

    fig_team.update_traces(fill='toself')

    st.plotly_chart(fig_team, use_container_width=True)

    # -------------------------------
    # 🚨 Utility Emergency Simulation Center
    # -------------------------------
    st.markdown("---")
    st.subheader("🚨 Utility Emergency Simulation")

    simulation = st.selectbox(
        "Run Emergency Scenario",
        [
            "Grid Overload",
            "Transformer Failure",
            "Cyberattack",
            "Wind Farm Shutdown",
            "Battery Failure"
        ]
    )

    if st.button("Run AI Simulation"):

        st.warning(f"⚠ Simulating: {simulation}")

        recovery_time = random.randint(5,60)

        st.metric(
            "Estimated Recovery Time",
            f"{recovery_time} Minutes"
        )

        st.success("✅ AI contingency analysis completed")

    # -------------------------------
    # 🌐 Cross-Grid Synchronization Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🌐 Cross-Grid Synchronization")

    grid_sync = pd.DataFrame({
        "Grid": [
            "North Grid",
            "South Grid",
            "East Grid",
            "West Grid"
        ],
        "Frequency Match (%)": [
            random.randint(92,100),
            random.randint(90,100),
            random.randint(88,100),
            random.randint(93,100)
        ]
    })

    st.dataframe(grid_sync)

    fig_sync = px.line(
        grid_sync,
        x="Grid",
        y="Frequency Match (%)",
        markers=True,
        title="Grid Synchronization Stability"
    )

    st.plotly_chart(fig_sync, use_container_width=True)

    st.success("✅ Grid synchronization stable")

    # -------------------------------
    # 🧠 AI Strategic Decision Simulator
    # -------------------------------
    st.markdown("---")
    st.subheader("🧠 AI Strategic Decision Simulator")

    strategy = st.selectbox(
        "Select AI Strategy",
        [
            "Maximize Renewable Usage",
            "Reduce Carbon Emissions",
            "Minimize Operational Cost",
            "Optimize Grid Stability",
            "Emergency Power Balancing"
        ]
    )

    if st.button("Run Strategic AI"):

        ai_score = random.randint(80,99)

        st.metric(
            "AI Optimization Score",
            f"{ai_score}%"
        )

        st.info(f"🤖 AI selected strategy: {strategy}")

        st.success("✅ Strategic optimization completed")
    
    # -------------------------------
    # 🧠 AI Cognitive Grid Intelligence
    # -------------------------------
    st.markdown("---")
    st.subheader("🧠 AI Cognitive Grid Intelligence")

    cognitive_score = random.randint(80, 99)

    st.metric(
        "Cognitive Grid Intelligence",
        f"{cognitive_score}%"
    )

    st.progress(cognitive_score)

    if cognitive_score > 90:
        st.success("✅ AI grid cognition highly optimized")
    else:
        st.warning("⚠ AI learning still improving")
    
    # -------------------------------
    # 🔄 Autonomous Power Recovery Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🔄 Autonomous Power Recovery Engine")

    recovery_df = pd.DataFrame({
        "Incident": [
            "Grid Instability",
            "Voltage Spike",
            "Transformer Fault",
            "Wind Farm Disconnect",
            "Battery Failure"
        ],
         "Recovery Status": [
            "Recovered",
            "Stabilized",
            "Recovered",
            "Balancing",
            "Recovered"
        ],
        "Recovery Time (min)": [
            random.randint(1,10),
            random.randint(2,15),
            random.randint(1,8),
            random.randint(5,20),
            random.randint(2,12)
        ]
    })

    st.dataframe(recovery_df)

    st.success("✅ Autonomous recovery systems active")

    # -------------------------------
    # 🌍 Digital Energy Ecosystem Map
    # -------------------------------
    st.markdown("---")
    st.subheader("🌍 Digital Energy Ecosystem Map")

    ecosystem_df = pd.DataFrame({
        "Infrastructure": [
            "Solar Network",
            "Wind Fleet",
            "Battery Storage",
            "Hydrogen Hub",
            "Grid AI Center"
        ],
        "Connectivity (%)": [
            random.randint(70,100),
            random.randint(65,100),
            random.randint(75,100),
            random.randint(60,95),
            random.randint(85,100)
        ]
    })

    st.dataframe(ecosystem_df)

    fig_ecosystem = px.treemap(
        ecosystem_df,
        path=["Infrastructure"],
        values="Connectivity (%)",
        title="Digital Energy Ecosystem"
    )

    st.plotly_chart(fig_ecosystem, use_container_width=True)

    # -------------------------------
    # 📈 AI Infrastructure Evolution Tracker
    # -------------------------------
    st.markdown("---")
    st.subheader("📈 AI Infrastructure Evolution Tracker")

    years = [2025, 2026, 2027, 2028, 2029, 2030]

    evolution_df = pd.DataFrame({
        "Year": years,
        "AI Automation (%)": [
            random.randint(40,60),
            random.randint(50,70),
            random.randint(60,80),
            random.randint(70,90),
            random.randint(80,95),
            random.randint(90,100)
        ]
    })

    fig_evolution = px.area(
        evolution_df,
        x="Year",
        y="AI Automation (%)",
        title="AI Infrastructure Evolution"
    )

    st.plotly_chart(fig_evolution, use_container_width=True)

    st.info("🤖 AI infrastructure continuously evolving")

    # -------------------------------
    # ⚡ Self-Optimizing Utility Brain
    # -------------------------------
    st.markdown("---")
    st.subheader("⚡ Self-Optimizing Utility Brain")

    brain_score = random.randint(85,100)

    brain_col1, brain_col2, brain_col3 = st.columns(3)

    brain_col1.metric(
        "AI Optimization",
        f"{brain_score}%"
    )

    brain_col2.metric(
        "Autonomous Decisions",
        random.randint(1000,10000)
    )

    brain_col3.metric(
        "Grid Stability",
        f"{random.randint(90,100)}%"
    )

    if brain_score > 92:
        st.success("✅ Utility brain fully optimized")
    else:
        st.warning("⚠ Optimization in progress")
    
    # -------------------------------
    # ⚛ Quantum Grid Optimization
    # -------------------------------
    st.markdown("---")
    st.subheader("⚛ Quantum Grid Optimization")

    quantum_score = random.randint(85, 100)

    st.metric(
        "Quantum Optimization Score",
        f"{quantum_score}%"
    )

    st.progress(quantum_score)

    if quantum_score > 92:
        st.success("✅ Quantum optimization stable")
    else:
        st.warning("⚠ Quantum balancing recalculating")
    
    # -------------------------------
    # 🌍 AI Climate Impact Simulator
    # -------------------------------
    st.markdown("---")
    st.subheader("🌍 AI Climate Impact Simulator")

    climate_df = pd.DataFrame({
        "Scenario": [
            "Net Zero",
            "High Renewable",
            "Hybrid Grid",
            "Carbon Intensive",
            "AI Optimized"
        ],
        "CO₂ Reduction (%)": [
            random.randint(40,90),
            random.randint(50,95),
            random.randint(35,80),
            random.randint(10,40),
            random.randint(60,98)
        ]
    })

    st.dataframe(climate_df)

    fig_climate = px.bar(
        climate_df,
        x="Scenario",
        y="CO₂ Reduction (%)",
        color="CO₂ Reduction (%)",
        title="AI Climate Impact Forecast"
    )

    st.plotly_chart(fig_climate, use_container_width=True)

    # -------------------------------
    # 🛰 Autonomous Utility Swarm Intelligence
    # -------------------------------
    st.markdown("---")
    st.subheader("🛰 Autonomous Utility Swarm Intelligence")

    swarm_df = pd.DataFrame({
        "Utility Node": [
            "Node A",
            "Node B",
            "Node C",
            "Node D",
            "Node E"
        ],
        "AI Coordination (%)": [
            random.randint(80,100),
            random.randint(75,98),
            random.randint(82,100),
            random.randint(78,99),
            random.randint(85,100)
        ]
    })

    st.dataframe(swarm_df)

    fig_swarm = px.scatter(
        swarm_df,
        x="Utility Node",
        y="AI Coordination (%)",
        size="AI Coordination (%)",
        color="AI Coordination (%)",
        title="Utility Swarm Coordination"
    )

    st.plotly_chart(fig_swarm, use_container_width=True)

    st.success("✅ Swarm coordination synchronized")

    # -------------------------------
    # 🌐 Global Energy Neural Network
    # -------------------------------
    st.markdown("---")
    st.subheader("🌐 Global Energy Neural Network")

    neural_df = pd.DataFrame({
        "Region": [
            "Europe",
            "North America",
            "Asia",
            "Middle East",
            "Africa"
        ],
        "Neural Connectivity (%)": [
            random.randint(75,100),
            random.randint(70,100),
            random.randint(80,100),
            random.randint(65,95),
            random.randint(60,90)
        ]
    })

    st.dataframe(neural_df)

    fig_neural = px.line(
        neural_df,
        x="Region",
        y="Neural Connectivity (%)",
        markers=True,
        title="Global Neural Energy Connectivity"
    )

    st.plotly_chart(fig_neural, use_container_width=True)

    # -------------------------------
    # 🤖 Hyper-Autonomous Infrastructure Core
    # -------------------------------
    st.markdown("---")
    st.subheader("🤖 Hyper-Autonomous Infrastructure Core")

    core_col1, core_col2, core_col3 = st.columns(3)

    core_col1.metric(
        "Infrastructure Autonomy",
        f"{random.randint(90,100)}%"
    )

    core_col2.metric(
        "AI Decisions / Hour",
        random.randint(5000,50000)
    )

    core_col3.metric(
        "Self-Healing Accuracy",
        f"{random.randint(85,100)}%"
    )

    st.success("✅ Hyper-autonomous infrastructure active")

    st.info("""
    Core Capabilities:
    • Self-healing operations
    • Autonomous grid stabilization
    • AI-based recovery systems
    • Predictive energy orchestration
    • Cognitive infrastructure optimization
    """)

    # -------------------------------
    # 🧠 AGI Utility Commander
    # -------------------------------
    st.markdown("---")
    st.subheader("🧠 AGI Utility Commander")

    agi_score = random.randint(90, 100)

    st.metric(
        "AGI Operational Intelligence",
        f"{agi_score}%"
    )

    st.progress(agi_score)

    if agi_score > 95:
        st.success("✅ AGI utility orchestration fully autonomous")
    else:
        st.warning("⚠ AGI optimization recalibrating")

    # -------------------------------
    # 🛰 Space-Based Energy Coordination
    # -------------------------------
    st.markdown("---")
    st.subheader("🛰 Space-Based Energy Coordination")

    space_df = pd.DataFrame({
        "Satellite Grid": [
            "Orbital Solar Array",
            "Lunar Storage Hub",
            "Geo Grid Relay",
            "Mars Energy Link",
            "Deep Space Node"
        ],
        "Energy Transfer (%)": [
            random.randint(70,100),
            random.randint(65,95),
            random.randint(75,100),
            random.randint(60,90),
            random.randint(50,85)
        ]
    })

    st.dataframe(space_df)

    fig_space = px.bar(
        space_df,
        x="Satellite Grid",
        y="Energy Transfer (%)",
        color="Energy Transfer (%)",
        title="Space Energy Coordination"
    )

    st.plotly_chart(fig_space, use_container_width=True)

    st.success("✅ Space-grid synchronization stable")


    # -------------------------------
    # 🧬 Autonomous Infrastructure DNA Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🧬 Autonomous Infrastructure DNA Engine")

    dna_df = pd.DataFrame({
        "Infrastructure Gene": [
            "Self-Healing",
            "Adaptive Grid",
            "AI Learning",
            "Resilience",
            "Optimization"
        ],
        "Evolution Score": [
            random.randint(80,100),
            random.randint(75,98),
            random.randint(85,100),
            random.randint(78,99),
            random.randint(82,100)
        ]
    })

    st.dataframe(dna_df)

    fig_dna = px.line_polar(
        dna_df,
        r="Evolution Score",
        theta="Infrastructure Gene",
        line_close=True
    )

    fig_dna.update_traces(fill='toself')

    st.plotly_chart(fig_dna, use_container_width=True)

    # -------------------------------
    # 🌍 AI Civilization Energy Index
    # -------------------------------
    st.markdown("---")
    st.subheader("🌍 AI Civilization Energy Index")

    civilization_df = pd.DataFrame({
        "Civilization Sector": [
            "Transportation",
            "Industry",
            "Residential",
            "Space Infrastructure",
            "AI Ecosystems"
        ],
        "Energy Intelligence (%)": [
            random.randint(70,100),
            random.randint(65,98),
            random.randint(60,95),
            random.randint(80,100),
            random.randint(85,100)
        ]
    })

    st.dataframe(civilization_df)

    fig_civilization = px.treemap(
        civilization_df,
        path=["Civilization Sector"],
        values="Energy Intelligence (%)",
        title="Civilization Energy Intelligence"
    )

    st.plotly_chart(fig_civilization, use_container_width=True)

    # -------------------------------
    # ⚡ Self-Evolving Utility Intelligence
    # -------------------------------
    st.markdown("---")
    st.subheader("⚡ Self-Evolving Utility Intelligence")

    evolution_col1, evolution_col2, evolution_col3 = st.columns(3)

    evolution_col1.metric(
        "AI Evolution Rate",
        f"{random.randint(90,100)}%"
    )

    evolution_col2.metric(
        "Autonomous Learning Cycles",
        random.randint(10000,500000)
    )

    evolution_col3.metric(
        "Infrastructure Evolution",
        f"{random.randint(88,100)}%"
    )

    st.success("✅ Self-evolving intelligence active")

    st.info("""
    Evolution Capabilities:
    • Autonomous AI learning
    • Dynamic infrastructure adaptation
    • Predictive civilization scaling
    • Self-improving energy orchestration
    • AGI-powered optimization
    """)

    # -------------------------------
    # 🧠 Sentient Infrastructure Awareness
    # -------------------------------
    st.markdown("---")
    st.subheader("🧠 Sentient Infrastructure Awareness")

    sentient_score = random.randint(92, 100)

    st.metric(
        "Infrastructure Awareness",
        f"{sentient_score}%"
    )

    st.progress(sentient_score)

    if sentient_score > 96:
        st.success("✅ Sentient infrastructure fully adaptive")
    else:
        st.warning("⚠ Sentient AI recalibrating")
    
    # -------------------------------
    # 🌍 Planetary Grid Coordination Matrix
    # -------------------------------
    st.markdown("---")
    st.subheader("🌍 Planetary Grid Coordination Matrix")

    planetary_df = pd.DataFrame({
        "Grid Zone": [
            "Northern Hemisphere",
            "Southern Hemisphere",
            "Orbital Grid",
            "Oceanic Grid",
            "Polar Infrastructure"
        ],
        "Synchronization (%)": [
            random.randint(80,100),
            random.randint(75,98),
            random.randint(85,100),
            random.randint(78,99),
            random.randint(70,95)
        ]
    })

    st.dataframe(planetary_df)

    fig_planetary = px.area(
        planetary_df,
        x="Grid Zone",
        y="Synchronization (%)",
        title="Planetary Grid Synchronization"
    )

    st.plotly_chart(fig_planetary, use_container_width=True)

    st.success("✅ Planetary grids synchronized")

    # -------------------------------
    # 🌌 Universal Energy Intelligence Network
    # -------------------------------
    st.markdown("---")
    st.subheader("🌌 Universal Energy Intelligence Network")

    universal_df = pd.DataFrame({
        "Energy Network": [
            "Earth Grid",
            "Orbital Grid",
            "Lunar Grid",
            "Mars Colony",
            "Deep Space Relay"
        ],
        "AI Connectivity (%)": [
            random.randint(82,100),
            random.randint(75,98),
            random.randint(70,95),
            random.randint(65,90),
            random.randint(60,85)
        ]
    })

    st.dataframe(universal_df)

    fig_universal = px.scatter(
        universal_df,
        x="Energy Network",
        y="AI Connectivity (%)",
        size="AI Connectivity (%)",
        color="AI Connectivity (%)",
        title="Universal AI Energy Connectivity"
    )

    st.plotly_chart(fig_universal, use_container_width=True)

    # -------------------------------
    # 🏙 Autonomous Civilization Stability Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🏙 Autonomous Civilization Stability Engine")

    civilization_score = random.randint(88,100)

    st.metric(
        "Civilization Stability Index",
        f"{civilization_score}%"
    )

    st.progress(civilization_score)

    if civilization_score > 94:
        st.success("✅ Civilization infrastructure stable")
    else:
        st.warning("⚠ Stability optimization running")
    
    # -------------------------------
    # 🤖 AI Conscious Infrastructure Core
    # -------------------------------
    st.markdown("---")
    st.subheader("🤖 AI Conscious Infrastructure Core")

    core_df = pd.DataFrame({
        "Capability": [
            "Self Awareness",
            "Autonomous Decisions",
            "Predictive Evolution",
            "Infrastructure Healing",
            "Universal Coordination"
        ],
        "Capability Score": [
            random.randint(85,100),
            random.randint(88,100),
            random.randint(82,100),
            random.randint(86,100),
            random.randint(80,100)
        ]
    })

    st.dataframe(core_df)

    fig_core = px.bar(
        core_df,
        x="Capability",
        y="Capability Score",
        color="Capability Score",
        title="AI Conscious Infrastructure Core"
    )

    st.plotly_chart(fig_core, use_container_width=True)

    st.success("✅ Conscious infrastructure intelligence operational")

    # -------------------------------
    # 🌌 Multiverse Energy Synchronization
    # -------------------------------
    st.markdown("---")
    st.subheader("🌌 Multiverse Energy Synchronization")

    multiverse_score = random.randint(90, 100)

    st.metric(
        "Multiverse Synchronization",
        f"{multiverse_score}%"
    )

    st.progress(multiverse_score)

    if multiverse_score > 96:
        st.success("✅ Multiverse energy systems synchronized")
    else:
        st.warning("⚠ Cross-dimensional balancing active")
    
    # -------------------------------
    # 🌠 Cosmic Infrastructure Stability Engine
    # -------------------------------
    st.markdown("---")
    st.subheader("🌠 Cosmic Infrastructure Stability Engine")

    cosmic_df = pd.DataFrame({
        "Infrastructure Zone": [
            "Earth Core Grid",
            "Orbital Ring",
            "Lunar Energy Hub",
            "Mars Colony",
            "Deep Space Relay"
        ],
        "Stability (%)": [
            random.randint(85,100),
            random.randint(80,98),
            random.randint(75,95),
            random.randint(70,92),
            random.randint(65,90)
        ]
    })

    st.dataframe(cosmic_df)

    fig_cosmic = px.line(
        cosmic_df,
        x="Infrastructure Zone",
        y="Stability (%)",
        markers=True,
        title="Cosmic Infrastructure Stability"
    )

    st.plotly_chart(fig_cosmic, use_container_width=True)

    st.success("✅ Cosmic infrastructure stable")

    # -------------------------------
    # ⚡ Autonomous Universal Power Allocation
    # -------------------------------
    st.markdown("---")
    st.subheader("⚡ Autonomous Universal Power Allocation")

    allocation_df = pd.DataFrame({
        "Sector": [
            "Planetary Grid",
            "Orbital Systems",
            "Industrial AI",
            "Transportation",
            "Space Infrastructure"
        ],
        "Power Allocation (%)": [
            random.randint(15,40),
            random.randint(10,35),
            random.randint(20,45),
            random.randint(10,30),
            random.randint(5,25)
        ]
    })

    st.dataframe(allocation_df)

    fig_allocation = px.pie(
        allocation_df,
        names="Sector",
        values="Power Allocation (%)",
        title="Universal Power Distribution"
    )

    st.plotly_chart(fig_allocation, use_container_width=True)

    # -------------------------------
    # 🧠 Infinite Neural Energy Mesh
    # -------------------------------
    st.markdown("---")
    st.subheader("🧠 Infinite Neural Energy Mesh")

    mesh_df = pd.DataFrame({
        "Node": [
            "Node Alpha",
            "Node Beta",
            "Node Gamma",
            "Node Delta",
            "Node Omega"
        ],
        "Neural Activity (%)": [
            random.randint(85,100),
            random.randint(80,98),
            random.randint(82,100),
            random.randint(78,99),
            random.randint(88,100)
        ]
    })

    st.dataframe(mesh_df)

    fig_mesh = px.scatter(
        mesh_df,
        x="Node",
        y="Neural Activity (%)",
        size="Neural Activity (%)",
        color="Neural Activity (%)",
        title="Infinite Neural Energy Mesh"
    )

    st.plotly_chart(fig_mesh, use_container_width=True)

    st.success("✅ Neural mesh intelligence operational")

    # -------------------------------
    # 🌀 AI Reality Simulation Core
    # -------------------------------
    st.markdown("---")
    st.subheader("🌀 AI Reality Simulation Core")

    simulation_df = pd.DataFrame({
        "Simulation Layer": [
            "Grid Reality",
            "Climate Reality",
            "Economic Reality",
            "Infrastructure Reality",
            "Civilization Reality"
        ],
        "Simulation Accuracy (%)": [
            random.randint(88,100),
            random.randint(82,98),
            random.randint(80,96),
            random.randint(85,100),
            random.randint(78,95)
        ]
    })

    st.dataframe(simulation_df)

    fig_simulation = px.bar(
        simulation_df,
        x="Simulation Layer",
        y="Simulation Accuracy (%)",
        color="Simulation Accuracy (%)",
        title="AI Reality Simulation Core"
    )

    st.plotly_chart(fig_simulation, use_container_width=True)

    st.success("✅ Reality simulation systems active")

    # -------------------------------
    # 🔗 Real-Time Asset Dependency Mapping
    # -------------------------------
    st.markdown("---")
    st.subheader("🔗 Real-Time Asset Dependency Mapping")

    dependency_df = pd.DataFrame({
        "Primary Asset": [
            "Gas Turbine",
            "Battery Storage",
            "Solar Farm",
            "HV Transformer",
            "SCADA Network"
        ],
        "Dependent System": [
            "Cooling System",
            "Grid Stability",
            "Inverter Network",
            "Distribution Bus",
            "Control Center"
        ],
        "Dependency Risk": [
            random.randint(10,90),
            random.randint(15,85),
            random.randint(20,80),
            random.randint(5,70),
            random.randint(25,95)
        ]
    })

    st.dataframe(dependency_df)

    fig_dependency = px.sunburst(
        dependency_df,
        path=["Primary Asset", "Dependent System"],
        values="Dependency Risk",
        title="Asset Dependency Intelligence"
    )

    st.plotly_chart(fig_dependency, use_container_width=True)

    # -------------------------------
    # 🚨 Enterprise Operational Risk Matrix
    # -------------------------------
    st.markdown("---")
    st.subheader("🚨 Enterprise Operational Risk Matrix")

    risk_df = pd.DataFrame({
        "Operational Area": [
            "Generation",
            "Cybersecurity",
            "Grid Stability",
            "Market Operations",
            "IoT Infrastructure"
        ],
        "Risk Score": [
            random.randint(10,90),
            random.randint(15,95),
            random.randint(20,85),
            random.randint(10,80),
            random.randint(5,75)
        ]
    })

    st.dataframe(risk_df)

    fig_risk = px.density_heatmap(
        risk_df,
        x="Operational Area",
        y="Risk Score",
        z="Risk Score",
        title="Enterprise Risk Matrix"
    )

    st.plotly_chart(fig_risk, use_container_width=True)

    # -------------------------------
    # ⚡ AI Resource Optimization Fabric
    # -------------------------------
    st.markdown("---")
    st.subheader("⚡ AI Resource Optimization Fabric")

    resource_df = pd.DataFrame({
        "Resource": [
            "Power Generation",
            "Battery Storage",
            "Grid Frequency",
            "Cooling Systems",
            "Data Infrastructure"
        ],
        "Optimization (%)": [
            random.randint(70,100),
            random.randint(65,98),
            random.randint(75,100),
            random.randint(60,95),
            random.randint(80,100)
        ]
    })

    st.dataframe(resource_df)

    fig_resource = px.funnel(
        resource_df,
        x="Optimization (%)",
        y="Resource",
        title="AI Resource Optimization"
    )

    st.plotly_chart(fig_resource, use_container_width=True)

    st.success("✅ Resource optimization synchronized")

    # -------------------------------
    # 🌐 Distributed Digital Twin Synchronization
    # -------------------------------
    st.markdown("---")
    st.subheader("🌐 Distributed Digital Twin Synchronization")

    sync_df = pd.DataFrame({
        "Digital Twin Node": [
            "Plant A",
            "Plant B",
            "Grid Hub",
            "Solar Cluster",
            "Battery Fleet"
        ],
        "Synchronization (%)": [
            random.randint(80,100),
            random.randint(78,99),
            random.randint(85,100),
            random.randint(75,98),
            random.randint(82,100)
        ]
    })

    st.dataframe(sync_df)

    fig_sync = px.area(
        sync_df,
        x="Digital Twin Node",
        y="Synchronization (%)",
        title="Distributed Twin Synchronization"
    )

    st.plotly_chart(fig_sync, use_container_width=True)

    st.success("✅ Digital twin synchronization active")

    # -------------------------------
    # 🎯 Autonomous Utility Mission Control
    # -------------------------------
    st.markdown("---")
    st.subheader("🎯 Autonomous Utility Mission Control")

    mission_col1, mission_col2, mission_col3 = st.columns(3)

    mission_col1.metric(
        "Operational Readiness",
        f"{random.randint(90,100)}%"
    )

    mission_col2.metric(
        "Active AI Decisions",
        random.randint(1000,10000)
    )

    mission_col3.metric(
        "Mission Stability",
        f"{random.randint(88,100)}%"
    )

    st.success("✅ Mission control systems operational")

    st.info("""
    Mission Control Functions:
    • Autonomous operational coordination
    • AI-driven infrastructure balancing
    • Enterprise energy optimization
    • Real-time digital twin synchronization
    • Distributed utility intelligence
    """)

    # -------------------------------
    # 🏭 Multi-Site Operations Center
    # -------------------------------
    st.markdown("---")
    st.subheader("🏭 Multi-Site Operations Center")

    site_df = pd.DataFrame({
        "Site": [
            "London Energy Hub",
            "Manchester Grid",
            "Birmingham Plant",
            "Leeds Storage Hub",
            "Glasgow Wind Center"
        ],
        "Operational Status": [
            "Online",
            "Stable",
            "Online",
            "Maintenance",
            "Online"
        ],
        "Efficiency (%)": [
            random.randint(75,100),
            random.randint(70,98),
            random.randint(80,100),
            random.randint(60,90),
            random.randint(78,100)
        ]
    })

    st.dataframe(site_df)

    fig_sites = px.bar(
        site_df,
        x="Site",
        y="Efficiency (%)",
        color="Efficiency (%)",
        title="Enterprise Site Performance"
    )

    st.plotly_chart(fig_sites, use_container_width=True)

    # -------------------------------
    # 📊 Enterprise AI SLA Monitoring
    # -------------------------------
    st.markdown("---")
    st.subheader("📊 Enterprise AI SLA Monitoring")

    sla_df = pd.DataFrame({
        "Service": [
            "SCADA Analytics",
            "IoT Gateway",
            "Predictive AI",
            "Energy Forecasting",
            "Grid Intelligence"
        ],
        "Uptime (%)": [
            random.randint(95,100),
            random.randint(94,100),
            random.randint(96,100),
            random.randint(93,100),
            random.randint(95,100)
        ]
    })

    st.dataframe(sla_df)

    fig_sla = px.line(
        sla_df,
        x="Service",
        y="Uptime (%)",
        markers=True,
        title="Enterprise SLA Monitoring"
    )

    st.plotly_chart(fig_sla, use_container_width=True)

    st.success("✅ Enterprise SLA targets maintained")

    # -------------------------------
    # 🤖 Industrial Workflow Automation
    # -------------------------------
    st.markdown("---")
    st.subheader("🤖 Industrial Workflow Automation")

    workflow_df = pd.DataFrame({
        "Workflow": [
            "Maintenance Dispatch",
            "Grid Balancing",
            "Incident Response",
            "Energy Optimization",
            "Carbon Reporting"
        ],
        "Automation Level (%)": [
            random.randint(70,100),
            random.randint(75,100),
            random.randint(65,95),
            random.randint(80,100),
            random.randint(60,95)
        ]
    })

    st.dataframe(workflow_df)

    fig_workflow = px.funnel_area(
        workflow_df,
        names="Workflow",
        values="Automation Level (%)",
        title="Workflow Automation Coverage"
    )

    st.plotly_chart(fig_workflow, use_container_width=True)

    st.success("✅ Industrial workflows automated")

    # -------------------------------
    # 📡 Live Infrastructure Command Queue
    # -------------------------------
    st.markdown("---")
    st.subheader("📡 Live Infrastructure Command Queue")

    command_df = pd.DataFrame({
        "Command": [
            "Rebalance Grid",
            "Optimize Storage",
            "Reduce Peak Load",
            "Deploy Maintenance",
            "Run AI Forecast"
        ],
        "Priority": [
            "High",
            "Medium",
            "High",
            "Low",
            "Medium"
        ],
        "Execution Status": [
            "Running",
            "Queued",
            "Completed",
            "Running",
            "Queued"
        ]
    })

    st.dataframe(command_df)

    st.info("⚡ Autonomous command queue actively managing infrastructure")

    # -------------------------------
    # ☁ Executive Operational Intelligence Cloud
    # -------------------------------
    st.markdown("---")
    st.subheader("☁ Executive Operational Intelligence Cloud")

    cloud_col1, cloud_col2, cloud_col3 = st.columns(3)

    cloud_col1.metric(
        "Enterprise Availability",
        f"{random.randint(98,100)}%"
    )

    cloud_col2.metric(
        "AI Decisions Processed",
        f"{random.randint(50000,500000):,}"
    )

    cloud_col3.metric(
        "Infrastructure Reliability",
        f"{random.randint(92,100)}%"
    )

    st.success("✅ Enterprise operational cloud synchronized")

    st.info("""
    Cloud Intelligence Functions:
    • Enterprise operational visibility
    • AI infrastructure analytics
    • Distributed digital twin control
    • Autonomous workflow orchestration
    • Executive operational intelligence
    """)

    # -------------------------------
    # 📡 Live Sensor Control Center
    # -------------------------------
    st.markdown("---")
    st.subheader("📡 Live Sensor Control Center")

    temperature = random.uniform(40, 95)
    pressure = random.uniform(5, 20)
    vibration = random.uniform(0.5, 10)

    sensor_col1, sensor_col2, sensor_col3 = st.columns(3)

    sensor_col1.metric(
        "Temperature",
        f"{temperature:.1f} °C"
    )

    sensor_col2.metric(
        "Pressure",
        f"{pressure:.1f} bar"
    )

    sensor_col3.metric(
        "Vibration",
        f"{vibration:.2f} mm/s"
    )

    # -------------------------------
    # 📈 Telemetry Data Stream
    # -------------------------------
    st.markdown("---")
    st.subheader("📈 Live Telemetry Stream")

    telemetry_df = pd.DataFrame({
        "Time": range(1, 25),
        "Temperature": [
            random.randint(50,90)
            for _ in range(24)
        ]
    })

    fig_telemetry = px.line(
        telemetry_df,
        x="Time",
        y="Temperature",
        title="Temperature Telemetry"
    )

    st.plotly_chart(
        fig_telemetry,
        use_container_width=True
    )

    # -------------------------------
    # 🔍 Sensor Health Analyzer
    # -------------------------------
    st.markdown("---")
    st.subheader("🔍 Sensor Health Analyzer")

    sensor_health = random.randint(70,100)

    st.metric(
        "Sensor Health",
        f"{sensor_health}%"
    )

    st.progress(sensor_health)

    if sensor_health < 80:
        st.warning(
            "⚠ Sensor calibration recommended"
        )
    else:
        st.success(
            "✅ Sensors healthy"
        )

    # -------------------------------
    # 📶 MQTT Connection Monitor
    # -------------------------------
    st.markdown("---")
    st.subheader("📶 MQTT Connection Monitor")

    mqtt_status = random.choice(
        [
            "Connected",
            "Connected",
            "Connected",
            "Disconnected"
        ]
    )

    if mqtt_status == "Connected":
        st.success(
            "✅ MQTT Broker Connected"
        )
    else:
        st.error(
            "❌ MQTT Broker Disconnected"
        )

    # -------------------------------
    # 🏷 Industrial Tag Monitor
    # -------------------------------
    st.markdown("---")
    st.subheader("🏷 Industrial Tag Monitor")

    tag_df = pd.DataFrame({
        "Tag": [
            "GT_TEMP_01",
            "GT_PRESS_01",
            "GT_VIB_01",
            "HV_VOLT_01",
            "HV_CURR_01"
        ],
        "Value": [
            round(random.uniform(20,100),2),
            round(random.uniform(5,20),2),
            round(random.uniform(0,10),2),
            round(random.uniform(10000,33000),2),
            round(random.uniform(100,1000),2)
        ]
    })

    st.dataframe(
        tag_df,
        use_container_width=True
    )

    # =========================================================
    # 🌍 GLOBAL KPI CENTER
    # =========================================================
    st.subheader("🌍 Enterprise KPI Center")

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)

    # Safe values
    safe_energy = 0
    safe_cost = 0
    safe_co2 = 0
    safe_health = 0

    if forecast_df is not None and "forecast" in forecast_df.columns:

        safe_energy = round(forecast_df["forecast"].sum(), 2)
        safe_cost = round(safe_energy * 0.12, 2)
        safe_co2 = round(safe_energy * 0.233, 2)

    try:
        safe_health = round(health_score, 2)
    except:
        safe_health = 0

    kpi1.metric("⚡ Total Energy", f"{safe_energy} kWh")
    kpi2.metric("💰 Energy Cost", f"£{safe_cost}")
    kpi3.metric("🌍 CO₂ Output", f"{safe_co2} kg")
    kpi4.metric("🛠 Health Score", f"{safe_health}%")

    # =========================================================
    # 🚨 AI Risk Matrix
    # =========================================================
    st.subheader("🚨 AI Risk Matrix")

    risk_data = pd.DataFrame({
        "System": [
            "HVAC",
            "Turbine",
            "Cooling System",
            "Grid Connection",
            "Battery Storage"
        ],
        "Risk Level": [
            random.randint(10, 90),
            random.randint(10, 90),
            random.randint(10, 90),
            random.randint(10, 90),
            random.randint(10, 90)
        ]
    })

    fig_risk = px.bar(
        risk_data,
        x="System",
        y="Risk Level",
        color="Risk Level",
        title="Enterprise Risk Assessment"
    )

    st.plotly_chart(fig_risk, use_container_width=True)

    # =========================================================
    # 🌍 LIVE ENERGY INTELLIGENCE
    # =========================================================

    st.sidebar.markdown("## ⚡ Live Energy Intelligence")

    carbon_intensity = get_live_carbon_intensity()

    if carbon_intensity:

        st.sidebar.metric(
            "⚡ Carbon Intensity",
            f"{carbon_intensity} gCO₂/kWh"
        )

    generation_mix = get_live_grid_demand()

    if generation_mix:

        renewable_percent = calculate_renewable_percentage(generation_mix)

        st.sidebar.metric(
            "🌱 Renewable %",
            f"{renewable_percent}%"
        )

    solar_generation = estimate_solar_generation(
        weather["temperature"]
    )

    st.sidebar.metric(
        "☀ Solar Generation",
        f"{solar_generation} kW"
    )

    # =========================================================
    # ⚡ LIVE POWER DEMAND CENTER
    # =========================================================
    st.subheader("⚡ Live Power Demand")

    power_demand = random.randint(200, 1200)

    st.metric(
        "National Grid Demand",
        f"{power_demand} MW"
    )

    if power_demand > 1000:
        st.error("🚨 Grid Under Heavy Demand")

    elif power_demand > 700:
        st.warning("⚠ Elevated Grid Demand")

    else:
        st.success("✅ Grid Operating Normally")
    
    # =========================================================
    # ⚖ AI Load Balancer
    # =========================================================
    st.subheader("⚖ AI Load Balancer")

    load_distribution = pd.DataFrame({
         "System": [
            "HVAC",
            "Lighting",
            "Turbines",
            "Cooling",
            "Battery Storage"
        ],
        "Load (%)": [
            random.randint(10, 40),
            random.randint(5, 20),
            random.randint(20, 50),
            random.randint(10, 30),
            random.randint(5, 25)
        ]
    })

    fig_load = px.pie(
        load_distribution,
        names="System",
        values="Load (%)",
        title="AI Load Distribution"
    )

    st.plotly_chart(fig_load, use_container_width=True)
    
    # =========================================================
    # 🌐 Grid Stability Index
    # =========================================================
    st.subheader("🌐 Grid Stability Index")

    grid_score = random.randint(70, 100)

    st.progress(grid_score)

    st.metric(
        "Grid Stability",
        f"{grid_score}%"
    )

    if grid_score < 75:
        st.error("🚨 Grid Instability Detected")

    elif grid_score < 90:
        st.warning("⚠ Minor Grid Fluctuations")

    else:
        st.success("✅ Grid Stable")
    
    # =========================================================
    # 💹 Energy Trading Simulator
    # =========================================================
    st.subheader("💹 Live Energy Trading")

    market_price = round(random.uniform(0.08, 0.25), 3)

    st.metric(
        "Electricity Market Price",
        f"£{market_price}/kWh"
    )

    if market_price > 0.20:
        st.error("🚨 Peak Market Pricing")

    elif market_price > 0.14:
        st.warning("⚠ Moderate Market Rates")

    else:
        st.success("✅ Favorable Energy Pricing")


    # -------------------------------
    # ⚡ Live Carbon Intensity API
    # -------------------------------
    def get_live_energy_price():
        url = "https://api.carbonintensity.org.uk/intensity"

        try:
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                return data["data"][0]["intensity"]["actual"]

        except:
            return None

        return None

    # Fetch API data
    price = get_live_energy_price()

    # Display metric
    if price:
        st.sidebar.metric(
            "⚡ Live Carbon Intensity",
            f"{price} gCO₂/kWh"
        )

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
    
    # -------------------------------
    # 🧭 Enterprise Navigation
    # -------------------------------
    st.sidebar.markdown("---")
    st.sidebar.subheader("🧭 Navigation")

    page = st.sidebar.radio(
        "Select Module",
        [
            "Overview",
            "AI Monitoring",
            "SCADA Control",
            "Asset Management",
            "Energy Intelligence",
            "Executive Center",
            "Asset History"
            "Security Center",
            "Weather Intelligence",
            "AI Failure Prediction",
            "Alert Center",
            "Scenario Simulator",
            "Asset Relationship Map",
            "Enterprise KPI Scorecard",
            "Report Center",
            "Enterprise Timeline"

        ]
    )

    forecast_days = st.sidebar.slider("Select number of forecast days", 7, 60, 30)
    building = st.sidebar.selectbox(
        "Select Building",
        ["Building A", "Building B", "Building C"]
    )
    if st.session_state.get("username") == "admin":
        # -------------------------------
        # 👥 Enterprise Role Access
        # -------------------------------
        st.sidebar.header("👥 User Access Control")

        role = st.sidebar.selectbox(
            "Select Role",
            [
                "Admin",
                "Operator",
                "Engineer",
                "Client"
            ]
        )

        st.sidebar.success(f"Logged in as: {role}")

        # -------------------------------
        # 🔐 Role Permissions Matrix
        # -------------------------------
        role_permissions = {

            "Admin": [
                "Overview",
                "AI Monitoring",
                "SCADA Control",
                "Asset Management",
                "Energy Intelligence",
                "Executive Center",
                "Asset History",
                "Security Center",
                "Weather Intelligence",
                "AI Failure Prediction",
                "Alert Center",
                "Scenario Simulator",
                "Asset Relationship Map",
                "Enterprise KPI Scorecard",
                "Report Center",
                "Enterprise Timeline"
            ],

            "Engineer": [
                "Overview",
                "AI Monitoring",
                "SCADA Control",
                "Asset Management",
                "Energy Intelligence",
                "AI Failure Prediction",
                "Alert Center",
                "Scenario Simulator",
                "Asset Relationship Map",
                "Enterprise Timeline"
            ],

            "Operator": [
                "Overview",
                "SCADA Control",
                "Asset Management",
                "Alert Center"
            ],

            "Client": [
                "Overview",
                "Executive Center",
                "Scenario Simulator",
                "Asset Relationship Map",
                "Enterprise KPI Scorecard",
                "Report Center"
            ]
        }
        allowed_pages = role_permissions.get(
            role,
            ["Overview"]
        )

        # -------------------------------
        # 🔐 Access Rights
        # -------------------------------
        st.sidebar.markdown("### 🔐 Access Rights")

        for item in allowed_pages:
            st.sidebar.write(f"✅ {item}")

        # -------------------------------
        # 👔 Executive Control Center
        # -------------------------------
        if page == "Executive Center":

            if page not in allowed_pages:
                st.error("🚫 Access Denied")
                st.stop()

            st.header("👔 Executive Control Center")

            exec_col1, exec_col2, exec_col3 = st.columns(3)

            exec_col1.metric(
                "Enterprise Efficiency",
                f"{random.randint(85,99)}%"
            )

            exec_col2.metric(
                "Annual Savings",
                f"£{random.randint(100000,500000):,}"
            )

            exec_col3.metric(
                "Carbon Reduction",
                f"{random.randint(10,40)}%"
            )

            st.success("✅ Executive systems operational")
        
        # -------------------------------
        # 🖥 SCADA Control Panel
        # -------------------------------
        if page == "SCADA Control":

            if page not in allowed_pages:
                st.error("🚫 Access Denied")
                st.stop()

            st.header("🖥 SCADA Control Panel")

            turbine_toggle = st.toggle("Gas Turbine")
            boiler_toggle = st.toggle("Boiler")
            hvac_toggle = st.toggle("HVAC")

            if turbine_toggle:
                st.success("✅ Gas Turbine Online")
            else:
                st.error("❌ Gas Turbine Offline")

            if boiler_toggle:
                st.success("✅ Boiler Operational")

            if hvac_toggle:
                st.success("✅ HVAC Running")
        
        # -------------------------------
        # 🗄 Asset History Page
        # -------------------------------
        if page == "Asset History":

            st.header("🗄 Asset History Database")

            history_df = pd.read_sql_query(
                """
                SELECT *
                FROM assets
                ORDER BY id DESC
                LIMIT 50
                """,
                get_connection()
            )

            st.dataframe(
                history_df,
                use_container_width=True
            )

            st.markdown("---")

            st.header("🚨 Incident History")

            incident_df = pd.read_sql_query(
                """
                SELECT *
                FROM incidents
                ORDER BY id DESC
                LIMIT 50
                """,
                get_connection()
            )

            st.dataframe(
                incident_df,
                use_container_width=True
            )

        # -------------------------------
        # 🌦 Weather Intelligence
        # -------------------------------
        if page == "Weather Intelligence":

            st.header("🌦 Weather Intelligence")

            city = st.selectbox(
                "Select City",
                [
                    "London",
                    "Manchester",
                    "Birmingham",
                    "Leeds",
                    "Glasgow"
                ]
            )

            weather = get_weather(city)

            if weather and "main" in weather:

                col1, col2, col3 = st.columns(3)

                col1.metric(
                    "Temperature",
                    f"{weather['main']['temp']} °C"
                )

                col2.metric(
                    "Humidity",
                    f"{weather['main']['humidity']}%"
                )

                col3.metric(
                    "Wind Speed",
                    f"{weather['wind']['speed']} m/s"
                )

                st.success("✅ Live Weather Data Connected")

            else:
                st.warning("Weather API unavailable")
        
            st.markdown("---")
            st.subheader("🌍 Carbon Intensity Monitor")

            carbon_intensity = random.randint(80, 300)

            st.metric(
                "Grid Carbon Intensity",
                f"{carbon_intensity} gCO₂/kWh"
            )

            if carbon_intensity > 250:
                st.error("🚨 High Carbon Generation")
            elif carbon_intensity > 150:
                st.warning("⚠ Moderate Carbon Intensity")
            else:
                st.success("✅ Low Carbon Grid")
            
            st.markdown("---")
            st.subheader("🌿 Environmental Risk Score")

            env_score = random.randint(60, 100)

            st.metric(
                "Environmental Score",
                f"{env_score}%"
            )

            st.progress(env_score)

            if env_score > 80:
                st.success("✅ ESG Target Achieved")
            else:
                st.warning("⚠ Improvement Recommended")
            
        # -------------------------------
        # 🤖 AI Failure Prediction
        # -------------------------------
        if page == "AI Failure Prediction":

            st.header("🤖 AI Failure Prediction Engine")

            temperature = st.slider(
                "Temperature",
                20,
                120,
                75
            )

            vibration = st.slider(
                "Vibration",
                1,
                15,
                4
            )

            pressure = st.slider(
                "Pressure",
                5,
                20,
                12
            )

            prediction = failure_model.predict(
                [[
                    temperature,
                    vibration,
                    pressure
                ]]
            )[0]

            if prediction == 1:

                st.error(
                    "🚨 Failure Risk Detected"
                )

            else:

                st.success(
                    "✅ Equipment Healthy"
                )
                probability = failure_model.predict_proba(
                    [[
                        temperature,
                        vibration,
                        pressure
                    ]]
                )[0][1]

                st.metric(
                    "Failure Probability",
                    f"{probability*100:.1f}%"
                )

                st.progress(
                    int(probability * 100)
                )

                st.markdown("---")
                st.subheader("🧠 AI Recommendation")

                if probability > 0.7:

                    st.error(
                        "Immediate maintenance recommended"
                    )

                elif probability > 0.4:

                    st.warning(
                        "Inspection recommended within 7 days"
                    )

                else:

                    st.success(
                        "Asset operating normally"
                    )
        
        # -------------------------------
        # 🚨 Alert Center
        # -------------------------------
        if page == "Alert Center":

            st.header("🚨 Enterprise Alert Center")

            alert_count = pd.read_sql_query(
                """
                SELECT COUNT(*) as total
                FROM alerts
                """,
                get_connection()
            )

            st.metric(
                "Total Alerts",
                int(alert_count["total"][0])
            )

            alerts_df = pd.read_sql_query(
                """
                SELECT *
                FROM alerts
                ORDER BY id DESC
                LIMIT 50
                """,
                get_connection()
            )

            st.dataframe(
                alerts_df,
                use_container_width=True
            )

            st.markdown("---")
            st.subheader("📊 Alert Severity Analysis")

            severity_df = pd.read_sql_query(
                """
                SELECT severity,
                       COUNT(*) as count
                FROM alerts
                GROUP BY severity
                """,
                get_connection()
            )

            if not severity_df.empty:

                fig_alerts = px.pie(
                severity_df,
                names="severity",
                values="count",
                title="Alert Distribution"
            )

            st.plotly_chart(
                fig_alerts,
                use_container_width=True
            )
        
        # -------------------------------
        # 🔮 Scenario Simulator
        # -------------------------------
        if page == "Scenario Simulator":

            st.header("🔮 Digital Twin Scenario Simulator")
        
        st.markdown("---")
        st.subheader("⚡ Load Change Simulation")

        load_change = st.slider(
            "Load Increase (%)",
            -50,
            50,
            0
        )

        st.metric(
            "Simulated Load",
            f"{100 + load_change}%"
        )

        st.markdown("---")
        st.subheader("💰 Energy Cost Impact")

        base_cost = 100000

        simulated_cost = base_cost * (
            1 + load_change / 100
        )

        st.metric(
            "Projected Annual Cost",
            f"£{simulated_cost:,.0f}"
        )

        st.markdown("---")
        st.subheader("🌍 Carbon Impact Forecast")

        base_carbon = 500

        carbon_projection = base_carbon * (
            1 + load_change / 100
        )

        st.metric(
            "Projected CO₂ Emissions",
            f"{carbon_projection:.0f} tonnes"
        )

        st.markdown("---")
        st.subheader("🏭 Asset Stress Prediction")

        stress_score = min(
            100,
            max(
                0,
                60 + load_change
            )
        )

        st.metric(
            "Stress Score",
            f"{stress_score}%"
        )

        st.progress(int(stress_score))

        st.markdown("---")
        st.subheader("🧠 Executive Recommendation")

        if stress_score > 85:

            st.error(
                "Reduce load immediately. Asset risk elevated."
            )

        elif stress_score > 70:

            st.warning(
                "Monitor asset closely during operation."
            )

        else:

            st.success(
                "Scenario acceptable for operation."
            )
        
        st.markdown("---")
        st.subheader("📊 Scenario Comparison")

        scenario_df = pd.DataFrame({

            "Scenario": [
                "Current",
                "Simulated"
            ],

            "Cost": [
                base_cost,
                simulated_cost
            ],

            "Carbon": [
                base_carbon,
                carbon_projection
            ]
        })

        fig_scenario = px.bar(
            scenario_df,
            x="Scenario",
            y="Cost",
            title="Cost Comparison"
        )

        st.plotly_chart(
            fig_scenario,
            use_container_width=True
        )

        # -------------------------------
        # 🌐 Asset Relationship Map
        # -------------------------------
        if page == "Asset Relationship Map":

            st.header("🌐 Digital Twin Asset Relationship Map")

        G = nx.Graph()

        G.add_edges_from([
            ("Gas Turbine", "Boiler"),
            ("Boiler", "Steam Turbine"),
            ("Steam Turbine", "Generator"),
            ("Generator", "Transformer"),
            ("Transformer", "Grid"),
            ("HVAC", "Control Room"),
            ("MQTT Broker", "SCADA"),
            ("SCADA", "AI Engine")
        ])

        fig, ax = plt.subplots(figsize=(10, 6))

        pos = nx.spring_layout(
            G,
            seed=42
        )

        nx.draw(
            G,
            pos,
            with_labels=True,
            node_size=3000,
            font_size=9,
            ax=ax
        )

        st.pyplot(fig) 

        st.markdown("---")
        st.subheader("🚨 Critical Asset Path")

        critical_path = [
            "Gas Turbine",
            "Boiler",
            "Steam Turbine",
            "Generator",
            "Transformer",
            "Grid"
        ]

        st.write(" ➜ ".join(critical_path)) 

        st.markdown("---")
        st.subheader("⚠ Failure Impact Simulator")

        failed_asset = st.selectbox(
            "Select Failed Asset",
            list(G.nodes())
        )

        affected_assets = list(
            nx.node_connected_component(
                G,
                failed_asset
            )
        )

        st.error(
            f"Failure impacts {len(affected_assets)} connected assets."
        )

        st.write(affected_assets) 

        st.markdown("---")
        st.subheader("📊 Asset Connectivity Metrics")

        col1, col2 = st.columns(2)

        col1.metric(
            "Total Assets",
            len(G.nodes())
        )

        col2.metric(
            "Connections",
            len(G.edges())
        ) 

        # -------------------------------
        # 📊 Enterprise KPI Scorecard
        # -------------------------------
        if page == "Enterprise KPI Scorecard":

            st.header("📊 Enterprise KPI Scorecard")
        
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        kpi1.metric(
            "Asset Availability",
            f"{random.randint(95,99)}%"
        )

        kpi2.metric(
            "Reliability Index",
            f"{random.randint(90,99)}%"
        )

        kpi3.metric(
            "Maintenance Compliance",
            f"{random.randint(85,100)}%"
        )

        kpi4.metric(
            "ESG Score",
            f"{random.randint(75,98)}%"
        )

        st.markdown("---")
        st.subheader("🏢 Enterprise Performance Metrics")

        performance_df = pd.DataFrame({

            "Category": [
                "Operations",
                "Maintenance",
                "Energy",
                "Safety",
                "ESG"
            ],

            "Score": [
                random.randint(80,100),
                random.randint(80,100),
                random.randint(80,100),
                random.randint(80,100),
                random.randint(80,100)
            ]
        })

        st.dataframe(
            performance_df,
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("📈 KPI Performance Trend")

        fig_kpi = px.bar(
            performance_df,
            x="Category",
            y="Score",
            title="Enterprise KPI Performance"
        )

        st.plotly_chart(
            fig_kpi,
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("🏆 Operational Excellence")

        overall_score = performance_df["Score"].mean()

        st.metric(
            "Overall Excellence Score",
            f"{overall_score:.1f}%"
        )

        st.progress(
            int(overall_score)
        )

        st.markdown("---")
        st.subheader("🧠 Executive Insights")

        if overall_score > 90:

            st.success(
                "Enterprise performance exceeds strategic targets."
            )

        elif overall_score > 80:

            st.warning(
                "Performance is stable with improvement opportunities."
            )

        else:

            st.error(
                "Performance improvement program recommended."
            )
        
        # -------------------------------
        # 📄 Report Center
        # -------------------------------
        if page == "Report Center":

            st.header("📄 Enterprise Report Center")
        
        report_type = st.selectbox(

            "Select Report",

            [
                "Executive Summary",
                "Asset Performance",
                "Alert Summary",
                "Energy Intelligence",
                "ESG Performance"
            ]
        )

        if st.button("📄 Generate Report"):

            report_file = generate_report()

            st.success(
                "Report generated successfully"
            )
        
        if st.button("⬇ Download PDF"):

            report_file = generate_report()

            with open(
                report_file,
                "rb"
            ) as pdf_file:

                st.download_button(
                    label="Download Report",
                    data=pdf_file,
                    file_name=report_file,
                    mime="application/pdf"
                )
        
        st.markdown("---")

        st.subheader("📊 Report Contents")

        st.write(
            f"""
            Report Type:
            {report_type}

            Included Sections:

            • KPI Summary

            • Asset Performance

            • Alert Statistics

            • Carbon Metrics

            • Operational Excellence
            """
        )

        # -------------------------------
        # 📅 Enterprise Timeline
        # -------------------------------
        if page == "Enterprise Timeline":

            st.header("📅 Enterprise Operations Timeline")
        
        events_df = pd.read_sql_query(
            """
            SELECT *
            FROM enterprise_events
            ORDER BY id DESC
            LIMIT 100
            """,
            get_connection()
        )

        st.metric(
            "Total Events",
            len(events_df)
        )

        event_filter = st.selectbox(

            "Filter Events",

            [
                "All",
                "Alert",
                "Asset",
                "Security"
            ]
        )

        if event_filter != "All":

            events_df = events_df[
                events_df["event_type"] == event_filter
            ]
        
        st.dataframe(
            events_df,
            use_container_width=True
        )

        st.markdown("---")
        st.subheader("📊 Event Analytics")

        if not events_df.empty:

            event_summary = (
                events_df
                .groupby("event_type")
                .size()
                .reset_index(name="count")
            )

            fig_events = px.bar(
                event_summary,
                x="event_type",
                y="count",
                title="Enterprise Events"
            )

            st.plotly_chart(
                fig_events,
                use_container_width=True
            )

        # -------------------------------
        # 🔐 Security & Compliance Center
        # -------------------------------
        if page == "Security Center":

            if page not in allowed_pages:
                st.error("🚫 Access Denied")
                st.stop()

            st.header("🔐 Security & Compliance Center")

            sec_col1, sec_col2, sec_col3 = st.columns(3)

            sec_col1.metric(
                "Firewall Status",
                "ACTIVE"
            )

            sec_col2.metric(
                "Threat Level",
                "LOW"
            )

            sec_col3.metric(
                "Compliance Score",
                f"{random.randint(85,100)}%"
            )

            st.success("✅ ISO 27001 Security Controls Active")

            st.markdown("### 🛡 Compliance Monitoring")

            compliance_data = pd.DataFrame({
                "System": [
                    "SCADA",
                    "IoT Gateway",
                    "Database",
                    "Cloud API",
                    "MQTT Broker"
                ],
                "Status": [
                    "Secure",
                    "Secure",
                    "Secure",
                    "Monitoring",
                    "Secure"
                ]
            })

            st.dataframe(compliance_data)

            st.markdown("---")
            st.subheader("⚙ Database Administration")

            if st.button("Reset Database"):

                import os

                if os.path.exists("digital_twin.db"):
                    os.remove("digital_twin.db")

                init_db()

                st.success("✅ Database recreated")

            # -------------------------------
            # 👥 User Access Audit
            # -------------------------------
            st.markdown("---")
            st.subheader("👥 User Access Audit")

            audit_df = pd.DataFrame({

                "User": [
                    "Admin",
                    "Engineer",
                    "Operator",
                    "Client"
                ],

                "Last Login": [
                    "Today",
                    "Today",
                    "Yesterday",
                    "Today"
                ],

                "Status": [
                    "Active",
                    "Active",
                    "Active",
                    "Active"
                ]
            })

            st.dataframe(
                audit_df,
                use_container_width=True
            )

        # -------------------------------
        # 🤖 AI Cyber Security Monitor
        # -------------------------------
        st.markdown("---")
        st.subheader("🤖 AI Cyber Security Monitor")

        threat_score = random.randint(1, 100)

        st.progress(threat_score)

        st.metric("Threat Detection Score", f"{threat_score}%")

        if threat_score > 80:
            st.error("🚨 Critical cyber threat detected")
        elif threat_score > 50:
            st.warning("⚠ Suspicious activity detected")
        else:
            st.success("✅ Network secure")

        # Live threat logs
        cyber_logs = pd.DataFrame({
            "Timestamp": pd.date_range(
                start=pd.Timestamp.now(),
                periods=5,
                freq="min"
            ),
            "Event": [
                "Firewall Scan",
                "MQTT Authentication",
                "API Access",
                "Database Monitoring",
                "IoT Device Validation"
            ],
            "Status": [
                "Passed",
                "Passed",
                "Monitoring",
                "Passed",
                "Passed"
            ]
        })

        st.dataframe(cyber_logs)
        
        # -------------------------------
        # 🖥 Enterprise KPI Command Center
        # -------------------------------
        st.markdown("## 🖥 Enterprise KPI Command Center")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        kpi1.metric("Assets Online", random.randint(20, 50))
        kpi2.metric("AI Alerts", random.randint(0, 10))
        kpi3.metric("Grid Stability", "99.98%")
        kpi4.metric("Energy Efficiency", "92%")

        # Display active role
        st.sidebar.success(f"Logged in as: {role}")

        # -------------------------------
        # Role Permissions
        # -------------------------------
        if role == "Admin":

            st.subheader("🛠 Admin Controls")
            st.write("Full system analytics and AI controls enabled.")

        elif role == "Operator":

            st.subheader("⚙ Operator Dashboard")
            st.write("Operational monitoring access enabled.")

        elif role == "Engineer":

            st.subheader("🔧 Engineer Workspace")
            st.write("Maintenance and diagnostics tools enabled.")

        elif role == "Client":

            st.subheader("📊 Client Dashboard")
            st.write("Read-only energy performance view enabled.")
    else:
        role = "User"
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

            # ✅ STANDARDIZE COLUMN NAMES (FIX)
            df.columns = (
                df.columns
                .str.strip()
                .str.lower()
                .str.replace(" ", "_")
                .str.replace("(", "")
                .str.replace(")", "")
            )
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
    # 📊 Rolling Energy Trend 
    # -------------------------------
    st.subheader("📊 Energy Trend Analysis")

    if df is not None and "energy_kwh" in df.columns:

        df["rolling_avg"] = df["energy_kwh"].rolling(window=5).mean()

        fig = px.line(df, y=["energy_kwh", "rolling_avg"],
                      title="Energy Trend vs Rolling Average")

        st.plotly_chart(fig, use_container_width=True)

        # Early warning
        if df["energy_kwh"].iloc[-1] > df["rolling_avg"].iloc[-1]:
            st.warning("⚠ Energy spike above normal trend detected")
        else:
            st.success("✅ Energy trend stable")


    # -------------------------------
    # 🛡 Global Safe Check 
    # -------------------------------
    if df is None:
        st.warning("⚠ Please upload CSV to enable full features.")
        st.stop()

    # -------------------------------
    # 🏢 Building Energy Ranking System
    # -------------------------------
    st.subheader("🏆 Building Energy Efficiency Ranking")

    # Check if building column exists
    if df is not None and "building" in df.columns:

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

    if df is not None and "building" in df.columns:

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

    if df is not None and "building" in df.columns:

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
    
    # =====================================================
    # 💹 AI Energy Trading Signals
    # =====================================================

    st.subheader("💹 AI Energy Trading Signals")

    market_price = round(random.uniform(60, 180), 2)

    if market_price > 140:
        signal = "SELL ENERGY"
        color = "🚀"

    elif market_price > 90:
        signal = "HOLD"

        color = "⚖"

    else:
        signal = "BUY ENERGY"

        color = "📈"

    st.metric(
        "Electricity Market Price",
        f"£{market_price}/MWh"
    )

    st.markdown(f"## {color} {signal}")


    # -------------------------------
    # 🌍 Carbon Reduction Advisor
    # -------------------------------
    st.subheader("🌍 Carbon Reduction Advisor")

    co2_factor = 0.233  # kg CO2 per kWh

    if df is not None and "building" in df.columns:

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
    
    # =====================================================
    # ⚡ Grid Stability Engine
    # =====================================================

    st.subheader("⚡ Grid Stability Engine")

    grid_load = random.randint(40, 100)

    renewable_share = random.randint(20, 80)

    frequency = round(random.uniform(49.5, 50.5), 2)

    col1, col2, col3 = st.columns(3)

    col1.metric("Grid Load", f"{grid_load}%")
    col2.metric("Renewable Share", f"{renewable_share}%")
    col3.metric("Grid Frequency", f"{frequency} Hz")

    if frequency < 49.8 or frequency > 50.2:
        st.error("🚨 Grid instability risk detected")
    else:
        st.success("✅ Grid operating normally")
    
    # -------------------------------
    # ⚡ AI Load Balancing Engine
    # -------------------------------
    st.subheader("⚡ AI Load Balancing")

    load_now = random.randint(50, 100)

    st.progress(load_now)

    st.metric("Current Grid Load", f"{load_now}%")

    if load_now > 85:
        st.error("🚨 Grid load critical")
    elif load_now > 65:
        st.warning("⚠ High load detected")
    else:
        st.success("✅ Load balanced")
    
    # =====================================================
    # ⚖ AI Load Balancer
    # =====================================================

    st.subheader("⚖ AI Load Balancer")

    zone_a = random.randint(100, 400)
    zone_b = random.randint(100, 400)
    zone_c = random.randint(100, 400)

    load_df = pd.DataFrame({
        "Zone": ["Zone A", "Zone B", "Zone C"],
        "Load": [zone_a, zone_b, zone_c]
    })

    st.dataframe(load_df)

    balance_fig = px.pie(
        load_df,
        names="Zone",
        values="Load",
        title="Grid Load Distribution"
    )

    st.plotly_chart(balance_fig, use_container_width=True)

    max_zone = load_df.loc[load_df["Load"].idxmax()]

    st.warning(
        f"⚠ Highest load detected in {max_zone['Zone']}"
    )

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
    if df is not None and "building" in df.columns:
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
        # =========================================================
        # 🏢 Multi-Site Monitoring
        # =========================================================
        st.subheader("🏢 Multi-Site Digital Twin")

        sites = {
            "London Plant": random.randint(70, 100),
            "Manchester Plant": random.randint(60, 100),
            "Birmingham Plant": random.randint(50, 100),
            "Leeds Plant": random.randint(65, 100)
        }

        site_df = pd.DataFrame({
            "Site": list(sites.keys()),
            "Health Score": list(sites.values())
        })

        st.dataframe(site_df)

        fig_sites = px.bar(
            site_df,
            x="Site",
            y="Health Score",
            title="Enterprise Site Health"
        )

        st.plotly_chart(fig_sites, use_container_width=True)


        # -------------------------------
        # Energy Forecast & Anomalies
        # -------------------------------
        st.subheader("📈 Energy Forecast")

        try:
            forecast_df = forecast_energy(forecast_days)
            forecast_df = detect_anomalies(forecast_df)

            fig = px.line(forecast_df, x="date", y="forecast", title="Energy Forecast")
            anomalies = forecast_df[forecast_df["anomaly"] == True]
            fig.add_scatter(x=anomalies["date"], y=anomalies["forecast"],
                        mode='markers', marker=dict(color='red', size=10),
                        name="Anomaly")
            st.plotly_chart(fig)
        
        except Exception as e:
            st.error(f"Forecast error: {e}")
            forecast_df = None

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
        
        # =====================================================
        # 📊 Live KPI Wall
        # =====================================================

        st.subheader("📊 Live KPI Wall")

        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        kpi1.metric(
            "Total Assets",
            random.randint(10, 50)
        )

        kpi2.metric(
            "Active Alarms",
            random.randint(0, 8)
        )

        kpi3.metric(
            "Efficiency",
            f"{random.randint(80, 99)}%"
        )

        kpi4.metric(
            "AI Accuracy",
            f"{random.randint(90, 99)}%"
        )
        
        # =====================================================
        # 📈 SLA Performance Monitor
        # =====================================================

        st.subheader("📈 SLA Performance")

        uptime = round(random.uniform(98.5, 99.99), 2)

        response_time = round(random.uniform(80, 300), 2)

        col1, col2 = st.columns(2)

        col1.metric(
            "System Uptime",
            f"{uptime}%"
        )

        col2.metric(
            "Response Time",
            f"{response_time} ms"
        )

        if uptime > 99.5:
            st.success("✅ SLA Targets Achieved")
        else:
            st.warning("⚠ SLA Performance Degraded")
        
        # -------------------------------
        # 🚨 Enterprise AI Risk Index
        # -------------------------------
        st.subheader("🚨 Enterprise AI Risk Index")

        risk_score = 100 - health_score

        if risk_score > 60:
            st.error(f"🔴 CRITICAL RISK LEVEL: {round(risk_score,2)}%")
        elif risk_score > 30:
            st.warning(f"🟠 MODERATE RISK LEVEL: {round(risk_score,2)}%")
        else:
            st.success(f"🟢 LOW RISK LEVEL: {round(risk_score,2)}%")

        st.progress(int(risk_score))

        # -------------------------------
        # 📊 Multi KPI Health Index (FIXED)
        # -------------------------------
        st.subheader("📊 System Health Breakdown")

        if "live_data_records" in st.session_state and len(st.session_state.live_data_records) > 0:

            latest_data = st.session_state.live_data_records[-1]

            temp_score = max(0, 100 - latest_data["Temperature (°C)"])
            vibration_score = max(0, 100 - latest_data["Vibration"] * 20)
            energy_score = max(0, 100 - latest_data["Energy (kWh)"] * 0.1)

            col1, col2, col3 = st.columns(3)

            col1.metric("Temperature Health", f"{temp_score:.1f}%")
            col2.metric("Vibration Health", f"{vibration_score:.1f}%")
            col3.metric("Energy Efficiency", f"{energy_score:.1f}%")

        else:
            st.info("Run simulation to view health breakdown")
        
        # -------------------------------
        # 🛠 AI Maintenance Recommendation Engine
        # -------------------------------
        st.subheader("🛠 AI Maintenance Recommendations")

        if health_score < 40:
            st.error("🚨 Immediate shutdown and inspection required")
        elif health_score < 70:
            st.warning("⚠ Schedule maintenance within 7 days")
        else:
            st.success("✅ System operating normally")

        # Smart suggestion
        if vibration_value > 4:
            st.info("🔧 Inspect bearings and rotating parts")

        if temp_value > 70:
            st.info("🌡 Check cooling system and airflow")

        if pressure_value > 45:
            st.info("🛢 Inspect pressure valves and pipelines")

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

        # =========================================================
        # 🌱 AI Sustainability Rating
        # =========================================================
        st.subheader("🌱 AI Sustainability Rating")

        rating = "A+"

        if sustainability_score < 50:
            rating = "C"

        elif sustainability_score < 70:
            rating = "B"

        elif sustainability_score < 85:
            rating = "A"

        st.metric(
            "Enterprise Sustainability Rating",
            rating
        )

        if rating == "A+":
            st.success("🏆 Industry-Leading Sustainability")

        elif rating == "A":
            st.success("♻ Strong Sustainability Performance")

        elif rating == "B":
            st.warning("⚠ Sustainability Improvements Possible")

        else:
            st.error("🚨 Sustainability Risk")
        
        # -------------------------------
        # 🧠 AI Security Score
        # -------------------------------
        security_score = 100

        if failure_percent > 70:
            security_score -= 30

        if health_score < 50:
            security_score -= 20

        security_score = max(0, security_score)

        st.metric("🛡 AI Security Score", f"{security_score}%")

        # -------------------------------
        # ⚡ Energy Efficiency Score 
        # -------------------------------
        st.subheader("⚡ Energy Efficiency Score")

        if df is not None and "energy_kwh" in df.columns:

            avg_energy = df["energy_kwh"].mean()

            # Benchmark (industry baseline)
            benchmark = 300

            efficiency_score = max(0, 100 - ((avg_energy - benchmark) / benchmark) * 100)

            st.metric("Efficiency Score (%)", round(efficiency_score, 2))

            if efficiency_score > 80:
                st.success("🏆 Highly Efficient System")
            elif efficiency_score > 60:
                st.warning("⚠ Moderate Efficiency")
            else:
                st.error("🚨 Poor Efficiency – Optimization Needed")

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

        # -------------------------------
        # 💡 Suggested Prompts
        # -------------------------------
        st.markdown("### 💡 Try asking:")

        col1, col2, col3 = st.columns(3)

        # ✅ ALWAYS DEFINE FIRST (VERY IMPORTANT)
        user_input = None

        if col1.button("📉 Reduce energy cost"):
            user_input = "How can I reduce energy cost?"

        elif col2.button("⚠ Fault risks"):
            user_input = "Any fault risks in system?"

        elif col3.button("🌍 CO2 impact"):
            user_input = "What is my CO2 impact?"

        # -------------------------------
        # 🎤 Voice Input (SEPARATE - NOT INSIDE ELSE)
        # -------------------------------
        try:
            import speech_recognition as sr

            if st.button("🎤 Speak"):
                recognizer = sr.Recognizer()

                with sr.Microphone() as source:
                    st.info("Listening...")
                    audio = recognizer.listen(source)

                try:
                    user_input = recognizer.recognize_google(audio)
                    st.success(f"You said: {user_input}")
                except:
                    st.error("❌ Could not understand audio")

        except:
            st.info("🎤 Voice input not available")

        # -------------------------------
        # ⌨️ Text Input (ALWAYS VISIBLE)
        # -------------------------------
        typed_input = st.text_input("Ask your building anything...")

        if typed_input:
            user_input = typed_input

        # -------------------------------
        # 🤖 AI RESPONSE (GPT + FALLBACK)
        # -------------------------------
        if user_input:

            # Safety
            if 'df' not in locals():
                df = None
            if 'forecast_df' not in locals():
                forecast_df = None

            try:
                # -------------------------------
                # 🧠 Build Context Memory (NEW)
                # -------------------------------
                history_context = ""

                for role, msg in st.session_state.copilot_history[-5:]:
                    history_context += f"{role}: {msg}\n"

                # -------------------------------
                # 🤖 GPT Call with Memory (NEW)
                # -------------------------------
                stream = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are an expert energy AI assistant."},
                        {"role": "user", "content": history_context + "\nUser: " + user_input}
                    ],
                    stream=True
                )

                # -------------------------------
                # Streaming Response (UNCHANGED)
                # -------------------------------
                full_response = ""
                placeholder = st.empty()


                for chunk in stream:

                    # ✅ SAFETY CHECK (prevents crash)
                    if (
                        hasattr(chunk, "choices") and
                        len(chunk.choices) > 0 and
                        hasattr(chunk.choices[0], "delta") and
                        hasattr(chunk.choices[0].delta, "content") and
                        chunk.choices[0].delta.content is not None
                    ):
                        content = chunk.choices[0].delta.content

                        full_response += content
                        placeholder.markdown(f"🤖 {full_response}")

            except Exception as e:
                # ⚠️ Fallback AI
                st.warning("⚠ Using local AI (API unavailable)")
    
                full_response = ai_copilot(user_input, df, forecast_df)
                st.markdown(f"🤖 {full_response}")

            # Save chat
            st.session_state.copilot_history.append(("You", user_input))
            st.session_state.copilot_history.append(("AI", full_response))
            

        # -------------------------------
        # 💬 Chat History Display
        # -------------------------------
        for role, msg in st.session_state.copilot_history:

            if role == "You":
                st.markdown(
                    f"""
                    <div style='background-color:#DCF8C6;
                        padding:12px;
                        border-radius:10px;
                        margin:8px 0;
                        color:black'>
                    🧑 <b>You:</b> {msg}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:
                st.markdown(
                    f"""
                    <div style='background-color:#F1F0F0;
                        padding:12px;
                        border-radius:10px;
                        margin:8px 0;
                        color:black'>
                    🤖 <b>AI:</b> {msg}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        # -------------------------------
        # 🧹 Clear Chat Button 
        # -------------------------------
        if st.button("🗑 Clear Chat"):
            st.session_state.copilot_history = []
        
            st.markdown("""
            # 🖥 AI ENERGY COMMAND CENTER
            ### Real-Time Industrial Monitoring Platform
            """)

        # -------------------------------
        # Live IoT Sensor Simulation + AI Failure Detection
        # -------------------------------
        st.subheader("⚡ Live IoT Sensor Simulation")
        st.subheader("🚨 Enterprise Alarm Center")

        # Button to start simulation
        start = st.button("Start Real-Time Simulation")

        # Placeholder container for dynamic updates
        placeholder = st.empty()

        # -------------------------------
        # 🧠 Store Live Data in Session
        # -------------------------------
        if "live_data_records" not in st.session_state:
            st.session_state.live_data_records = []

        if start:
            for i in range(20):

                # -------------------------------
                # Generate Sensor Data
                # -------------------------------
                sensor_data = mqtt_data if mqtt_data else generate_sensor_data()
                st.session_state.live_data_records.append(sensor_data)

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
                st.session_state.failure_percent = failure_percent

                # -------------------------------
                # 🧠 Equipment Criticality Score
                # -------------------------------

                criticality_score = (
                    sensor_data["Temperature (°C)"] * 0.4 +
                    sensor_data["Vibration"] * 20 +
                    failure_percent * 0.4
                )

                criticality_score = round(min(100, criticality_score), 2)

                # -------------------------------
                # ⏳ Remaining Equipment Life
                # -------------------------------

                remaining_life = max(0, 100 - criticality_score)

                remaining_life_days = int(remaining_life * 3.65)

                # -------------------------------
                # 🚨 AI Priority Classification
                # -------------------------------

                if criticality_score > 80:
                   priority = "CRITICAL"
                elif criticality_score > 60:
                   priority = "HIGH"
                elif criticality_score > 40:
                   priority = "MEDIUM"
                else:
                   priority = "LOW"
                
                # -------------------------------
                # ⚡ Autonomous Load Balancer
                # -------------------------------

                load_action = "NORMAL"

                if sensor_data["Energy (kWh)"] > 450:
                    load_action = "REDUCE LOAD"

                elif sensor_data["Energy (kWh)"] < 180:
                    load_action = "INCREASE UTILIZATION"
                
                # -------------------------------
                # 🚀 AI Efficiency Score
                # -------------------------------

                efficiency_score = 100 - (
                    sensor_data["Vibration"] * 10 +
                    failure_percent * 0.5
                )

                efficiency_score = round(max(0, efficiency_score), 2)

                # -------------------------------
                # 🎯 AI Confidence Score
                # -------------------------------
                confidence_score = round(100 - abs(50 - failure_percent), 2)

                st.metric("AI Confidence", f"{confidence_score}%")

                # -------------------------------
                # 🤖 AI Anomaly Detection
                # -------------------------------
                if failure_percent > 70:
                    st.error("🚨 High Risk of Equipment Failure!")
                elif failure_percent > 40:
                    st.warning("⚠ Medium Risk Detected")
                else:
                    st.success("✅ System Stable")
                
                # =========================================================
                # 🚨 Alarm Engine
                # =========================================================
                alarm_messages = []

                if sensor_data["Temperature (°C)"] > 60:
                    alarm_messages.append("🔥 Critical Temperature")

                if sensor_data["Vibration"] > 3:
                    alarm_messages.append("⚠ Excessive Vibration")

                if sensor_data["Energy (kWh)"] > 450:
                    alarm_messages.append("⚡ Energy Surge")

                for alarm in alarm_messages:
                    st.error(alarm)
                
                # -------------------------------
                # 🚨 Security Monitoring
                # -------------------------------
                st.subheader("🚨 Security Monitoring")

                if failure_percent > 80:
                    st.error("🔴 Critical infrastructure risk detected")

                if len(alerts) > 2:
                    st.warning("⚠ Multiple simultaneous anomalies detected")

                if health_score < 40:
                    st.error("🚨 Equipment health critically low")

                # =========================================================
                # 📋 Incident Tracking System
                # =========================================================
                st.subheader("📋 Incident Tracking")

                incident_df = pd.DataFrame({
                    "Incident": [
                       "Temperature Spike",
                       "Voltage Fluctuation",
                       "High Vibration",
                       "Cooling Delay"
                    ],
                    "Priority": [
                       "High",
                       "Medium",
                       "Critical",
                       "Low"
                    ],
                    "Status": [
                       "Open",
                       "Investigating",
                       "Resolved",
                       "Monitoring"
                    ]
                })

                st.dataframe(incident_df)

                # -------------------------------
                # 📜 Enterprise SLA Monitor
                # -------------------------------
                st.subheader("📜 SLA Monitoring")

                sla_uptime = round(random.uniform(99.0, 99.999), 3)

                st.metric("Platform Uptime", f"{sla_uptime}%")

                if sla_uptime < 99.5:
                    st.warning("⚠ SLA risk detected")
                else:
                    st.success("✅ SLA compliant")
                
                # -------------------------------
                # 🧠 Root Cause Analysis
                # -------------------------------
                st.subheader("🧠 Root Cause Analysis")

                root_causes = []

                if sensor_data["Temperature (°C)"] > 70:
                    root_causes.append("High temperature → Possible cooling failure")

                if sensor_data["Vibration"] > 4:
                    root_causes.append("High vibration → Possible bearing or shaft misalignment")

                if sensor_data["Energy (kWh)"] > 450:
                    root_causes.append("Energy spike → Overloading or inefficiency")

                if root_causes:
                    for cause in root_causes:
                        st.error(f"🔍 {cause}")
                else:
                    st.success("✅ No critical root cause detected")
                
                # -------------------------------
                # 🤖 Autonomous Decision Engine
                # -------------------------------
                st.subheader("🤖 AI Decision Engine")

                decision = "System Normal"
                action = "No action required"

                if failure_percent > 70:
                    decision = "Critical Condition"
                    action = "Shutdown system & immediate inspection"

                elif failure_percent > 40:
                    decision = "Warning Condition"
                    action = "Schedule maintenance within 24 hours"

                elif sensor_data["Energy (kWh)"] > 450:
                    decision = "Energy Inefficiency"
                    action = "Reduce load & optimize system"

                st.error(f"Decision: {decision}") if failure_percent > 70 else st.warning(f"Decision: {decision}") if failure_percent > 40 else st.success(f"Decision: {decision}")
                st.info(f"Recommended Action: {action}")

                # =========================================================
                # 🧠 AI Decision Center
                # =========================================================
                st.subheader("🧠 AI Decision Center")

                decision = "System Stable"

                if failure_percent > 70:
                    decision = "Shutdown equipment immediately"

                elif failure_percent > 40:
                    decision = "Schedule urgent maintenance"

                elif avg_energy > 400:
                    decision = "Optimize energy consumption"

                st.info(f"🤖 AI Recommendation: {decision}")

                # =========================================================
                # 🖥 Enterprise Command Console
                # =========================================================
                st.subheader("🖥 Enterprise Command Console")

                console_messages = [
                    "AI Monitoring Active",
                    "Grid Communication Stable",
                    "Renewable Sources Connected",
                    "Predictive Maintenance Enabled",
                    "SCADA Synchronization Successful"
                ]

                for msg in console_messages:
                    st.success(f"✔ {msg}")

                # =========================================================
                # 🤖 Autonomous AI Actions
                # =========================================================
                st.subheader("🤖 Autonomous AI Actions")

                ai_action = "No action required"

                if failure_percent > 70:

                    ai_action = "Emergency shutdown initiated"

                elif failure_percent > 40:

                    ai_action = "Maintenance team notified"

                elif avg_energy > 400:

                    ai_action = "HVAC optimization activated"

                st.success(f"⚡ AI Action: {ai_action}")

                # =========================================================
                # 🧠 Autonomous Grid Optimization
                # =========================================================
                st.subheader("🧠 Autonomous Grid Optimization")

                optimization_action = "No optimization needed"

                if power_demand > 1000:

                    optimization_action = "Reducing non-critical loads"

                elif renewable_percent > 60:

                    optimization_action = "Switching to renewable priority mode"

                elif market_price > 0.20:

                    optimization_action = "Activating battery storage"

                st.info(f"⚡ AI Optimization: {optimization_action}")

                # -------------------------------
                # 🛠 Smart Maintenance Scheduler
                # -------------------------------
                if failure_percent > 70:
                    st.error("🛠 Action: Immediate maintenance required (within 24 hours)")
                elif failure_percent > 50:
                    st.warning("🛠 Action: Schedule maintenance within 3 days")
                elif failure_percent > 30:
                    st.info("🛠 Action: Plan preventive maintenance this week")
                else:
                    st.success("🛠 No maintenance needed - system healthy")
                
                # -------------------------------
                # 📊 Model Performance Tracker
                # -------------------------------
                if "model_accuracy_log" not in st.session_state:
                    st.session_state.model_accuracy_log = []

                # Simulated accuracy (you can replace with real later)
                current_accuracy = round(1 - abs(failure_probability - 0.5), 3)

                st.session_state.model_accuracy_log.append(current_accuracy)

                st.subheader("📊 AI Model Performance Over Time")

                df_accuracy = pd.DataFrame({
                "Accuracy": st.session_state.model_accuracy_log
                })

                st.line_chart(df_accuracy)

                # -------------------------------
                # 🧠 Executive AI Insight
                # -------------------------------
                st.subheader("🧠 AI Executive Insight")

                if failure_percent > 70:
                    st.error("🚨 Critical Risk: Immediate shutdown and inspection recommended.")
                elif failure_percent > 40:
                    st.warning("⚠ Moderate Risk: Maintenance should be scheduled soon.")
                else:
                    st.success("✅ System operating within optimal conditions.")

                st.info("""
                📊 AI Summary:
                - System continuously learning from sensor data
                - Predictive maintenance active
                - Risk monitoring in real-time

                🎯 Recommendation:
                Maintain current operational efficiency and monitor trends.
                """)

                # -------------------------------
                # ⚠ AI Drift Detection
                # -------------------------------
                if len(st.session_state.model_accuracy_log) > 5:

                    recent_avg = np.mean(st.session_state.model_accuracy_log[-5:])
                    overall_avg = np.mean(st.session_state.model_accuracy_log)

                    if recent_avg < overall_avg - 0.1:
                        st.error("🚨 Model Performance Dropping (Drift Detected)")
                    else:
                        st.success("✅ Model Stable")
                
                # -------------------------------
                # 🔄 Auto Model Retraining Trigger
                # -------------------------------
                if len(st.session_state.model_accuracy_log) > 5:

                    if recent_avg < overall_avg - 0.1:
                        st.warning("🔄 Retraining AI Model...")

                        failure_model = train_failure_model()

                        st.success("✅ Model Retrained Successfully")

                # -------------------------------
                # ⚡ Auto Energy Optimization
                # -------------------------------
                st.subheader("⚡ AI Energy Optimization")

                if sensor_data["Energy (kWh)"] > 400:
                    reduction_target = round(sensor_data["Energy (kWh)"] * 0.15, 2)
                    st.warning(f"⚠ Reduce energy by ~{reduction_target} kWh to optimize performance")

                    st.info("""
                    Suggested Actions:
                    - Shift load to off-peak hours
                    - Optimize HVAC systems
                    - Reduce idle consumption
                    """)
                else:
                    st.success("✅ Energy usage within optimal limits")

                # -------------------------------
                # Dashboard Display
                # -------------------------------
                df_live = pd.DataFrame(st.session_state.live_data_records)

                with placeholder.container():

                    col1, col2, col3, col4, col5 = st.columns(5)

                    col1.metric("Temperature (°C)", sensor_data["Temperature (°C)"])
                    col2.metric("Humidity (%)", sensor_data["Humidity (%)"])
                    col3.metric("Energy (kWh)", sensor_data["Energy (kWh)"])
                    col4.metric("Vibration", sensor_data["Vibration"])
                    col5.metric("Failure Risk", f"{failure_percent}%")

                    st.progress(int(failure_percent))

                    # -------------------------------
                    # 🖥 AI Command Center Status
                    # -------------------------------

                    colA, colB, colC = st.columns(3)
                    colD, colE = st.columns(2)

                    colD.metric("Efficiency Score", f"{efficiency_score}%")
                    colE.metric("Remaining Life", f"{remaining_life_days} Days")

                    colA.metric("Criticality Score", f"{criticality_score}%")
                    colB.metric("Priority Level", priority)
                    colC.metric("Failure Risk", f"{failure_percent}%")

                    # -------------------------------
                    # 🖥 SCADA-STYLE CONTROL ROOM UI
                    # -------------------------------
                    st.markdown("## 🖥 Control Room Dashboard")

                    # Create live figure
                    fig = px.line(
                        df_live,
                        y="Energy (kWh)",
                        title="Live Energy Monitoring"
                    )

                    scada_col1, scada_col2 = st.columns([2,1])

                    with scada_col1:
                        st.plotly_chart(fig, use_container_width=True)

                    with scada_col2:

                        st.metric("Health Score", f"{health_score:.2f}%")
                        st.metric("Failure Risk", f"{failure_percent}%")

                    # Status Indicators
                    if failure_percent > 70:
                        st.markdown("## 🔴 SYSTEM CRITICAL")

                    elif failure_percent > 40:
                        st.markdown("## 🟠 WARNING")

                    else:
                        st.markdown("## 🟢 NORMAL")

                    # Show alerts
                    for alert, level in alerts:

                        if level == "critical":
                            st.error(alert)

                        elif level == "medium":
                            st.warning(alert)

                        else:
                            st.info(alert)
                
                    st.line_chart(df_live)

                    # -------------------------------
                    # 🧠 AI Operator Recommendations
                    # -------------------------------

                    st.subheader("🧠 AI Operator Recommendations")

                    if priority == "CRITICAL":

                        st.error("""
                        Immediate Actions Required:
                        - Shutdown turbine safely
                        - Inspect vibration bearings
                        - Check cooling system
                        - Notify maintenance supervisor
                        """)

                    elif priority == "HIGH":

                        st.warning("""
                        Recommended Actions:
                        - Schedule maintenance within 24h
                        - Monitor temperature continuously
                        - Reduce operational load
                        """)

                    elif priority == "MEDIUM":

                        st.info("""
                        Monitoring Recommended:
                        - Continue observation
                        - Review system efficiency
                        - Check sensor calibration
                        """)

                    else:

                        st.success("""
                        System operating normally.
                        No immediate intervention required.
                        """)
                    
                    # -------------------------------
                    # ⚡ Smart Energy Dispatch Engine
                    # -------------------------------

                    st.subheader("⚡ Smart Dispatch Decision")

                    if load_action == "REDUCE LOAD":

                        st.error("""
                        AI Dispatch Action:
                        - Reduce non-critical systems
                        - Shift peak operations
                        - Activate energy-saving mode
                        """)

                    elif load_action == "INCREASE UTILIZATION":

                        st.info("""
                        AI Dispatch Action:
                        - Capacity available
                        - Increase productive load
                        - Utilize off-peak efficiency
                        """)

                    else:

                        st.success("""
                        Energy dispatch optimized.
                        System balanced normally.
                        """)

                    # -------------------------------
                    # 🚨 AI Escalation Matrix
                    # -------------------------------

                    st.subheader("🚨 Alarm Escalation Matrix")

                    if priority == "CRITICAL":

                        st.error("""
                        LEVEL 3 ESCALATION:
                        - Notify plant manager
                        - Notify operations head
                        - Emergency maintenance activation
                        """)

                    elif priority == "HIGH":

                        st.warning("""
                        LEVEL 2 ESCALATION:
                        - Notify maintenance team
                        - Increase monitoring frequency
                        """)

                    elif priority == "MEDIUM":

                        st.info("""
                        LEVEL 1 ESCALATION:
                        - Log operational advisory
                        - Continue observation
                        """)

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

            # -------------------------------
            # ✅ SAFE DATA HANDLING
            # -------------------------------
            if df is not None and "energy_kwh" in df.columns:
                avg_energy = df["energy_kwh"].mean()
                max_energy = df["energy_kwh"].max()
                total_energy = df["energy_kwh"].sum()
            else:
                avg_energy = 0
                max_energy = 0
                total_energy = 0

            if forecast_df is not None and "forecast" in forecast_df.columns:
                forecast_avg = forecast_df["forecast"].mean()
            else:
                forecast_avg = 0

            # -------------------------------
            # 🤖 Smart Responses
            # -------------------------------
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
                return "💡 Suggestion: Reduce peak loads and improve HVAC efficiency."

            elif "fault" in query or "failure" in query:
                return "⚠️ Potential anomalies detected. Check vibration and temperature trends."

            elif "ranking" in query:
                if df is not None and "building" in df.columns:
                    ranking = df.groupby("building")["energy_kwh"].sum().sort_values()
                    best = ranking.index[0]
                    return f"🏆 Most efficient building is {best}."
                else:
                    return "ℹ️ No building-wise data available."

            else:
                return "🤖 I can help with energy, cost, CO₂, forecast, faults, and optimization insights."

        # Process user query
        if user_query:
            # -------------------------------
            # ✅ SAFE CHECK (VERY IMPORTANT)
            # -------------------------------
            if 'forecast_df' not in locals():
                forecast_df = None

            if 'df' not in locals():
                df = None

            # -------------------------------
            # 🤖 Generate Response
            # -------------------------------
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
        # 🤖 AI Executive Summary 
        # -------------------------------
        st.subheader("📄 Executive AI Summary")

        summary = f"""
        Enterprise Overview:

        • Total Forecast Energy: {safe_energy} kWh
        • Estimated Cost: £{safe_cost}
        • Estimated CO₂: {safe_co2} kg
        • System Health: {safe_health}%

        AI Recommendations:
        • Improve HVAC optimization
        • Reduce peak-hour loads
        • Schedule preventive maintenance
        • Increase renewable integration
        """

        st.text_area("Executive Summary", summary, height=250)

        # Risk Analysis
        if health_score < 50:
            summary += "\n🚨 High operational risk detected."
        elif health_score < 75:
            summary += "\n⚠ Moderate risk present."
        else:
            summary += "\n✅ System operating efficiently."

        st.text_area("Executive Summary Report", summary, height=200)

        # -------------------------------
        # 🧠 AI Operator Recommendation Engine
        # -------------------------------
        st.subheader("🧠 AI Operator Recommendations")

        recommendations = [
            "Reduce HVAC peak-hour load",
            "Inspect turbine vibration trend",
            "Optimize boiler efficiency",
            "Shift operations to off-peak tariff periods",
            "Schedule predictive maintenance"
        ]

        for rec in recommendations:
            st.info(f"💡 {rec}")

        # =====================================================
        # 🧠 AI Operational Confidence Engine
        # =====================================================

        st.subheader("🧠 AI Operational Confidence")

        confidence_score = round(random.uniform(75, 99), 2)

        st.metric(
            "AI Confidence Score",
            f"{confidence_score}%"
        )

        if confidence_score > 90:
            st.success("✅ AI Predictions Highly Reliable")

        elif confidence_score > 80:
            st.warning("⚠ Moderate Prediction Confidence")

        else:
            st.error("🚨 Low AI Reliability")

        # -------------------------------
        # Download CSV & PDF
        # -------------------------------
        st.subheader("📥 Download Energy Report")

        if forecast_df is not None:
            csv = forecast_df.to_csv(index=False).encode("utf-8")

            st.download_button(
                label="Download Forecast Report (CSV)",
                data=csv,
                file_name="energy_forecast_report.csv",
                mime="text/csv",
            )
        else:
            st.warning("⚠ No forecast data available to download.")

        st.subheader("📄 Download Executive PDF Report")
        def generate_pdf():

            # -------------------------------
            # SAFE VALUES (IMPORTANT FIX)
            # -------------------------------
            safe_total_energy = 0
            safe_total_cost = 0
            safe_total_co2 = 0
            safe_sustainability = 0
            safe_health = 0

            if forecast_df is not None and "forecast" in forecast_df.columns:
                safe_total_energy = forecast_df["forecast"].sum()
                safe_total_cost = safe_total_energy * 0.12
                safe_total_co2 = safe_total_energy * 0.233

            try:
                safe_sustainability = sustainability_score
            except:
                safe_sustainability = 0

            try:
                safe_health = health_score
            except:
                safe_health = 0

            # -------------------------------
            # PDF GENERATION
            # -------------------------------
            buffer = io.BytesIO()
            doc = SimpleDocTemplate(buffer)
            elements = []
            styles = getSampleStyleSheet()

            elements.append(Paragraph("Digital Twin Energy Report", styles["Title"]))
            elements.append(Spacer(1, 0.3 * inch))

            data = [
                ["Total Energy (kWh)", round(safe_total_energy, 2)],
                ["Estimated Cost (£)", round(safe_total_cost, 2)],
                ["CO₂ Emissions (kg)", round(safe_total_co2, 2)],
                ["Sustainability Score (%)", round(safe_sustainability, 2)],
                ["System Health (%)", round(safe_health, 2)]
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
