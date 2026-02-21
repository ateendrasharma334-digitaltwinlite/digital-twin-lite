import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor


# -------------------------------------------------
# ML-Based Energy Forecast Function
# -------------------------------------------------
def forecast_energy(days):
    """
    Train a RandomForest model on synthetic historical data
    and predict future energy consumption.
    """

    # -------------------------------
    # 1. Generate Historical Data
    # -------------------------------
    historical_days = 200

    dates = pd.date_range(
        end=datetime.today(),
        periods=historical_days
    )

    # Simulated realistic energy pattern
    energy = (
        100
        + np.sin(np.linspace(0, 20, historical_days)) * 10
        + np.random.normal(0, 5, historical_days)
    )

    df_hist = pd.DataFrame({
        "date": dates,
        "energy": energy
    })

    # -------------------------------
    # 2. Feature Engineering
    # -------------------------------
    df_hist["day"] = df_hist["date"].dt.day
    df_hist["month"] = df_hist["date"].dt.month
    df_hist["weekday"] = df_hist["date"].dt.weekday

    X = df_hist[["day", "month", "weekday"]]
    y = df_hist["energy"]

    # -------------------------------
    # 3. Train ML Model
    # -------------------------------
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    # -------------------------------
    # 4. Generate Future Dates
    # -------------------------------
    future_dates = pd.date_range(
        start=datetime.today(),
        periods=days
    )

    df_future = pd.DataFrame({
        "date": future_dates
    })

    df_future["day"] = df_future["date"].dt.day
    df_future["month"] = df_future["date"].dt.month
    df_future["weekday"] = df_future["date"].dt.weekday

    # -------------------------------
    # 5. Predict Future Energy
    # -------------------------------
    predictions = model.predict(
        df_future[["day", "month", "weekday"]]
    )

    df_future["forecast"] = predictions

    return df_future[["date", "forecast"]]


# -------------------------------------------------
# Anomaly Detection Function
# -------------------------------------------------
def detect_anomalies(df):
    """
    Detect anomalies based on statistical threshold.
    """
    df = df.copy()

    threshold = df["forecast"].mean() + 2 * df["forecast"].std()

    df["anomaly"] = df["forecast"] > threshold

    return df