# main.py
import logging
from typing import List

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 1. Import your existing ETL logic
from fatigue.jobs.inference_etl import InferenceETLJob

# Initialize Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("api")

app = FastAPI()

# Global variables for model and ETL
model = None
etl_job = None


# --- 2. Define Input Schemas (What the Bluetooth/App sends) ---
class HRDataPoint(BaseModel):
    HRTIME: str  # ISO format: "2026-01-18T10:00:00"
    HR: float


class SleepDataPoint(BaseModel):
    START: str
    END: str


class FatigueRequest(BaseModel):
    hr_history: List[HRDataPoint]  # Need ~30 mins of history for rolling features
    sleep_history: List[SleepDataPoint]  # Need recent sleep logs


# --- 3. The Adapter Class (The Magic Glue) ---
class RealTimeETL(InferenceETLJob):
    """
    Inherits everything from InferenceETLJob but overrides
    file-reading methods to work with API JSON data.
    """

    def __init__(self):
        # Initialize parent with default settings
        super().__init__()

    def process_request(self, hr_list, sleep_list):
        """
        Main entry point for API data.
        """
        # A. Convert JSON to DataFrames
        df_hr_raw = pd.DataFrame([h.dict() for h in hr_list])
        df_sleep_raw = pd.DataFrame([s.dict() for s in sleep_list])

        # B. Clean HR (Replaces _clean_hr)
        # We manually apply the cleaning logic here since _clean_hr reads CSVs
        df_hr_raw["HRTIME"] = pd.to_datetime(df_hr_raw["HRTIME"])
        df_hr_raw = df_hr_raw.sort_values("HRTIME")
        min_hr, max_hr = self.settings.hr_limits
        df_hr_raw = df_hr_raw[(df_hr_raw["HR"] > min_hr) & (df_hr_raw["HR"] < max_hr)]
        # Group duplicates if any
        df_hr = df_hr_raw.groupby("HRTIME", as_index=False)[["HR"]].mean()

        # C. Clean Sleep (Replaces _clean_sleep)
        df_sleep_raw["START"] = pd.to_datetime(df_sleep_raw["START"])
        df_sleep_raw["END"] = pd.to_datetime(df_sleep_raw["END"])
        df_sleep_raw["duration"] = (
            df_sleep_raw["END"] - df_sleep_raw["START"]
        ).dt.total_seconds() / 3600
        min_s, max_s = self.settings.sleep_limits
        df_sleep = df_sleep_raw[
            (df_sleep_raw["duration"] > min_s) & (df_sleep_raw["duration"] < max_s)
        ]
        df_sleep = df_sleep.sort_values("END")

        # D. Engineer Features (REUSE your existing complex logic!)
        # These methods accept DataFrames, so they work perfectly.
        df_hr_eng = self._engineer_hr(df_hr)
        if df_hr_eng is None or df_hr_eng.empty:
            raise ValueError("Not enough HR data to calculate rolling features (need >30 mins)")

        df_sleep_eng = self._engineer_sleep(df_sleep)
        df_sleep_eng = df_sleep_eng.rename(columns={"END": "last_sleep_end"})

        # E. Merge (Logic copied from _process_user)
        df_continuous = pd.merge_asof(
            df_hr_eng.sort_values("HRTIME"),
            df_sleep_eng.sort_values("last_sleep_end"),
            left_on="HRTIME",
            right_on="last_sleep_end",
            direction="backward",
        )

        # F. Final Calculations
        df_continuous["hours_awake"] = (
            df_continuous["HRTIME"] - df_continuous["last_sleep_end"]
        ).dt.total_seconds() / 3600

        # Impute default if no sleep data found
        df_continuous["hours_awake"] = df_continuous["hours_awake"].fillna(14.0)

        # Add Circadian & Inertia
        df_continuous["sleep_inertia_idx"] = 1 / (df_continuous["hours_awake"] + 0.1)
        hr_hour = df_continuous["HRTIME"].dt.hour + (df_continuous["HRTIME"].dt.minute / 60)
        import numpy as np

        df_continuous["circadian_sin"] = np.sin(2 * np.pi * hr_hour / 24)
        df_continuous["circadian_cos"] = np.cos(2 * np.pi * hr_hour / 24)

        # G. Select Final Columns (Must match training EXACTLY)
        # Note: We take the LAST row (the most recent moment) for prediction
        latest_row = df_continuous.iloc[[-1]].copy()

        required_features = [
            "mean_hr_5min",
            "hr_volatility_5min",
            "hr_mean_total",
            "hr_std_total",
            "hr_zscore",
            "hr_jumpiness_5min",
            "stress_cv",
            "hours_awake",
            "cum_sleep_debt",
            "sleep_inertia_idx",
            "circadian_sin",
            "circadian_cos",
        ]

        # Ensure only valid columns exist
        final_features = latest_row[required_features]
        return final_features


# --- 4. API Endpoints ---
@app.on_event("startup")
def load_artifacts():
    global model, etl_job
    try:
        # Load your trained model
        model = joblib.load("python_model.pkl")
        # Initialize the Helper Class
        etl_job = RealTimeETL()
        log.info("Model and ETL loaded successfully.")
    except Exception as e:
        log.error(f"Failed to load artifacts: {e}")
        # Only for debugging locally, remove in prod
        model = "dummy"
        etl_job = RealTimeETL()


@app.get("/health")
def health():
    return {"status": "active"}


@app.post("/predict")
def predict_fatigue(payload: FatigueRequest):
    if not model:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 1. Run the ETL Pipeline
        features_df = etl_job.process_request(payload.hr_history, payload.sleep_history)

        # 2. Make Prediction
        if model == "dummy":
            prediction = 0.5
            proba = 0.5
        else:
            prediction = model.predict(features_df)[0]
            # If your model supports probabilities (e.g., Random Forest)
            proba = (
                model.predict_proba(features_df)[0][1] if hasattr(model, "predict_proba") else 0.0
            )

        return {
            "fatigue_prediction": int(prediction),
            "fatigue_probability": float(proba),
            "timestamp": payload.hr_history[-1].HRTIME,
            "features_used": features_df.to_dict(orient="records")[0],
        }

    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        log.error(f"Prediction Error: {e}")
        raise HTTPException(status_code=500, detail="Internal processing error")
