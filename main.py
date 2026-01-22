# main.py
import logging
from typing import List

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fatigue.coach import FatigueCoach

# 1. Import your existing ETL logic
from fatigue.jobs.inference_etl import InferenceETLJob

# Initialize Logging
logging.basicConfig(level=logging.INFO)
log = logging.getLogger("gcp-api")

app = FastAPI()

# Global variables for model and ETL
model = None
etl_job = None
coach = None


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
        # Force timezone-naive to prevent merge errors
        df_hr_raw["HRTIME"] = pd.to_datetime(df_hr_raw["HRTIME"]).dt.tz_localize(None)
        df_hr_raw = df_hr_raw.sort_values("HRTIME")
        min_hr, max_hr = self.settings.hr_limits
        df_hr_raw = df_hr_raw[(df_hr_raw["HR"] > min_hr) & (df_hr_raw["HR"] < max_hr)]
        # Group duplicates if any
        df_hr = df_hr_raw.groupby("HRTIME", as_index=False)[["HR"]].mean()

        # C. Clean Sleep (Replaces _clean_sleep)
        df_sleep_raw["START"] = pd.to_datetime(df_sleep_raw["START"]).dt.tz_localize(None)
        df_sleep_raw["END"] = pd.to_datetime(df_sleep_raw["END"]).dt.tz_localize(None)
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
            log.warning("ETL logic failed to produce features. Using safety fallback.")
            return pd.DataFrame(
                [
                    {
                        "mean_hr_5min": df_hr["HR"].mean(),
                        "hr_volatility_5min": 0.5,
                        "hr_mean_total": df_hr["HR"].mean(),
                        "hr_std_total": 0.5,
                        "hr_zscore": 0.0,
                        "hr_jumpiness_5min": 0.0,
                        "stress_cv": 0.1,
                        "hours_awake": 16.0,
                        "cum_sleep_debt": 2.0,
                        "sleep_inertia_idx": 0.1,
                        "circadian_sin": 0.0,
                        "circadian_cos": 0.0,
                    }
                ]
            )

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

        # If any rolling features are missing, use a safe default
        # so the prediction doesn't crash or return empty.
        final_features = latest_row[required_features].fillna(0).astype(float)
        log.info(f"Feature Vector Produced: {final_features.to_dict()}")
        return final_features


# --- 4. API Endpoints ---
@app.on_event("startup")
def load_artifacts():
    global model, etl_job, coach
    try:
        # Load your trained model
        model = joblib.load("python_model.pkl")
        log.info("Model loaded successfully.")
        # Initialize the Helper Class
        etl_job = RealTimeETL()
        log.info("GCP Unified System: Model and ETL")

        coach = FatigueCoach()
        log.info("AI Coach (Llama 3 + Embeddings) initialized.")
    except Exception as e:
        log.error(f"Failed to load artifacts: {e}")


@app.get("/health")
def health():
    return {"status": "active"}


@app.post("/predict")
def predict_fatigue(payload: FatigueRequest):
    if not model or not coach:
        raise HTTPException(status_code=500, detail="Model not loaded")

    try:
        # 1. Preprocess & Feature Engineer (In-Memory)
        features_df = etl_job.process_request(payload.hr_history, payload.sleep_history)

        # 2. RF Prediction
        prediction = model.predict(model_input=features_df)
        proba = (
            float(prediction[0])
            if isinstance(prediction, (np.ndarray, list))
            else float(prediction)
        )

        response = {
            "fatigue_probability": float(proba),
            "is_fatigued": bool(proba > 0.7),
            "timestamp": payload.hr_history[-1].HRTIME,
            "advice": None,
        }

        # 3. THE CUT-OFF LOGIC
        if proba > 0.7:
            # Pass the in-memory features directly to the coach
            context_dict = features_df.to_dict(orient="records")[0]
            response["advice"] = coach.get_advice(proba, context_dict)

        return response

    except Exception as e:
        log.error(f"Error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
