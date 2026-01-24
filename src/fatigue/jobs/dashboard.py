import pandas as pd
import requests
import streamlit as st

# --- CONFIG ---
API_URL = "http://localhost:8080/predict"

st.set_page_config(page_title="Fatigue API Raw Interface", page_icon="💓", layout="wide")

st.title("💓 Fatigue AI: Raw Data Injector")
st.markdown("Directly interface with the `FatigueRequest` schema. No filters, no mocks.")

# --- DATA INPUT AREA ---
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("📥 Raw Input Data")

    # 1. Heart Rate History Input
    st.write("**Heart Rate History (HRTIME & HR)**")
    hr_default = [
        {"HRTIME": "2026-01-24T10:00:00", "HR": 100.0},
        {"HRTIME": "2026-01-24T10:05:00", "HR": 80.0},
        {"HRTIME": "2026-01-24T10:10:00", "HR": 98.0},
    ]
    # Using data_editor allows you to add/delete rows like an Excel sheet
    hr_df = st.data_editor(pd.DataFrame(hr_default), num_rows="dynamic", use_container_width=True)

    # 2. Sleep History Input
    st.write("**Sleep History (START & END)**")
    sleep_default = [
        {"START": "2026-01-21T00:00:00", "END": "2026-01-21T05:00:00"},
        {"START": "2026-01-22T00:00:00", "END": "2026-01-22T05:00:00"},
        {"START": "2026-01-23T00:00:00", "END": "2026-01-23T05:00:00"},
        {"START": "2026-01-24T00:00:00", "END": "2026-01-24T05:00:00"},
    ]
    sleep_df = st.data_editor(
        pd.DataFrame(sleep_default), num_rows="dynamic", use_container_width=True
    )

# --- EXECUTION ---
if st.button("🚀 Execute Prediction (POST)"):
    # Convert DataFrames back to the list of dicts format required by the API
    payload = {
        "hr_history": hr_df.to_dict(orient="records"),
        "sleep_history": sleep_df.to_dict(orient="records"),
    }

    with col_output:
        st.subheader("📤 API Response")
        with st.spinner("Calling local inference engine..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=10)
                response.raise_for_status()
                data = response.json()

                # --- METRICS ---
                prob = data.get("fatigue_probability", 0)
                m1, m2 = st.columns(2)
                m1.metric("Probability", f"{prob * 100:.2f}%")
                m2.metric("Status", "FATIGUED" if data.get("is_fatigued") else "ALERT")

                # --- AI COACH BOX ---
                if data.get("advice"):
                    st.success(f"**AI Coach Advice:**\n\n{data['advice']}")
                else:
                    st.info("System is alert. No recovery plan triggered.")

                # --- RAW JSON INSPECTOR ---
                with st.expander("View Full JSON Response"):
                    st.json(data)

            except Exception as e:
                st.error(f"Inference Error: {e}")
                if hasattr(e, "response") and e.response is not None:
                    st.json(e.response.json())

# --- FOOTER: SYSTEM STATUS ---
st.divider()
st.caption(f"Connected to: {API_URL} | Protocol: Pydantic v2 | Feature Engineering: RealTimeETL")
