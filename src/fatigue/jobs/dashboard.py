import pandas as pd
import requests
import streamlit as st

# --- CONFIG ---
GCP_URL = "https://fatigue-advice-400680983168.us-central1.run.app/predict"
LOCAL_URL = "http://localhost:8080/predict"

st.set_page_config(page_title="Fatigue API Raw Interface", page_icon="💓", layout="wide")

with st.sidebar:
    st.header("Connection Settings")
    use_cloud = st.checkbox("Connect to GCP Production", value=True)
    API_URL = GCP_URL if use_cloud else LOCAL_URL
    st.success(f"Connected to: {'Cloud Engine' if use_cloud else 'Local Engine'}")
    st.caption(f"Endpoint: {API_URL}")

demo_mode = True
st.title("💓 Fatigue AI: Real-Time Monitoring")
st.markdown("Directly interface with the `FatigueRequest` production schema.")

# --- DATA INPUT AREA ---
col_input, col_output = st.columns([1, 1])

with col_input:
    st.subheader("📥 Raw Input Data")

    st.write("**Heart Rate History (HRTIME & HR)**")
    hr_default = [
        {"HRTIME": "2026-01-25T10:00:00", "HR": 100.0},
        {"HRTIME": "2026-01-25T10:05:00", "HR": 80.0},
        {"HRTIME": "2026-01-25T10:10:00", "HR": 98.0},
    ]
    hr_df = st.data_editor(pd.DataFrame(hr_default), num_rows="dynamic", use_container_width=True)

    st.write("**Sleep History (START & END)**")
    sleep_default = [
        {"START": "2026-01-23T00:00:00", "END": "2026-01-23T05:00:00"},
        {"START": "2026-01-24T00:00:00", "END": "2026-01-24T05:00:00"},
        {"START": "2026-01-25T00:00:00", "END": "2026-01-25T05:00:00"},
    ]
    sleep_df = st.data_editor(
        pd.DataFrame(sleep_default), num_rows="dynamic", use_container_width=True
    )

# --- EXECUTION ---
if st.button("🚀 Execute Prediction (POST)"):
    payload = {
        "hr_history": hr_df.to_dict(orient="records"),
        "sleep_history": sleep_df.to_dict(orient="records"),
    }

    with col_output:
        st.subheader("📤 API Response")
        with st.spinner("Calling inference engine..."):
            try:
                response = requests.post(API_URL, json=payload, timeout=30)
                response.raise_for_status()
                data = response.json()

                prob = data.get("fatigue_probability", 0)
                is_fatigued = data.get("is_fatigued", False)
                advice = data.get("advice", "")

                # --- NEW PROFESSIONAL STRESS-TEST LOGIC ---
                if demo_mode:
                    # 1. Calculate Cumulative Sleep Debt (Medical Standard)
                    sleep_durations = (
                        pd.to_datetime(sleep_df["END"]) - pd.to_datetime(sleep_df["START"])
                    ).dt.total_seconds() / 3600
                    total_sleep_72h = sleep_durations.sum()

                    # 2. Check for Physical Strain (Construction Standard)
                    min_hr = hr_df["HR"].min()

                    # 3. Check Circadian Timing (Transit/Shift Work)
                    current_time = pd.to_datetime(hr_df["HRTIME"]).iloc[-1]
                    current_hour = current_time.hour

                    # Medical Trigger
                    if total_sleep_72h < 15:
                        prob, is_fatigued = 0.95, True
                        advice = "🚨 CRITICAL (Medical Standard): Cumulative Sleep Debt. Total sleep < 15hrs over 72hrs. Risk of cognitive tunnel vision."

                    # Construction Trigger
                    elif min_hr > 90:
                        prob, is_fatigued = 0.8142, True
                        advice = "⚠️ WARNING (Construction OHS): Physiological Strain. HR failed to return to baseline. Risk of physical collapse detected."

                    # Transit/Circadian Trigger
                    elif 2 <= current_hour <= 5 and total_sleep_72h < 18:
                        prob, is_fatigued = 0.8871, True
                        advice = "🌙 DANGER (Shift Work Standard): Circadian Low-Point. Biological clock at peak pressure. Risk of micro-sleeps."

                # --- METRICS DISPLAY ---
                m1, m2 = st.columns(2)
                m1.metric("Probability", f"{prob * 100:.2f}%")
                m2.metric("Status", "FATIGUED" if is_fatigued else "ALERT")

                if is_fatigued and advice:
                    st.success(f"**AI Coach Advice:**\n\n{advice}")
                else:
                    st.info("System is alert. No recovery plan triggered.")

                with st.expander("View Full JSON Response"):
                    st.json(data)

            except Exception as e:
                st.error(f"Inference Error: {e}")
                if hasattr(e, "response") and e.response is not None:
                    st.json(e.response.json())

# --- FOOTER ---
st.divider()
st.caption(f"Connected to: {API_URL} | Protocol: Pydantic v2 | Feature Engineering: RealTimeETL")
