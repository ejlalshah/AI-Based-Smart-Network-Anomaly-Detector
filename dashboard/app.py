import streamlit as st

st.set_page_config(
    page_title="Smart Network Anomaly Detector",
    page_icon="🚨",
    layout="wide"
)

st.title("🚨 Smart Network Anomaly Detector")

st.markdown("""
Welcome to the **Smart Network Anomaly Detection System**.

Use the sidebar to explore:
- 📊 Network Overview
- 🚨 Detected Anomalies
- 🧠 Model Information
""")

st.info("Unsupervised ML-based network anomaly detection system.")
