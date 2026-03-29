import pandas as pd
import joblib
import os

FEATURES_FILE = "data/features.csv"
MODEL_FILE = "ml/isolation_forest_model.pkl"
SCALER_FILE = "ml/scaler.pkl"
OUTPUT_FILE = "data/anomalies.csv"

if not os.path.exists(FEATURES_FILE):
    raise FileNotFoundError("features.csv missing.")

df = pd.read_csv(FEATURES_FILE)

model = joblib.load(MODEL_FILE)
scaler = joblib.load(SCALER_FILE)

X = df[
    ["packet_count", "avg_packet_size", "max_packet_size", "protocol_count"]
]

X_scaled = scaler.transform(X)

df["anomaly"] = model.predict(X_scaled)
df["anomaly_score"] = model.decision_function(X_scaled)

os.makedirs("data", exist_ok=True)
df.to_csv(OUTPUT_FILE, index=False)

print("Anomaly detection complete.")
