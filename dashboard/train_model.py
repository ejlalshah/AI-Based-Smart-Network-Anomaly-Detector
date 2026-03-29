import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import joblib
import os

INPUT_FILE = "data/features.csv"
MODEL_FILE = "ml/isolation_forest_model.pkl"
SCALER_FILE = "ml/scaler.pkl"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("features.csv not found. Run feature_extraction.py first.")

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("features.csv is empty.")

X = df[
    ["packet_count", "avg_packet_size", "max_packet_size", "protocol_count"]
]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

model = IsolationForest(
    n_estimators=100,
    contamination=0.05,
    random_state=42
)
model.fit(X_scaled)

os.makedirs("ml", exist_ok=True)
joblib.dump(model, MODEL_FILE)
joblib.dump(scaler, SCALER_FILE)

print("Model training complete.")
