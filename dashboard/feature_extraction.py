import pandas as pd
import os

INPUT_FILE = "data/packets.csv"
OUTPUT_FILE = "data/features.csv"

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError("packets.csv not found. Run packet_capture.py first.")

df = pd.read_csv(INPUT_FILE)

if df.empty:
    raise ValueError("packets.csv is empty. Capture some traffic first.")

df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%Y-%m-%d %H-%M-%S",
    errors="coerce"
)

features = df.groupby("src_ip").agg(
    packet_count=("packet_size", "count"),
    avg_packet_size=("packet_size", "mean"),
    max_packet_size=("packet_size", "max"),
    protocol_count=("protocol", "nunique")
).reset_index()

os.makedirs("data", exist_ok=True)
features.to_csv(OUTPUT_FILE, index=False)

print("Feature extraction complete.")
