"""
Step 5: Pre-compute TFT demand forecasts for all test days

Generates data/processed/tft_forecasts.csv — required by the Streamlit demo app.
Run this once after training (needs data/processed/X_test.npy locally).

Usage:
  python scripts/05_precompute_forecasts.py
"""
import sys
import numpy as np
import pandas as pd
import joblib
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.demand_forecaster import predict, load_model

if __name__ == "__main__":
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc = ROOT / cfg["data"]["processed_dir"]

    X_test = np.load(proc / "X_test.npy")
    scalers = joblib.load(proc / "scalers.pkl")
    demand_scaler = scalers["demand"]

    tft = load_model(str(ROOT / "models" / "demand_tft_final.pt"),
                     input_size=X_test.shape[2], model_type="tft",
                     hidden_size=256, num_layers=3)
    print(f"TFT loaded  input_size={X_test.shape[2]}")

    y_norm = predict(tft, X_test)
    y_mwh  = np.clip(
        demand_scaler.inverse_transform(y_norm.reshape(-1, 1)).reshape(y_norm.shape),
        0, None
    )

    test_raw = pd.read_csv(proc / "test_raw.csv", parse_dates=["datetime"])
    seq_len  = 168

    records = []
    for i, fc in enumerate(y_mwh):
        pred_start = seq_len + i
        if pred_start >= len(test_raw):
            break
        dt = test_raw.iloc[pred_start]["datetime"]
        if dt.hour == 0:
            row = {"date": dt.strftime("%Y-%m-%d")}
            for h in range(24):
                row[f"h{h:02d}"] = float(fc[h])
            records.append(row)

    out = pd.DataFrame(records)
    out.to_csv(proc / "tft_forecasts.csv", index=False)
    print(f"Saved {len(out)} days → {proc}/tft_forecasts.csv")
