"""
Step 3: Train demand forecasting models and compare

Trains three models on 2014-2023 KPX Honam demand data:
  - BiLSTM + Attention
  - Temporal Fusion Transformer (TFT)  ← main model
  - VMD + CNN-LSTM

Saves best model to models/demand_tft_final.pt
Generates comparison figures in logs/

Usage:
  python scripts/03_train_tft.py
"""
import sys
import argparse
import yaml
import numpy as np
import joblib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.demand_forecaster import train as train_model, predict, evaluate, load_model
from src.forecasting.vmd_cnn_lstm import augment_with_vmd, train as train_vmd, predict as predict_vmd


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["lstm", "tft", "vmd", "all"], default="all",
                        help="Model to train (default: all)")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    args = parser.parse_args()

    cfg = load_config()
    proc = ROOT / cfg["data"]["processed_dir"]

    X_train = np.load(proc / "X_train.npy")
    y_train = np.load(proc / "y_train.npy")
    X_val   = np.load(proc / "X_val.npy")
    y_val   = np.load(proc / "y_val.npy")
    X_test  = np.load(proc / "X_test.npy")
    y_test  = np.load(proc / "y_test.npy")

    scalers       = joblib.load(proc / "scalers.pkl")
    demand_scaler = scalers.get("demand")

    print(f"Train: {X_train.shape}  Val: {X_val.shape}  Test: {X_test.shape}")

    models_dir = str(ROOT / "models")
    common = dict(
        X_train=X_train, y_train=y_train,
        X_val=X_val,     y_val=y_val,
        save_dir=models_dir,
        hidden_size=256,
        num_layers=3,
        epochs=args.epochs,
        batch_size=128,
        lr=5e-4,
        patience=args.patience,
    )

    results = {}

    if args.model in ("lstm", "all"):
        print("\n--- BiLSTM + Attention ---")
        model = train_model(**common, model_type="lstm", save_name="demand_lstm_final.pt")
        pred  = predict(model, X_test)
        m     = evaluate(y_test, pred, scaler=demand_scaler)
        results["BiLSTM+Attn"] = m
        print(f"  MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  MAPE={m['MAPE(%)']:.4f}%  R²={m['R2']:.4f}")

    if args.model in ("tft", "all"):
        print("\n--- Temporal Fusion Transformer ---")
        model = train_model(**common, model_type="tft", save_name="demand_tft_final.pt")
        pred  = predict(model, X_test)
        m     = evaluate(y_test, pred, scaler=demand_scaler)
        results["TFT"] = m
        print(f"  MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  MAPE={m['MAPE(%)']:.4f}%  R²={m['R2']:.4f}")

    if args.model in ("vmd", "all"):
        print("\n--- VMD + CNN-LSTM ---")
        X_tr_v, X_va_v, X_te_v = augment_with_vmd(X_train, X_val, X_test, demand_col_idx=0, K=5)
        model = train_vmd(
            X_tr_v, y_train, X_va_v, y_val,
            save_dir=models_dir,
            save_name="demand_vmd_final.pt",
            cnn_channels=32,
            kernel_sizes=(3, 7, 14, 24),
            lstm_hidden=256,
            lstm_layers=2,
            epochs=args.epochs,
            batch_size=128,
            lr=5e-4,
            patience=args.patience,
        )
        pred = predict_vmd(model, X_te_v)
        m    = evaluate(y_test, pred, scaler=demand_scaler)
        results["VMD+CNN-LSTM"] = m
        print(f"  MAE={m['MAE']:.1f}  RMSE={m['RMSE']:.1f}  MAPE={m['MAPE(%)']:.4f}%  R²={m['R2']:.4f}")

    if len(results) > 1:
        print("\n" + "=" * 50)
        print("Model Comparison")
        best = min(results.items(), key=lambda x: x[1]["MAPE(%)"])
        for name, m in results.items():
            mark = " ← best" if name == best[0] else ""
            print(f"  {name:16s}  MAPE={m['MAPE(%)']:.4f}%{mark}")
        print(f"\nBest model: {best[0]} saved to {models_dir}/")


if __name__ == "__main__":
    main()
