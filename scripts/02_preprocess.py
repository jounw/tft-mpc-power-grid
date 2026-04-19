"""
Step 2: Preprocess raw data → training arrays

Merges KPX demand, weather, holiday features and generates:
  data/processed/X_train.npy, X_val.npy, X_test.npy
  data/processed/y_train.npy, y_val.npy, y_test.npy
  data/processed/scalers.pkl
  data/processed/test_raw.csv

Usage:
  python scripts/02_preprocess.py
"""
import sys
import yaml
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocessor import run

if __name__ == "__main__":
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
