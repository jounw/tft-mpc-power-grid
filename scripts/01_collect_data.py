"""
Step 1: Collect raw data (weather, holiday, KPX power demand)

Usage:
  python scripts/01_collect_data.py --step all
  python scripts/01_collect_data.py --step weather
  python scripts/01_collect_data.py --step holiday
  python scripts/01_collect_data.py --step kpx
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data_collection.collect_all import main

if __name__ == "__main__":
    main()
