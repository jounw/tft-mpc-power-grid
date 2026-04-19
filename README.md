# TFT-MPC Power Grid Optimization
## An Integrated TFT–MPC Framework for Optimal ESS and LNG Dispatch under Grid Congestion

> **Case Study: Honam Region, South Korea**

---

### Overview

This repository implements a two-stage operational framework for the Honam power system:

1. **Demand Forecasting** — Temporal Fusion Transformer (TFT) trained on hourly KPX data (2014–2023) with weather and calendar features. Compared against BiLSTM+Attention and VMD+CNN-LSTM baselines.

2. **MPC Optimization** — Rolling-horizon Model Predictive Control dispatches ESS (5,600 MWh / 1,400 MW), LNG (420–2,100 MW), and grid import/export every hour using TFT demand forecasts to minimize operating cost and curtailment.

#### Key Results (2024 test set vs. rule-based baseline)

| Metric | Rule-Based | MPC-TFT | Δ |
|---|---|---|---|
| Total operating cost | baseline | −8.17% | ✓ |
| Grid export | baseline | +16.5% | ✓ |
| Daytime grid import | baseline | −2.2% | ✓ |
| LNG generation | baseline | −2.9% | ✓ |
| Peak renewable curtailment | baseline | up to −17% | ✓ |

---

### Repository Structure

```
├── src/
│   ├── data_collection/       # Weather (KMA API), holiday, KPX demand collectors
│   ├── preprocessing/         # Feature engineering, train/val/test split
│   ├── forecasting/
│   │   ├── demand_forecaster.py   # BiLSTM+Attn & TFT models
│   │   ├── peak_corrector.py      # Peak-hour post-processing
│   │   └── vmd_cnn_lstm.py        # VMD+CNN-LSTM baseline
│   └── optimization/
│       └── mpc.py                 # Rolling-horizon LP solver (HiGHS)
├── scripts/
│   ├── 01_collect_data.py     # Step 1: collect raw data
│   ├── 02_preprocess.py       # Step 2: build training arrays
│   ├── 03_train_tft.py        # Step 3: train & compare forecast models
│   └── 04_run_mpc.py          # Step 4: MPC-TFT simulation & comparison
├── data/
│   ├── raw/                   # KPX CSV, weather, holiday (see below)
│   └── processed/             # test_raw.csv, scalers.pkl (X/y arrays excluded)
├── configs/
│   └── config.yaml            # Data paths, date ranges, API keys (env vars)
├── models/                    # Saved .pt weights (git-ignored, see below)
└── logs/                      # Simulation outputs
```

---

### Quick Start

#### 1. Install dependencies

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

#### 2. Configure API keys

```bash
export KMA_API_KEY="your_kma_key"          # 기상청 공공데이터포털
export DATA_GO_KR_API_KEY="your_key"       # 공공데이터포털 (공휴일)
```

#### 3. Run the pipeline

```bash
# Step 1 – collect raw data (skip if using provided data/raw/)
python scripts/01_collect_data.py --step all

# Step 2 – preprocess (generates X_train/val/test.npy)
python scripts/02_preprocess.py

# Step 3 – train demand forecast models
python scripts/03_train_tft.py              # train all 3 models
python scripts/03_train_tft.py --model tft  # TFT only

# Step 4 – run MPC simulation
python scripts/04_run_mpc.py
```

---

### Data

| File | Size | Description |
|---|---|---|
| `data/raw/kpx/kpx_honam.csv` | 2.3 MB | KPX hourly demand/solar/wind, Honam region |
| `data/raw/weather/weather_광주.csv` | 151 KB | KMA weather (Gwangju station 156) |
| `data/raw/holiday/holidays.csv` | 5 KB | Korean public holidays 2014–2024 |
| `data/processed/test_raw.csv` | 2.4 MB | 2024 test features (merged) |
| `data/processed/scalers.pkl` | 8 KB | Fitted StandardScaler objects |

**Note:** Training numpy arrays (`X_train.npy`, `X_val.npy`, `X_test.npy`) exceed GitHub's 100 MB limit. Regenerate them with `python scripts/02_preprocess.py` after placing raw data files.

KPX raw source files (annual CSVs) can be downloaded from [EPSIS](https://epsis.kpx.or.kr) → 전력통계 → 지역별 시간대별 전력거래량.

---

### Model Details

#### TFT (Temporal Fusion Transformer)

- Input window: 168 h (7 days)
- Forecast horizon: 24 h
- Features: demand, temperature, humidity, wind speed, solar radiation, rainfall, hour/day/month cyclic encodings, holiday flag
- Hidden size: 256, Layers: 3
- Loss: Peak-Weighted Huber (×2.5 for peak hours 9–11, 18–21)

| Model | MAE (MWh) | RMSE (MWh) | MAPE (%) | R² |
|---|---|---|---|---|
| BiLSTM + Attn | 482 | 613 | 5.504 | 0.663 |
| **TFT** | **386** | **500** | **4.497** | **0.776** |
| VMD+CNN-LSTM | 413 | 527 | 4.793 | 0.751 |

#### MPC Formulation

Minimize: `Σ (Cost_LNG + Cost_ESS + Cost_Import − Revenue_Export)`

Subject to:
- Power balance: nuclear + coal + LNG + solar + wind ± ESS ± grid = demand
- LNG: 420–2,100 MW, ramp ≤ 500 MW/h
- ESS: 5,600 MWh capacity, 95% round-trip efficiency, SoC 10–90%
- Grid: import ≤ 3,000 MW (daytime 1,600 MW), export ≤ 3,000 MW
- Solver: HiGHS LP (`scipy.optimize.linprog`)

---

### References

1. Lim, B. et al. (2021). Temporal Fusion Transformers for interpretable multi-horizon time series forecasting. *Int. J. Forecast.*, 37(4), 1748–1764.
2. Dragomiretskiy, K. & Zosso, D. (2014). Variational Mode Decomposition. *IEEE Trans. Signal Processing*, 62(3), 531–544.
