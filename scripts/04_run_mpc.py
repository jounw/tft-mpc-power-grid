"""
Step 4: Run MPC-TFT simulation and compare against rule-based baseline

Runs rolling-horizon MPC optimization over the 2024 test set using TFT demand
forecasts, then compares against a rule-based ESS dispatch baseline.

Outputs:
  logs/mpc_results.csv      — hourly dispatch for every simulated day
  logs/mpc_summary.csv      — daily aggregates
  logs/comparison_table.txt — printed comparison vs rule-based

Usage:
  python scripts/04_run_mpc.py
  python scripts/04_run_mpc.py --days 30   # limit to first N days
"""
import sys
import argparse
import warnings
import yaml
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.forecasting.demand_forecaster import predict, load_model
from src.optimization.mpc import (
    run_rolling_mpc,
    NUCLEAR, COAL, LNG_MIN, LNG_CAPACITY, ESS_CAPACITY, ESS_MAX_PWR, ESS_EFF,
    SOC_MIN, SOC_MAX, SOC_INIT, EXP_CAP, IMP_CAP,
)

# ── System Marginal Price profile ──────────────────────────────────────────
_BASE_SMP = 176.0  # KRW/kWh average
_SEASON_MULT = {12: 1.25, 1: 1.25, 2: 1.25, 6: 1.30, 7: 1.30, 8: 1.30,
                3: 1.00, 4: 1.00, 5: 1.00, 9: 1.00, 10: 1.00, 11: 1.00}


def make_price_profile(month: int, seed: int = 0):
    rng = np.random.default_rng(seed)
    mult = _SEASON_MULT.get(month, 1.0)
    noise = rng.uniform(0.92, 1.08, 24).astype(np.float32)
    imp = _BASE_SMP * mult * noise
    exp = imp * 0.95
    return imp, exp


def sim_rule_based(profile, init_soc: float = SOC_INIT):
    """Simple rule-based ESS dispatch: charge when surplus, discharge when deficit."""
    dm = np.asarray(profile["demand"], dtype=float)
    so = np.asarray(profile["solar"], dtype=float)
    wi = np.asarray(profile["wind"], dtype=float)

    soc = init_soc
    rows = []
    for t in range(24):
        fixed = NUCLEAR + COAL + so[t] + wi[t]
        bal = fixed - dm[t]
        chg = dis = 0.0
        if bal > 0:
            chg = max(0.0, min(ESS_MAX_PWR, bal, (SOC_MAX - soc) * ESS_CAPACITY / ESS_EFF))
            soc += chg * ESS_EFF / ESS_CAPACITY
        elif bal < 0:
            dis = max(0.0, min(ESS_MAX_PWR, -bal, (soc - SOC_MIN) * ESS_CAPACITY * ESS_EFF))
            soc -= dis / (ESS_EFF * ESS_CAPACITY)
        soc = float(np.clip(soc, SOC_MIN, SOC_MAX))

        ess_net = chg - dis * ESS_EFF
        lng = float(np.clip(dm[t] - (fixed - ess_net), LNG_MIN, LNG_CAPACITY))
        total = fixed + lng - ess_net
        surplus = max(0.0, total - dm[t])
        export = min(EXP_CAP, surplus)
        deficit = max(0.0, dm[t] - (total - export))
        grid_import = min(deficit, IMP_CAP)
        curtailment = max(0.0, (total - export) - dm[t] - grid_import)
        rows.append(dict(lng=lng, grid_import=grid_import, export_mwh=export,
                         curtailment=curtailment, soc=soc))
    return pd.DataFrame(rows), soc


def build_tft_forecasts(proc: Path, cfg: dict) -> dict:
    """Load TFT model and generate 24h ahead demand forecasts for test days."""
    X_test = np.load(proc / "X_test.npy")
    scalers = joblib.load(proc / "scalers.pkl")
    demand_scaler = scalers["demand"]

    model_path = ROOT / "models" / "demand_tft_final.pt"
    tft = load_model(str(model_path), input_size=X_test.shape[2],
                     model_type="tft", hidden_size=256, num_layers=3)
    print(f"TFT loaded: {model_path}  input_size={X_test.shape[2]}")

    y_norm = predict(tft, X_test)
    y_mwh  = np.clip(demand_scaler.inverse_transform(y_norm.reshape(-1, 1)).reshape(y_norm.shape), 0, None)

    test_raw = pd.read_csv(proc / "test_raw.csv", parse_dates=["datetime"])
    seq_len  = 168
    date_forecast: dict = {}
    for i, fc in enumerate(y_mwh):
        pred_start = seq_len + i
        if pred_start >= len(test_raw):
            break
        dt = test_raw.iloc[pred_start]["datetime"]
        if dt.hour == 0:
            date_forecast[dt.strftime("%Y-%m-%d")] = fc
    print(f"TFT forecast days: {len(date_forecast)}")
    return date_forecast


def build_day_profiles(test_csv: str) -> list:
    """Build daily demand/solar/wind profiles from test_raw.csv."""
    df = pd.read_csv(test_csv, parse_dates=["datetime"]).set_index("datetime").sort_index()
    profiles = []
    for date in df.index.normalize().unique():
        date_str = date.strftime("%Y-%m-%d")
        try:
            day = df.loc[date_str].sort_index()
        except KeyError:
            continue
        if len(day) < 24:
            continue
        profiles.append({
            "date": date,
            "demand": day["demand"].values[:24].astype(np.float32),
            "solar":  day.get("solar",  pd.Series(np.zeros(24))).values[:24].astype(np.float32),
            "wind":   day.get("wind",   pd.Series(np.zeros(24))).values[:24].astype(np.float32),
        })
    return profiles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=None, help="Limit to first N days")
    args = parser.parse_args()

    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    proc = ROOT / cfg["data"]["processed_dir"]
    logs = ROOT / "logs"
    logs.mkdir(exist_ok=True)

    date_forecast = build_tft_forecasts(proc, cfg)
    profiles = build_day_profiles(str(proc / "test_raw.csv"))
    profiles = [p for p in profiles if p["date"].strftime("%Y-%m-%d") in date_forecast]
    if args.days:
        profiles = profiles[:args.days]
    print(f"Simulating {len(profiles)} days...")

    mpc_records, rule_records = [], []
    soc_mpc = SOC_INIT
    soc_rule = SOC_INIT

    for idx, profile in enumerate(profiles):
        date_str = profile["date"].strftime("%Y-%m-%d")
        fc = date_forecast[date_str]
        imp_p, exp_p = make_price_profile(profile["date"].month, seed=idx)

        # MPC-TFT
        try:
            df_mpc = run_rolling_mpc(
                profile, fc.tolist(), imp_p.tolist(), exp_p.tolist(), soc0=soc_mpc
            )
            soc_mpc = float(df_mpc["soc"].iloc[-1])
            df_mpc["date"] = date_str
            df_mpc["scenario"] = "MPC-TFT"
            mpc_records.append(df_mpc)
        except RuntimeError as e:
            print(f"[WARN] MPC failed on {date_str}: {e}")

        # Rule-based
        df_rule, soc_rule = sim_rule_based(profile, init_soc=soc_rule)
        df_rule["date"] = date_str
        df_rule["scenario"] = "Rule-Based"
        rule_records.append(df_rule)

        if (idx + 1) % 30 == 0:
            print(f"  {idx + 1}/{len(profiles)} days done")

    all_df = pd.concat(mpc_records + rule_records, ignore_index=True)
    all_df.to_csv(logs / "mpc_results.csv", index=False)

    # ── Daily aggregate summary ──────────────────────────────────────────
    metrics = ["lng", "grid_import", "export_mwh", "curtailment", "cost"]
    avail   = [c for c in metrics if c in all_df.columns]
    summary = all_df.groupby(["date", "scenario"])[avail].sum().reset_index()
    summary.to_csv(logs / "mpc_summary.csv", index=False)

    # ── Comparison table ─────────────────────────────────────────────────
    agg = all_df.groupby("scenario")[avail].sum()
    lines = ["\n=== MPC-TFT vs Rule-Based (Annual Totals) ===", f"{'Metric':20s}  {'Rule-Based':>14s}  {'MPC-TFT':>14s}  {'Δ%':>8s}"]
    if "MPC-TFT" in agg.index and "Rule-Based" in agg.index:
        for col in avail:
            r = agg.loc["Rule-Based", col]
            m = agg.loc["MPC-TFT",   col]
            pct = (m - r) / (abs(r) + 1e-9) * 100
            lines.append(f"  {col:18s}  {r:14.1f}  {m:14.1f}  {pct:+8.2f}%")
    report = "\n".join(lines)
    print(report)
    (logs / "comparison_table.txt").write_text(report)
    print(f"\nResults saved to {logs}/")


if __name__ == "__main__":
    main()
