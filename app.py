import sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import src.optimization.mpc as _mpc  # monkey-patch용 모듈 참조

# ── 04_run_mpc.py 동일 SMP 프로파일 ──────────────────────────────────────────
_BASE_SMP     = 176.0
_SEASON_MULT  = {12:1.25,1:1.25,2:1.25,6:1.30,7:1.30,8:1.30,
                 3:1.00,4:1.00,5:1.00,9:1.00,10:1.00,11:1.00}

def make_price_profile(month: int, seed: int = 0):
    rng  = np.random.default_rng(seed)
    mult = _SEASON_MULT.get(month, 1.0)
    imp  = _BASE_SMP * mult * rng.uniform(0.92, 1.08, 24).astype(np.float32)
    return imp, imp * 0.95

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="TFT-MPC Power Grid Demo",
    page_icon="⚡",
    layout="wide",
)

# ── Load data ───────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    proc = ROOT / "data" / "processed"
    test_raw  = pd.read_csv(proc / "test_raw.csv", parse_dates=["datetime"])
    fc_path   = proc / "tft_forecasts.csv"
    if not fc_path.exists():
        raise FileNotFoundError("tft_forecasts")
    forecasts = pd.read_csv(fc_path)
    return test_raw, forecasts

try:
    test_raw, forecasts = load_data()
    DATA_OK = True
    TFT_AVAILABLE = True
except FileNotFoundError as e:
    if "tft_forecasts" in str(e):
        # tft_forecasts.csv 없으면 test_raw만 로드하고 actual demand를 fallback으로 사용
        proc = ROOT / "data" / "processed"
        test_raw = pd.read_csv(proc / "test_raw.csv", parse_dates=["datetime"])
        forecasts = None
        DATA_OK = True
        TFT_AVAILABLE = False
    else:
        DATA_OK = False
        TFT_AVAILABLE = False


# ── Simulation helpers (04_run_mpc.py 동일 로직) ───────────────────────────
def run_rule_based(dm, solar, p):
    """04_run_mpc.py sim_rule_based와 동일. 슬라이더 파라미터 반영."""
    nuc = p["nuclear"]
    coal = p["coal"]
    ess_cap = p["ess_cap"]
    ess_pwr = p["ess_pwr"]
    ess_eff = p["ess_eff"]
    soc_min = p["soc_min"]
    soc_max = p["soc_max"]
    imp_cap = p["imp_cap"]
    exp_cap = p["exp_cap"]

    soc = p["soc_init"]
    rows = []
    for t in range(24):
        fixed = nuc + coal + solar[t]
        bal   = fixed - dm[t]
        chg = dis = 0.0
        if bal > 0:
            chg = max(0.0, min(ess_pwr, bal, (soc_max - soc) * ess_cap / ess_eff))
            soc += chg * ess_eff / ess_cap
        elif bal < 0:
            dis = max(0.0, min(ess_pwr, -bal, (soc - soc_min) * ess_cap * ess_eff))
            soc -= dis / (ess_eff * ess_cap)
        soc = float(np.clip(soc, soc_min, soc_max))

        ess_net = chg - dis * ess_eff
        lng = float(np.clip(dm[t] - (fixed - ess_net), p["lng_min"], p["lng_max"]))
        total   = fixed + lng - ess_net
        surplus = max(0.0, total - dm[t])
        export  = min(exp_cap, surplus)
        deficit = max(0.0, dm[t] - (total - export))
        imp     = min(deficit, imp_cap)
        curtail = max(0.0, (total - export) - dm[t] - imp)
        cost    = lng * 120 + imp * 176 - export * 176 * 0.45
        rows.append(dict(nuclear=nuc, coal=coal, lng=lng,
                         solar=solar[t], ess_charge=chg, ess_discharge=dis,
                         grid_import=imp, export_mwh=export,
                         curtailment=curtail, soc=soc, cost=cost, demand=dm[t]))
    return pd.DataFrame(rows)


def run_mpc(dm, solar, fc_demand, p, month: int = 1):
    """04_run_mpc.py run_rolling_mpc와 완전히 동일한 로직.
    슬라이더 파라미터를 mpc 모듈 상수에 monkey-patch 후 호출."""
    imp_price, exp_price = make_price_profile(month)

    profile = {
        "demand": dm,
        "solar":  solar,
        "wind":   np.zeros(24, dtype=np.float32),
        "solar_forecast": solar,
        "wind_forecast":  np.zeros(24, dtype=np.float32),
    }

    # 원본 상수 백업
    _KEYS = ["NUCLEAR", "COAL", "LNG_CAP", "LNG_MIN", "LNG_RAMP",
             "ESS_CAP", "ESS_PWR", "ESS_MAX_PWR",
             "SOC_INIT", "SOC_MIN", "SOC_MAX", "EXP_CAP", "IMP_CAP"]
    _orig = {k: getattr(_mpc, k) for k in _KEYS if hasattr(_mpc, k)}

    try:
        _mpc.NUCLEAR     = float(p["nuclear"])
        _mpc.COAL        = float(p["coal"])
        _mpc.LNG_CAP     = float(p["lng_max"])
        _mpc.LNG_MIN     = float(p["lng_min"])
        _mpc.ESS_CAP     = float(p["ess_cap"])
        _mpc.ESS_PWR     = float(p["ess_pwr"])
        _mpc.ESS_MAX_PWR = float(p["ess_pwr"])
        _mpc.SOC_INIT    = float(p["soc_init"])
        _mpc.SOC_MIN     = float(p["soc_min"])
        _mpc.SOC_MAX     = float(p["soc_max"])
        _mpc.EXP_CAP     = float(p["exp_cap"])
        _mpc.IMP_CAP     = float(p["imp_cap"])

        df = _mpc.run_rolling_mpc(
            profile,
            fc_demand.tolist(),
            imp_price.tolist(),
            exp_price.tolist(),
            soc0=p["soc_init"],
            lng0=p["lng_min"],
        )
        return df
    except RuntimeError:
        return run_rule_based(dm, solar, p)
    finally:
        for k, v in _orig.items():
            setattr(_mpc, k, v)


# ── Chart helpers ───────────────────────────────────────────────────────────
COLORS = dict(
    nuclear="#1f3a5f", coal="#5a4a3a", lng="#e07b39",
    solar="#f5c842",   grid_import="#2ca02c", export="#17becf",
    curtail="#d62728", rule="#ff7f0e", mpc="#1f77b4",
)

def gen_mix_chart(df_rule, df_mpc):
    hours = list(range(24))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Rule-Based", "MPC-TFT"],
                        shared_yaxes=True)
    for col, (df, label) in enumerate([(df_rule, "Rule"), (df_mpc, "MPC")], 1):
        for src, color in [("nuclear", COLORS["nuclear"]),
                           ("coal",    COLORS["coal"]),
                           ("lng",     COLORS["lng"]),
                           ("solar",   COLORS["solar"]),
                           ("grid_import", COLORS["grid_import"])]:
            fig.add_trace(go.Bar(x=hours, y=df[src], name=src,
                                 marker_color=color,
                                 legendgroup=src,
                                 showlegend=(col == 1)),
                          row=1, col=col)
        fig.add_trace(go.Scatter(x=hours, y=df["demand"], mode="lines",
                                 name="Demand", line=dict(color="white", width=2, dash="dot"),
                                 legendgroup="demand", showlegend=(col == 1)),
                      row=1, col=col)
    fig.update_layout(barmode="stack", height=380,
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      font_color="white", legend_bgcolor="#0e1117",
                      margin=dict(t=40, b=20))
    fig.update_xaxes(title_text="Hour", tickfont_color="white")
    fig.update_yaxes(title_text="MW", tickfont_color="white")
    return fig


def soc_chart(df_rule, df_mpc):
    hours = list(range(24))
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hours, y=df_rule["soc"], name="Rule-Based",
                             line=dict(color=COLORS["rule"], width=2)))
    fig.add_trace(go.Scatter(x=hours, y=df_mpc["soc"], name="MPC-TFT",
                             line=dict(color=COLORS["mpc"], width=2)))
    fig.add_hrect(y0=0.35, y1=0.65, fillcolor="gray", opacity=0.1,
                  annotation_text="target SoC zone")
    fig.update_layout(height=280, title="ESS State of Charge",
                      yaxis=dict(range=[0, 1], title="SoC"),
                      xaxis_title="Hour",
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      font_color="white", margin=dict(t=40, b=20))
    return fig


def curtail_cost_chart(df_rule, df_mpc):
    hours = list(range(24))
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=["Curtailment (MWh)", "Cumulative Cost (KRW×10⁶)"])
    for df, color, name in [(df_rule, COLORS["rule"], "Rule-Based"),
                             (df_mpc,  COLORS["mpc"],  "MPC-TFT")]:
        fig.add_trace(go.Bar(x=hours, y=df["curtailment"], name=name,
                             marker_color=color, opacity=0.8,
                             legendgroup=name, showlegend=True),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=hours, y=df["cost"].cumsum() / 1e6,
                                 name=name, line=dict(color=color, width=2),
                                 legendgroup=name, showlegend=False),
                      row=1, col=2)
    fig.update_layout(height=280, barmode="group",
                      paper_bgcolor="#0e1117", plot_bgcolor="#0e1117",
                      font_color="white", margin=dict(t=40, b=20))
    return fig


def summary_metrics(df_rule, df_mpc):
    metrics = {
        "Total Cost (M KRW)": ("cost", 1e6, True),
        "LNG Generation (GWh)": ("lng", 1e3, True),
        "Grid Import (GWh)": ("grid_import", 1e3, True),
        "Grid Export (GWh)": ("export_mwh", 1e3, False),
        "Curtailment (MWh)": ("curtailment", 1, True),
    }
    cols = st.columns(len(metrics))
    for col, (label, (key, div, lower_is_better)) in zip(cols, metrics.items()):
        r = df_rule[key].sum() / div
        m = df_mpc[key].sum() / div
        delta = m - r
        pct   = delta / (abs(r) + 1.0) * 100  # +1 prevents explosion when r≈0
        arrow = "▼" if delta < 0 else "▲"
        good  = (delta < 0) == lower_is_better
        color = "#2ecc71" if good else "#e74c3c"
        col.metric(
            label=label,
            value=f"{m:.2f}",
            delta=f"{arrow} {abs(pct):.1f}% vs Rule",
            delta_color="normal" if good else "inverse",
        )


# ── Main UI ─────────────────────────────────────────────────────────────────
st.title("⚡ TFT-MPC Power Grid Optimization")
st.caption("Honam Region — ESS & LNG Dispatch Simulator")

if not DATA_OK:
    st.error("`data/processed/test_raw.csv` not found.")
    st.stop()

if not TFT_AVAILABLE:
    st.info("💡 TFT 예측값(tft_forecasts.csv) 없음 — 실제 수요를 Perfect Forecast로 대체합니다. "
            "A100에서 `python scripts/05_precompute_forecasts.py` 실행 후 재배포하면 TFT 예측 사용 가능합니다.")

# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Parameters")

    # Date
    if TFT_AVAILABLE:
        available_dates = sorted(forecasts["date"].tolist())
    else:
        available_dates = sorted(test_raw["datetime"].dt.strftime("%Y-%m-%d").unique().tolist())
    selected_date = st.selectbox("📅 Date", available_dates, index=0)

    st.divider()
    st.subheader("🏭 Fixed Generation")
    nuclear = st.slider("Nuclear (MW)", 3000, 6000, 4300, step=100)
    coal    = st.slider("Coal (MW)",     500, 2000, 1000, step=100)

    st.divider()
    st.subheader("🔥 LNG")
    lng_min = st.slider("LNG Min (MW)",  200,  800,  420, step=20)
    lng_max = st.slider("LNG Max (MW)", 1200, 3000, 2100, step=100)

    st.divider()
    st.subheader("🔋 ESS")
    ess_cap  = st.slider("Capacity (MWh)", 1000, 8000, 5600, step=200)
    ess_pwr  = st.slider("Max Power (MW)",  500, 2000, 1400, step=100)
    soc_init = st.slider("Initial SoC",    0.1,  0.9,  0.5, step=0.05)

    st.divider()
    st.subheader("🌐 Grid")
    imp_cap = st.slider("Import Cap (MW)",  500, 5000, 3000, step=100)
    exp_cap = st.slider("Export Cap (MW)",  500, 5000, 3000, step=100)

    st.divider()
    st.subheader("☀️ Renewable Multiplier")
    renew_mult = st.slider("Solar/Wind ×", 0.0, 2.0, 1.0, step=0.1)

params = dict(
    nuclear=nuclear, coal=coal,
    lng_min=lng_min, lng_max=lng_max,
    ess_cap=ess_cap, ess_pwr=ess_pwr, ess_eff=0.95,
    soc_init=soc_init, soc_min=0.10, soc_max=0.90,
    imp_cap=imp_cap, exp_cap=exp_cap,
)

# ── Load day data ─────────────────────────────────────────────────────────
day_data = test_raw[test_raw["datetime"].dt.strftime("%Y-%m-%d") == selected_date].copy()

if len(day_data) < 24:
    st.warning("No data for selected date.")
    st.stop()

demand_actual  = day_data["demand"].values[:24].astype(float)
selected_month = pd.to_datetime(selected_date).month

# ── 합성 태양광 프로파일 ────────────────────────────────────────────────────
# 호남권 태양광 설비 ~10 GW (2024), 시간별 실측 없어 가우시안 근사
_SOLAR_PEAK_MW = {1:700, 2:900, 3:1200, 4:1600, 5:1900,
                  6:2000, 7:1800, 8:1900, 9:1500, 10:1100, 11:800, 12:600}
_peak = _SOLAR_PEAK_MW.get(selected_month, 1200)
_hours = np.arange(24)
solar_actual = np.clip(
    _peak * np.exp(-0.5 * ((_hours - 12.0) / 3.0) ** 2), 0, None
).astype(float) * renew_mult

if TFT_AVAILABLE:
    fc_row    = forecasts[forecasts["date"] == selected_date]
    fc_demand = fc_row.iloc[0, 1:].values.astype(float) if not fc_row.empty else demand_actual
else:
    fc_demand = demand_actual  # Perfect forecast fallback

# weather summary (일별 대표값)
wc1, wc2, wc3, wc4 = st.columns(4)
wc1.metric("🌡 Avg Temp",      f"{day_data['temp_mean'].mean():.1f} °C")
wc2.metric("💧 Humidity",      f"{day_data['humidity'].mean():.0f} %")
wc3.metric("💨 Wind Speed",    f"{day_data['wind_speed'].mean():.1f} m/s")
wc4.metric("☀️ Solar Gen (est)", f"{solar_actual.sum():.0f} MWh (daily)")

st.divider()

# ── Run simulations ───────────────────────────────────────────────────────
with st.spinner("Simulating..."):
    df_rule = run_rule_based(demand_actual, solar_actual, params)
    df_mpc  = run_mpc(demand_actual, solar_actual, fc_demand, params, month=selected_month)

# ── Summary metrics ───────────────────────────────────────────────────────
st.subheader("📊 Daily Summary — MPC-TFT vs Rule-Based")
summary_metrics(df_rule, df_mpc)

st.divider()

# ── Charts ────────────────────────────────────────────────────────────────
st.subheader("🏭 Generation Mix")
st.plotly_chart(gen_mix_chart(df_rule, df_mpc), use_container_width=True)

col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(soc_chart(df_rule, df_mpc), use_container_width=True)
with col2:
    st.plotly_chart(curtail_cost_chart(df_rule, df_mpc), use_container_width=True)

# ── Raw dispatch table ─────────────────────────────────────────────────────
with st.expander("📋 Hourly Dispatch Table"):
    tab1, tab2 = st.tabs(["Rule-Based", "MPC-TFT"])
    with tab1:
        st.dataframe(df_rule.round(1), use_container_width=True)
    with tab2:
        st.dataframe(df_mpc.round(1), use_container_width=True)
