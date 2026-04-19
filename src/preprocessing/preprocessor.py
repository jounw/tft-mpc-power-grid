"""
데이터 전처리: KPX(수요/태양광/풍력) + 기상 + 공휴일 병합 및 피처 생성
출력: 시간별 학습 데이터프레임
"""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
import joblib
import yaml


# ────────────────────────────────────────────
# 1. 로드
# ────────────────────────────────────────────

def load_kpx(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["datetime"])
    return df.sort_values("datetime").reset_index(drop=True)


def load_weather_daily(path: str) -> pd.DataFrame:
    """6시간 기상 → 일별 집계"""
    df = pd.read_csv(path, parse_dates=["datetime"])
    df["date"] = df["datetime"].dt.date

    # 호남권 관측소 단순 평균 (가중치 없이)
    daily = df.groupby("date").agg(
        temp_mean=("temp", "mean"),
        temp_max=("temp", "max"),
        temp_min=("temp", "min"),
        humidity=("humidity", "mean"),
        wind_speed=("wind_speed", "mean"),
        solar_rad=("solar_rad", "sum"),
        rainfall=("rainfall", "sum"),
    ).reset_index()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily


def load_holidays(path: str) -> set:
    df = pd.read_csv(path, parse_dates=["date"])
    return set(df["date"].dt.date)


# ────────────────────────────────────────────
# 2. 피처 엔지니어링
# ────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """날짜/시간 사이클 인코딩"""
    df = df.copy()
    dt = df["datetime"]

    df["hour"]      = dt.dt.hour
    df["dayofweek"] = dt.dt.dayofweek   # 0=월 ~ 6=일
    df["month"]     = dt.dt.month
    df["dayofyear"] = dt.dt.dayofyear
    df["is_weekend"]= (dt.dt.dayofweek >= 5).astype(int)

    # 사이클 인코딩 (hour, month, dayofyear)
    df["hour_sin"]  = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"]  = np.cos(2 * np.pi * df["hour"] / 24)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)
    return df


def add_weather_features(df: pd.DataFrame, weather_daily: pd.DataFrame) -> pd.DataFrame:
    """일별 기상 피처 병합 + 냉난방도일 계산"""
    df = df.copy()
    df["date"] = df["datetime"].dt.normalize()
    df = pd.merge(df, weather_daily, on="date", how="left")

    # 냉방도일(CDD) / 난방도일(HDD) — 기준온도 18°C
    df["cdd"] = (df["temp_mean"] - 18).clip(lower=0)
    df["hdd"] = (18 - df["temp_mean"]).clip(lower=0)
    # 온도의 비선형성 캡처
    df["temp_sq"] = df["temp_mean"] ** 2

    df.drop(columns=["date"], inplace=True)
    return df


def add_holiday_features(df: pd.DataFrame, holiday_dates: set) -> pd.DataFrame:
    df = df.copy()
    df["is_holiday"] = df["datetime"].dt.date.apply(
        lambda d: 1 if d in holiday_dates else 0
    )
    df["is_off_day"] = ((df["is_weekend"] == 1) | (df["is_holiday"] == 1)).astype(int)
    return df


def add_lag_features(df: pd.DataFrame, target: str = "demand") -> pd.DataFrame:
    """수요 래그 피처 (직전 1일, 1주일, 2주일)"""
    df = df.copy()
    df[f"{target}_lag24"]  = df[target].shift(24)   # 전일 동시각
    df[f"{target}_lag168"] = df[target].shift(168)  # 전주 동시각
    df[f"{target}_lag336"] = df[target].shift(336)  # 2주 전 동시각
    # 직전 24시간 평균
    df[f"{target}_roll24_mean"] = df[target].shift(1).rolling(24).mean()
    return df


def add_renewable_features(df: pd.DataFrame) -> pd.DataFrame:
    """재생에너지 비중, 잉여 발전 계산"""
    df = df.copy()
    solar = df["solar"].fillna(0) if "solar" in df.columns else 0
    wind  = df["wind"].fillna(0)  if "wind"  in df.columns else 0
    df["renewable_total"] = solar + wind
    df["renewable_ratio"] = df["renewable_total"] / (df["demand"].replace(0, np.nan))
    # 잉여 발전량 (양수면 출력제한 위험)
    df["surplus"]         = df["renewable_total"] - df["demand"]
    return df


# ────────────────────────────────────────────
# 3. 코로나 제외 + train/val/test 분리
# ────────────────────────────────────────────

def split_dataset(df: pd.DataFrame, cfg: dict):
    exclude = cfg["data"]["exclude"]
    for ex_s, ex_e in exclude:
        mask = (df["datetime"] >= ex_s) & (df["datetime"] <= ex_e)
        df = df[~mask]

    train_mask = (df["datetime"] >= cfg["data"]["train_start"]) & \
                 (df["datetime"] <= cfg["data"]["train_end"])
    test_mask  = (df["datetime"] >= cfg["data"]["test_start"])  & \
                 (df["datetime"] <= cfg["data"]["test_end"])

    # Val: train 마지막 1년
    val_start  = str(int(cfg["data"]["train_end"][:4]) - 1) + cfg["data"]["train_end"][4:]
    val_mask   = (df["datetime"] >= val_start) & \
                 (df["datetime"] <= cfg["data"]["train_end"])
    pure_train = train_mask & ~val_mask

    return df[pure_train], df[val_mask], df[test_mask]


# ────────────────────────────────────────────
# 4. 스케일링
# ────────────────────────────────────────────

SCALE_COLS = [
    "demand", "solar", "wind",
    "temp_mean", "temp_max", "temp_min",
    "humidity", "wind_speed", "solar_rad", "rainfall",
    "cdd", "hdd", "temp_sq",
    "demand_lag24", "demand_lag168", "demand_lag336",
    "demand_roll24_mean", "renewable_total", "renewable_ratio", "surplus",
]


def fit_scalers(train_df: pd.DataFrame, save_dir: str = "data/processed"):
    scalers = {}
    for col in SCALE_COLS:
        if col in train_df.columns:
            sc = StandardScaler()
            sc.fit(train_df[[col]].dropna())
            scalers[col] = sc
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    joblib.dump(scalers, f"{save_dir}/scalers.pkl")
    return scalers


def apply_scalers(df: pd.DataFrame, scalers: dict) -> pd.DataFrame:
    df = df.copy()
    for col, sc in scalers.items():
        if col in df.columns:
            df[col] = sc.transform(df[[col]])
    return df


# ────────────────────────────────────────────
# 5. 시퀀스 생성 (LSTM 입력용)
# ────────────────────────────────────────────

FEATURE_COLS = [
    "demand_lag24", "demand_lag168", "demand_roll24_mean",
    "temp_mean", "temp_max", "temp_min", "cdd", "hdd", "temp_sq",
    "humidity", "wind_speed", "solar_rad", "rainfall",
    "hour_sin", "hour_cos", "month_sin", "month_cos", "dow_sin", "dow_cos",
    "is_off_day", "is_holiday",
    "solar", "wind", "renewable_ratio", "surplus",
]


def make_sequences(df: pd.DataFrame, seq_len: int = 168, horizon: int = 24):
    """
    seq_len 시간의 과거 → horizon 시간 앞 수요 예측
    Returns: X (N, seq_len, features), y (N, horizon)
    """
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    X_arr = df[feat_cols].values
    y_arr = df["demand"].values

    X, y = [], []
    for i in range(seq_len, len(df) - horizon + 1):
        x_seq = X_arr[i - seq_len:i]
        y_seq = y_arr[i:i + horizon]
        if np.isnan(x_seq).any() or np.isnan(y_seq).any():
            continue
        X.append(x_seq)
        y.append(y_seq)

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ────────────────────────────────────────────
# 6. 통합 실행
# ────────────────────────────────────────────

def run(cfg: dict):
    raw = cfg["data"]["raw_dir"]
    proc = cfg["data"]["processed_dir"]
    Path(proc).mkdir(parents=True, exist_ok=True)

    # 로드
    kpx      = load_kpx(f"{raw}/kpx/kpx_honam.csv")
    weather  = load_weather_daily(f"{raw}/weather/weather_all.csv")
    holidays = load_holidays(f"{raw}/holiday/holidays.csv")

    # 피처 추가
    df = add_time_features(kpx)
    df = add_weather_features(df, weather)
    df = add_holiday_features(df, holidays)
    df = add_lag_features(df)
    df = add_renewable_features(df)
    df = df.dropna().reset_index(drop=True)

    # 분리
    train_df, val_df, test_df = split_dataset(df, cfg)
    print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # 스케일링
    scalers  = fit_scalers(train_df, proc)
    train_sc = apply_scalers(train_df, scalers)
    val_sc   = apply_scalers(val_df,   scalers)
    test_sc  = apply_scalers(test_df,  scalers)

    # 시퀀스
    X_train, y_train = make_sequences(train_sc)
    X_val,   y_val   = make_sequences(val_sc)
    X_test,  y_test  = make_sequences(test_sc)

    # 저장
    import numpy as np
    np.save(f"{proc}/X_train.npy", X_train)
    np.save(f"{proc}/y_train.npy", y_train)
    np.save(f"{proc}/X_val.npy",   X_val)
    np.save(f"{proc}/y_val.npy",   y_val)
    np.save(f"{proc}/X_test.npy",  X_test)
    np.save(f"{proc}/y_test.npy",  y_test)

    # 원본(비정규화) test 저장 — RL 환경에서 사용
    test_df.to_csv(f"{proc}/test_raw.csv", index=False, encoding="utf-8-sig")

    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    print(f"저장 완료: {proc}/")
    return X_train, y_train, X_val, y_val, X_test, y_test


if __name__ == "__main__":
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    run(cfg)
