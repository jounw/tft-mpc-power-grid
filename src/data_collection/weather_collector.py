"""
기상청 API 허브(apihub.kma.go.kr) ASOS 지상관측 시간자료 수집
- 범위 쿼리(tm1/tm2)는 실시간만 반환 → 단일 tm 루프 방식 사용
- 6시간 간격으로 수집 (00, 06, 12, 18시): 일별 통계 산출에 충분
"""

import time
import requests
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional
import yaml


API_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"

# 응답 46컬럼 고정 포맷 (기상청 API 허브 help=1 기준)
COLUMNS = [
    "datetime", "stn_id",
    "wd", "ws",
    "gst_wd", "gst_ws", "gst_tm",
    "pa", "ps", "pt", "pr",
    "ta", "td", "hm", "pv",
    "rn", "rn_day", "rn_jun", "rn_int",
    "sd_hr3", "sd_day", "sd_tot",
    "wc", "wp", "ww",
    "ca_tot", "ca_mid", "ch_min",
    "ct", "ct_top", "ct_mid", "ct_low",
    "vs", "ss", "si",
    "st_gd", "ts",
    "te005", "te01", "te02", "te03",
    "st_sea", "wh", "bf",
    "ir", "ix",
]

RENAME = {
    "ta": "temp",
    "hm": "humidity",
    "ws": "wind_speed",
    "wd": "wind_dir",
    "si": "solar_rad",
    "rn": "rainfall",
}


def fetch_single_obs(api_key: str, station_id: int, tm: str) -> Optional[dict]:
    """단일 시각 관측값 조회 (tm: 'YYYYMMDDHHMI')"""
    resp = requests.get(
        API_URL,
        params={"tm": tm, "stn": station_id, "help": 0, "authKey": api_key},
        timeout=15,
    )
    resp.raise_for_status()

    lines = [l for l in resp.text.splitlines() if not l.startswith("#") and l.strip()]
    if not lines:
        return None

    vals = lines[0].split()
    n = len(vals)
    row = dict(zip(COLUMNS[:n], vals))

    # 결측 처리
    for k, v in row.items():
        if v in ("-9", "-9.0", "-99", "-999", "-9.00"):
            row[k] = None

    row["datetime"] = pd.to_datetime(row.get("datetime", tm), format="%Y%m%d%H%M", errors="coerce")
    row["station_id"] = station_id

    for old, new in RENAME.items():
        if old in row:
            row[new] = row.pop(old)

    keep = ["datetime", "station_id"] + list(RENAME.values())
    return {k: row.get(k) for k in keep}


def collect_weather(
    api_key: str,
    stations: dict,
    start_date: str,
    end_date: str,
    exclude_periods: list,
    save_dir: str = "data/raw/weather",
    interval_hours: int = 12,  # 12시간 간격 (00, 12시) — 일별 집계에 충분
) -> pd.DataFrame:
    """전체 기간·관측소 수집 후 저장 (연도별 중간 저장으로 중단 후 재개 가능)"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    excludes = [
        (datetime.strptime(s, "%Y-%m-%d"), datetime.strptime(e, "%Y-%m-%d"))
        for s, e in exclude_periods
    ]

    start_year = int(start_date[:4])
    end_year   = int(end_date[:4])
    all_dfs    = []

    for name, stn_id in stations.items():
        final_file = save_path / f"weather_{name}.csv"
        if final_file.exists():
            print(f"[{name}] 완료 파일 존재 → 스킵")
            all_dfs.append(pd.read_csv(final_file, parse_dates=["datetime"]))
            continue

        print(f"\n[{name} ({stn_id})] 수집 시작")
        year_dfs = []

        for year in range(start_year, end_year + 1):
            year_file = save_path / f"weather_{name}_{year}.csv"
            if year_file.exists():
                print(f"  {year}년 캐시 사용")
                year_dfs.append(pd.read_csv(year_file, parse_dates=["datetime"]))
                continue

            rows = []
            cur = datetime(year, 1, 1)
            end_dt = min(datetime(year, 12, 31, 23, 59),
                         datetime.strptime(end_date, "%Y-%m-%d"))

            while cur <= end_dt:
                skip = any(ex_s <= cur <= ex_e for ex_s, ex_e in excludes)
                if not skip:
                    tm_str = cur.strftime("%Y%m%d%H%M")
                    try:
                        row = fetch_single_obs(api_key, stn_id, tm_str)
                        if row:
                            rows.append(row)
                    except Exception as e:
                        print(f"  {tm_str} 오류: {e}")
                    time.sleep(0.15)
                cur += timedelta(hours=interval_hours)

            if rows:
                ydf = pd.DataFrame(rows)
                numeric_cols = [c for c in ydf.columns if c not in ("datetime", "station_id")]
                ydf[numeric_cols] = ydf[numeric_cols].apply(pd.to_numeric, errors="coerce")
                ydf = ydf.sort_values("datetime").reset_index(drop=True)
                ydf.to_csv(year_file, index=False, encoding="utf-8-sig")
                print(f"  {year}년 저장: {len(ydf)}행")
                year_dfs.append(ydf)

        if year_dfs:
            stn_df = pd.concat(year_dfs, ignore_index=True)
            stn_df["station_name"] = name
            stn_df.to_csv(final_file, index=False, encoding="utf-8-sig")
            print(f"  → {name} 완료: {len(stn_df)}행")
            # 연도별 캐시 삭제
            for year in range(start_year, end_year + 1):
                yf = save_path / f"weather_{name}_{year}.csv"
                if yf.exists():
                    yf.unlink()
            all_dfs.append(stn_df)

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True)
    combined_file = save_path / "weather_all.csv"
    combined.to_csv(combined_file, index=False, encoding="utf-8-sig")
    print(f"\n전체 저장: {combined_file} ({len(combined)}행)")
    return combined


def aggregate_daily(weather_df: pd.DataFrame) -> pd.DataFrame:
    """6시간 관측 → 일별 통계 집계 (예측 피처용)"""
    df = weather_df.copy()
    df["date"] = df["datetime"].dt.date

    agg = df.groupby(["date", "station_name"]).agg(
        temp_mean=("temp", "mean"),
        temp_max=("temp", "max"),
        temp_min=("temp", "min"),
        humidity_mean=("humidity", "mean"),
        wind_speed_mean=("wind_speed", "mean"),
        solar_rad_sum=("solar_rad", "sum"),
        rainfall_sum=("rainfall", "sum"),
    ).reset_index()

    agg["date"] = pd.to_datetime(agg["date"])
    return agg


def load_weather(save_dir: str = "data/raw/weather") -> Optional[pd.DataFrame]:
    path = Path(save_dir) / "weather_all.csv"
    if not path.exists():
        return None
    return pd.read_csv(path, parse_dates=["datetime"])


if __name__ == "__main__":
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    collect_weather(
        api_key=cfg["weather"]["api_key"],
        stations=cfg["weather"]["stations"],
        start_date=cfg["data"]["train_start"],
        end_date=cfg["data"]["test_end"],
        exclude_periods=cfg["data"]["exclude"],
        save_dir=f"{cfg['data']['raw_dir']}/weather",
    )
