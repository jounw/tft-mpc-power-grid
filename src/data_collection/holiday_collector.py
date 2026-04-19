"""
공공데이터포털 한국천문연구원 특일정보 API - 공휴일 수집
API: https://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo
"""

import time
import requests
import pandas as pd
from pathlib import Path
import yaml


API_URL = "http://apis.data.go.kr/B090041/openapi/service/SpcdeInfoService/getRestDeInfo"


def fetch_holidays_by_year(api_key: str, year: int) -> pd.DataFrame:
    """연도별 공휴일 조회"""
    rows = []
    for month in range(1, 13):
        params = {
            "serviceKey": api_key,
            "pageNo": 1,
            "numOfRows": 50,
            "_type": "json",
            "solYear": year,
            "solMonth": f"{month:02d}",
        }
        try:
            resp = requests.get(API_URL, params=params, timeout=15)
            resp.raise_for_status()
            body = resp.json().get("response", {}).get("body", {})
            items = body.get("items", {})
            if items:
                item = items.get("item", [])
                if isinstance(item, dict):
                    item = [item]
                rows.extend(item)
        except Exception as e:
            print(f"  {year}-{month:02d} 오류: {e}")
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["locdate"].astype(str), format="%Y%m%d")
    df = df[["date", "dateName", "isHoliday"]].rename(
        columns={"dateName": "holiday_name", "isHoliday": "is_holiday"}
    )
    return df


def collect_holidays(
    api_key: str,
    start_year: int,
    end_year: int,
    save_dir: str = "data/raw/holiday",
) -> pd.DataFrame:
    """전체 기간 공휴일 수집 및 저장"""
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for year in range(start_year, end_year + 1):
        print(f"  {year}년 공휴일 수집...", end=" ")
        df = fetch_holidays_by_year(api_key, year)
        if not df.empty:
            all_dfs.append(df)
            print(f"{len(df)}건")
        else:
            print("없음")

    if not all_dfs:
        return pd.DataFrame()

    combined = pd.concat(all_dfs, ignore_index=True).sort_values("date")
    out_file = save_path / "holidays.csv"
    combined.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"저장: {out_file} ({len(combined)}건)")
    return combined


def make_holiday_feature(
    date_range: pd.DatetimeIndex,
    holiday_df: pd.DataFrame,
) -> pd.DataFrame:
    """날짜 인덱스에 공휴일 피처 추가"""
    holiday_dates = set(holiday_df["date"].dt.date)

    df = pd.DataFrame(index=date_range)
    df["is_holiday"] = df.index.date
    df["is_holiday"] = df["is_holiday"].apply(lambda d: 1 if d in holiday_dates else 0)
    df["is_weekend"] = (df.index.dayofweek >= 5).astype(int)
    df["is_off_day"] = ((df["is_holiday"] == 1) | (df["is_weekend"] == 1)).astype(int)
    return df


def load_holidays(save_dir: str = "data/raw/holiday") -> pd.DataFrame:
    path = Path(save_dir) / "holidays.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["date"])


if __name__ == "__main__":
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    start_year = int(cfg["data"]["train_start"][:4])
    end_year   = int(cfg["data"]["test_end"][:4])

    collect_holidays(
        api_key=cfg["holiday"]["api_key"],
        start_year=start_year,
        end_year=end_year,
        save_dir=f"{cfg['data']['raw_dir']}/holiday",
    )
