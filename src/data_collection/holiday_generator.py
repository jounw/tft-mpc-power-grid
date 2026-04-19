"""
한국 공휴일 직접 생성 (API 불필요)
holidays 라이브러리 사용 — 음력 공휴일(설날, 추석, 부처님오신날) 자동 계산
"""

import holidays
import pandas as pd
from pathlib import Path


def generate_holidays(
    start_year: int = 2014,
    end_year:   int = 2025,
    save_dir:   str = "data/raw/holiday",
) -> pd.DataFrame:
    Path(save_dir).mkdir(parents=True, exist_ok=True)

    rows = []
    for year in range(start_year, end_year + 1):
        kr = holidays.KR(years=year)
        for date, name in sorted(kr.items()):
            rows.append({"date": date, "holiday_name": name})

    df = pd.DataFrame(rows)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)

    out = Path(save_dir) / "holidays.csv"
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"저장: {out} ({len(df)}건, {start_year}~{end_year})")
    print(df.groupby(df["date"].dt.year).size().to_string())
    return df


if __name__ == "__main__":
    import yaml
    with open("configs/config.yaml") as f:
        cfg = yaml.safe_load(f)

    generate_holidays(
        start_year=int(cfg["data"]["train_start"][:4]),
        end_year=int(cfg["data"]["test_end"][:4]),
        save_dir=f"{cfg['data']['raw_dir']}/holiday",
    )
