"""
KPX 한국전력거래소 지역별 시간대별 전력거래량 파서
파일: 한국전력거래소_지역별 시간대별 전력거래량(01~24)/*.csv
컬럼: 거래일자, 시간(1~24), 지역, 전력거래량(MWh)
"""

import glob
import pandas as pd
from pathlib import Path
import yaml


# 호남권 지역 목록
HONAM_REGIONS = ['광주시', '전라남도', '전라북도', '전북특별자치도']


def parse_kpx_folder(
    folder: str,
    regions: list = HONAM_REGIONS,
    start_year: int = 2014,
    end_year:   int = 2025,
    exclude_periods: list = None,
    save_dir: str = "data/raw/kpx",
) -> pd.DataFrame:
    """연도별 CSV 파일 읽어서 호남권 시간별 전력거래량 추출"""

    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    files = sorted(glob.glob(f"{folder}/*.csv"))

    for fpath in files:
        # 파일명 끝 연도 파싱
        year_str = Path(fpath).stem[-4:]
        if not year_str.isdigit():
            continue
        year = int(year_str)
        if not (start_year <= year <= end_year):
            continue

        print(f"  {year}년 파싱 중...", end=" ", flush=True)
        try:
            df = pd.read_csv(fpath, encoding="cp949", low_memory=False)
            df.columns = [c.strip() for c in df.columns]
            # 전력거래량 숫자 강제 변환 (혼합 타입 대응)
            df["전력거래량(MWh)"] = pd.to_numeric(df["전력거래량(MWh)"], errors="coerce")

            # 호남권 필터
            honam = df[df["지역"].isin(regions)].copy()

            # 시간 컬럼명 통일 (2014~2016: '시간', 2017~: '거래시간')
            time_col = "시간" if "시간" in honam.columns else "거래시간"
            honam["hour"] = honam[time_col].astype(int) - 1
            honam["datetime"] = pd.to_datetime(honam["거래일자"]) + \
                                 pd.to_timedelta(honam["hour"], unit="h")

            # 지역 합산 (광주+전남+전북 → 호남 총계)
            hourly = honam.groupby("datetime")["전력거래량(MWh)"].sum().reset_index()
            hourly.columns = ["datetime", "demand"]

            all_dfs.append(hourly)
            print(f"{len(hourly)}행")
        except Exception as e:
            print(f"오류: {e}")

    if not all_dfs:
        print("파싱된 데이터 없음")
        return pd.DataFrame()

    merged = pd.concat(all_dfs, ignore_index=True).sort_values("datetime").reset_index(drop=True)

    # 코로나 기간 제외
    if exclude_periods:
        for ex_s, ex_e in exclude_periods:
            mask = (merged["datetime"] >= ex_s) & (merged["datetime"] <= ex_e)
            merged = merged[~mask]
            print(f"  코로나 제외: {ex_s} ~ {ex_e} ({mask.sum()}행 제거)")

    out_file = save_path / "kpx_honam.csv"
    merged.to_csv(out_file, index=False, encoding="utf-8-sig")
    print(f"\n저장: {out_file} ({len(merged)}행)")
    print(f"기간: {merged['datetime'].min()} ~ {merged['datetime'].max()}")
    merged["demand"] = pd.to_numeric(merged["demand"], errors="coerce")
    print(f"수요 평균: {merged['demand'].mean():.1f} MWh/h")
    return merged


def load_kpx(save_dir: str = "data/raw/kpx") -> pd.DataFrame:
    path = Path(save_dir) / "kpx_honam.csv"
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, parse_dates=["datetime"])


if __name__ == "__main__":
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    folder = "data/raw/kpx/한국전력거래소_지역별 시간대별 전력거래량(01~24)"
    parse_kpx_folder(
        folder=folder,
        regions=HONAM_REGIONS,
        start_year=int(cfg["data"]["train_start"][:4]),
        end_year=int(cfg["data"]["test_end"][:4]),
        exclude_periods=cfg["data"]["exclude"],
        save_dir=f"{cfg['data']['raw_dir']}/kpx",
    )
