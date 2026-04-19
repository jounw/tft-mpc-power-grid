"""
전체 데이터 수집 통합 실행 스크립트
사용법:
  python src/data_collection/collect_all.py --step all
  python src/data_collection/collect_all.py --step weather
  python src/data_collection/collect_all.py --step holiday
  python src/data_collection/collect_all.py --step kpx
"""

import argparse
import sys
from pathlib import Path

# 프로젝트 루트를 경로에 추가
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import yaml
from src.data_collection.weather_collector import collect_weather
from src.data_collection.holiday_collector import collect_holidays
from src.data_collection.kpx_parser import parse_all_kpx


def load_config() -> dict:
    with open(ROOT / "configs" / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def step_weather(cfg: dict):
    print("=" * 50)
    print("[1/3] 기상청 시간별 기상 데이터 수집")
    print("=" * 50)

    api_key = cfg["weather"]["api_key"]
    if api_key == "YOUR_KMA_API_KEY":
        print("경고: config.yaml의 weather.api_key를 설정하세요.")
        print("  → 공공데이터포털(data.go.kr) 가입 후 '기상청_지상(종관, ASOS) 시간자료 조회서비스' API 키 발급")
        return

    collect_weather(
        api_key=api_key,
        stations=cfg["weather"]["stations"],
        start_date=cfg["data"]["train_start"],
        end_date=cfg["data"]["test_end"],
        exclude_periods=cfg["data"]["exclude"],
        save_dir=str(ROOT / cfg["data"]["raw_dir"] / "weather"),
        interval_hours=24,   # 1일 1회 정오 수집 (~22분)
    )


def step_holiday(cfg: dict):
    print("=" * 50)
    print("[2/3] 공휴일 데이터 수집")
    print("=" * 50)

    api_key = cfg["holiday"]["api_key"]
    if api_key == "YOUR_DATA_GO_KR_API_KEY":
        print("경고: config.yaml의 holiday.api_key를 설정하세요.")
        print("  → 공공데이터포털(data.go.kr) 가입 후 '한국천문연구원_특일 정보' API 키 발급")
        return

    start_year = int(cfg["data"]["train_start"][:4])
    end_year   = int(cfg["data"]["test_end"][:4])

    collect_holidays(
        api_key=api_key,
        start_year=start_year,
        end_year=end_year,
        save_dir=str(ROOT / cfg["data"]["raw_dir"] / "holiday"),
    )


def step_kpx(cfg: dict):
    print("=" * 50)
    print("[3/3] KPX 전력 데이터 파싱")
    print("=" * 50)
    print("KPX EPSIS(https://epsis.kpx.or.kr) 수동 다운로드 필요")
    print("  다운로드 후 configs/config.yaml의 kpx 경로를 수정하세요.\n")

    parse_all_kpx(
        demand_file=str(ROOT / cfg["kpx"]["demand_file"]),
        solar_file=str(ROOT / cfg["kpx"]["solar_file"]),
        wind_file=str(ROOT / cfg["kpx"]["wind_file"]),
        save_dir=str(ROOT / cfg["data"]["raw_dir"] / "kpx"),
        exclude_periods=cfg["data"]["exclude"],
    )


def main():
    parser = argparse.ArgumentParser(description="데이터 수집 실행")
    parser.add_argument(
        "--step",
        choices=["all", "weather", "holiday", "kpx"],
        default="all",
        help="수집할 단계 (기본: all)",
    )
    args = parser.parse_args()

    cfg = load_config()

    if args.step in ("all", "weather"):
        step_weather(cfg)
    if args.step in ("all", "holiday"):
        step_holiday(cfg)
    if args.step in ("all", "kpx"):
        step_kpx(cfg)

    print("\n완료.")


if __name__ == "__main__":
    main()
