"""
피크 타임 보정 모듈
- HourBiasCorrector: 검증 세트 잔차에서 시간대별 편향을 학습, 추론 시 보정 적용
  - X의 hour_sin/hour_cos 피처로 각 시퀀스의 실제 시작 시각을 디코딩
  - 피크 시간대(오전 9-11시, 오후 18-21시)의 체계적 과소/과대 예측을 제거
"""

import numpy as np
import joblib

# 한국 전력 피크 시간대 (한 시간 단위)
PEAK_HOURS = list(range(9, 12)) + list(range(18, 22))   # 9,10,11,18,19,20,21


# FEATURE_COLS 내 hour_sin, hour_cos의 인덱스
# ["demand_lag24","demand_lag168","demand_roll24_mean",
#  "temp_mean","temp_max","temp_min","cdd","hdd","temp_sq",
#  "humidity","wind_speed","solar_rad","rainfall",
#  "hour_sin","hour_cos", ...]
HOUR_SIN_IDX = 13
HOUR_COS_IDX = 14


def _decode_start_hours(X: np.ndarray) -> np.ndarray:
    """
    X: (N, seq_len, features)
    X의 마지막 타임스텝 hour_sin/hour_cos → 예측 직전 시각 h → 예측 시작 시각 = (h+1)%24
    Returns: (N,) int array, 0~23
    """
    h_sin = X[:, -1, HOUR_SIN_IDX]
    h_cos = X[:, -1, HOUR_COS_IDX]
    # atan2 → 라디안 → 24h 스케일 변환 → 반올림
    h_float = np.arctan2(h_sin, h_cos) * 24.0 / (2.0 * np.pi)
    h_prev = np.round(h_float).astype(int) % 24        # 마지막 입력 시각
    return (h_prev + 1) % 24                            # 예측 시작 시각


class HourBiasCorrector:
    """
    시간대별 편향 보정기

    fit():
        검증 세트 예측 잔차(y_true - y_pred)를 실제 hour-of-day 기준으로 집계해
        시간대별 평균 편향 corrections_[0..23] 을 계산합니다.

    transform():
        추론 결과에 시간대별 보정값을 더해 반환합니다.
    """

    def __init__(self):
        self.corrections_ = np.zeros(24)   # (24,) normalized scale
        self.std_          = np.ones(24)
        self.count_        = np.zeros(24)
        self.fitted_       = False

    # ── 학습 ──────────────────────────────────────────────

    def fit(self, X_val: np.ndarray, y_val_true: np.ndarray, y_val_pred: np.ndarray):
        """
        Parameters
        ----------
        X_val      : (N, seq_len, features)  정규화된 입력
        y_val_true : (N, 24)  정규화된 실제 수요
        y_val_pred : (N, 24)  정규화된 예측 수요
        """
        N = len(X_val)
        h_starts = _decode_start_hours(X_val)   # (N,)

        accum  = [[] for _ in range(24)]
        for i in range(N):
            for j in range(24):
                actual_hour = (h_starts[i] + j) % 24
                accum[actual_hour].append(float(y_val_true[i, j] - y_val_pred[i, j]))

        for h in range(24):
            if accum[h]:
                arr = np.array(accum[h])
                self.corrections_[h] = arr.mean()
                self.std_[h]          = arr.std()
                self.count_[h]        = len(arr)

        self.fitted_ = True
        self._print_summary()
        return self

    def _print_summary(self):
        print("\n[피크 보정] 시간대별 편향 (정규화 스케일)")
        print(f"  {'Hour':>4}  {'Correction':>12}  {'Std':>8}  {'Count':>7}  {'Peak':>5}")
        print("  " + "-" * 45)
        for h in range(24):
            tag = "<-- PEAK" if h in PEAK_HOURS else ""
            print(
                f"  {h:4d}  {self.corrections_[h]:+12.5f}  "
                f"{self.std_[h]:8.5f}  {self.count_[h]:7.0f}  {tag}"
            )

    # ── 추론 ──────────────────────────────────────────────

    def transform(self, X: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X      : (N, seq_len, features)
        y_pred : (N, 24)  정규화된 예측값

        Returns
        -------
        y_corrected : (N, 24)
        """
        if not self.fitted_:
            raise RuntimeError("fit()을 먼저 호출하세요.")

        h_starts   = _decode_start_hours(X)
        corrected  = y_pred.copy()
        step_idx   = np.arange(24)                      # (24,)

        # 각 시퀀스의 실제 시각 매핑 → (N, 24)
        actual_hours = (h_starts[:, None] + step_idx[None, :]) % 24  # broadcast
        correction_vals = self.corrections_[actual_hours]             # (N, 24)
        corrected += correction_vals

        return corrected

    # ── 저장 / 로드 ────────────────────────────────────────

    def save(self, path: str):
        joblib.dump({
            "corrections": self.corrections_,
            "std":         self.std_,
            "count":       self.count_,
        }, path)
        print(f"[피크 보정] 저장 완료: {path}")

    @classmethod
    def load(cls, path: str) -> "HourBiasCorrector":
        data = joblib.load(path)
        obj = cls()
        obj.corrections_ = data["corrections"]
        obj.std_          = data["std"]
        obj.count_        = data["count"]
        obj.fitted_       = True
        return obj
