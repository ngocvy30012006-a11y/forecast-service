from __future__ import annotations

import numpy as np
import pandas as pd

LAGS = [1, 2, 3, 7, 14, 28]
# Rolling features không được dùng mặc định: thực nghiệm trên holdout 2016 cho thấy
# thêm rolling mean/std làm MAE xấu đi (1.695 vs 1.681 khi chỉ dùng lags + season).
ROLL_WINDOWS: tuple[int, ...] = ()
MIN_HISTORY = max(LAGS + list(ROLL_WINDOWS)) + 2  # 30 điểm tối thiểu


def _required_history(roll_windows: tuple[int, ...]) -> int:
    return max(LAGS + list(roll_windows)) + 2


def _day_of_year(index: pd.DatetimeIndex) -> np.ndarray:
    """Tương thích pandas < 3 (dayofyear) và >= 3 (day_of_year)."""
    doy = getattr(index, "day_of_year", None)
    if doy is None:  # pandas < 3.0
        doy = index.dayofyear
    return np.asarray(doy)


def _validate_series(s: pd.Series, roll_windows: tuple[int, ...] = ROLL_WINDOWS) -> None:
    if not isinstance(s, pd.Series):
        raise TypeError("s phải là pd.Series")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("s.index phải là DatetimeIndex")
    if not s.index.is_monotonic_increasing:
        raise ValueError("s.index phải tăng dần")
    if s.index.has_duplicates:
        raise ValueError("s.index có timestamp trùng")

    needed = _required_history(roll_windows)
    if len(s) < needed:
        raise ValueError(f"Cần ít nhất {needed} điểm, chỉ có {len(s)}")


def make_features(
    s: pd.Series,
    *,
    roll_windows: tuple[int, ...] = ROLL_WINDOWS,
    dropna: bool = False,
) -> pd.DataFrame:
    """Tạo feature tại mỗi thời điểm t chỉ từ dữ liệu <= t-1 (không leakage)."""
    _validate_series(s, roll_windows)

    df = pd.DataFrame({"y": s})
    for lag in LAGS:
        df[f"lag_{lag}"] = s.shift(lag)

    past = s.shift(1)  # quan trọng: tránh leakage
    for w in roll_windows:
        rolling = past.rolling(w)
        df[f"roll_mean_{w}"] = rolling.mean()
        df[f"roll_std_{w}"] = rolling.std()

    # Mùa vụ: chuẩn hoá theo độ dài năm thực tế để không lệch pha ở năm nhuận
    doy0 = _day_of_year(s.index) - 1               # 0..364/365
    year_len = np.where(s.index.is_leap_year, 366, 365)
    frac = doy0 / year_len                          # 0..1, khép kín chu kỳ
    df["sin_doy"] = np.sin(2 * np.pi * frac)
    df["cos_doy"] = np.cos(2 * np.pi * frac)

    return df.dropna() if dropna else df


def make_inference_features(
    history: pd.Series,
    *,
    roll_windows: tuple[int, ...] = ROLL_WINDOWS,
) -> pd.DataFrame:
    """Feature cho đúng ngày kế tiếp sau history.index.max() (1 hàng).

    Dùng placeholder NaN cho ngày tương lai để shift/rolling tính đúng như lúc train.
    Với horizon > 1, hãy gọi lặp lại (recursive) — xem Forecaster.predict.
    """
    _validate_series(history, roll_windows)

    next_idx = history.index.max() + pd.Timedelta(days=1)
    extended = pd.concat([history, pd.Series([np.nan], index=[next_idx])])
    feats = make_features(extended, roll_windows=roll_windows)
    return feats.loc[[next_idx]]
