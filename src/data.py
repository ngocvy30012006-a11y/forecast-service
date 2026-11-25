from __future__ import annotations

import io
import logging
import urllib.request
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

URL = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
DATE_COL = "Date Time"
TARGET_COL = "T (degC)"
DEFAULT_CACHE = Path("data/jena_climate_2009_2016.csv.zip")


def _download(url: str, cache: Path, timeout: int = 30) -> bytes:
    """Tải file về cache (atomic write), trả về bytes. Không tải lại nếu đã có."""
    if cache.exists():
        logger.info("Dùng cache dữ liệu: %s", cache)
        return cache.read_bytes()

    logger.info("Tải dữ liệu từ %s", url)
    req = urllib.request.Request(url, headers={"User-Agent": "forecast-service/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        content = resp.read()  # HTTPError tự raise với status không phải 2xx

    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(cache.suffix + ".tmp")
    tmp.write_bytes(content)
    tmp.replace(cache)  # atomic: tránh cache hỏng nếu process bị kill
    return content


def load_daily_temperature(cache: Path = DEFAULT_CACHE) -> pd.Series:
    """Trả về chuỗi nhiệt độ trung bình ngày (tz-naive), index liên tục theo ngày."""
    raw = _download(URL, cache)

    df = pd.read_csv(
        io.BytesIO(raw),
        compression="zip",                 # chỉ định rõ, không dựa vào suy luận từ URL
        usecols=[DATE_COL, TARGET_COL],    # chỉ đọc 2/14 cột: giảm ~7x RAM & thời gian
        parse_dates=[DATE_COL],
        date_format="%d.%m.%Y %H:%M:%S",   # pandas >= 2.0 (bản cũ dùng format=)
    )
    df = df.sort_values(DATE_COL).drop_duplicates(subset=DATE_COL, keep="last")

    s = (
        df.set_index(DATE_COL)[TARGET_COL]
        .resample("D")
        .mean()
        .interpolate(method="time", limit_direction="both")
        .asfreq("D")
    )

    missing = int(s.isna().sum())
    if missing:
        logger.warning("Còn %d ngày thiếu dữ liệu sau nội suy", missing)

    s.name = "temp"
    return s


def time_split(s: pd.Series, test_days: int = 365) -> tuple[pd.Series, pd.Series]:
    """Chia train/test theo mốc thời gian (không shuffle, không chia theo số dòng)."""
    if isinstance(test_days, bool) or not isinstance(test_days, int):
        raise TypeError("test_days phải là int")
    if test_days <= 0:
        raise ValueError("test_days phải > 0")
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("s.index phải là DatetimeIndex")
    if not s.index.is_monotonic_increasing:
        raise ValueError("s.index phải tăng dần theo thời gian")

    cutoff = s.index.max() - pd.Timedelta(days=test_days)
    train, test = s.loc[s.index <= cutoff], s.loc[s.index > cutoff]
    if train.empty or test.empty:
        raise ValueError(
            f"test_days={test_days} không hợp lệ với dữ liệu {len(s)} điểm"
        )
    return train, test
