"""Demo dự báo từ model đã lưu (mặc định: models/model.joblib).

Chạy từ thư mục gốc repo:

    python experiments/predict_demo.py
    python experiments/predict_demo.py --horizon 14
    python experiments/predict_demo.py --as-of 2016-06-01 --horizon 7

`--as-of` cắt history tại ngày đó để mô phỏng dự báo tại một mốc thời gian.
Chạy `python train.py` trước nếu chưa có model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Cho phép chạy trực tiếp `python experiments/predict_demo.py` từ thư mục gốc repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_daily_temperature  # noqa: E402
from src.forecast import DEFAULT_MODEL_PATH, Forecaster  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Dự báo nhiệt độ từ model đã lưu."
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=7,
        help="Số ngày cần dự báo, 1..30 (mặc định: 7)",
    )
    parser.add_argument(
        "--as-of",
        default=None,
        metavar="YYYY-MM-DD",
        help="Cắt history tại ngày này để mô phỏng dự báo tại mốc thời gian",
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help=f"Đường dẫn model bundle (mặc định: {DEFAULT_MODEL_PATH})",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if not 1 <= args.horizon <= 30:
        raise SystemExit("--horizon phải trong khoảng 1..30")
    if not args.model.exists():
        raise SystemExit(
            f"Không tìm thấy model: {args.model}\n"
            "Chạy `python train.py` trước để tạo model."
        )

    history = load_daily_temperature()
    if args.as_of:
        try:
            as_of = pd.Timestamp(args.as_of)
        except ValueError as exc:
            raise SystemExit(f"--as-of không hợp lệ: {args.as_of}") from exc
        history = history.loc[:as_of]
        if history.empty:
            raise SystemExit(f"Không có dữ liệu trước {as_of.date()}")

    forecaster = Forecaster(args.model)
    preds = forecaster.predict(history, horizon=args.horizon)

    print(
        f"History đến {history.index.max().date()} ({len(history)} ngày), "
        f"model trained_until={forecaster.trained_until}"
    )
    print(f"{'date':<12}{'temperature':>12}")
    for idx, val in preds.items():
        print(f"{idx.strftime('%Y-%m-%d'):<12}{val:>12.2f}")


if __name__ == "__main__":
    main()
