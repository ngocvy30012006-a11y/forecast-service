"""So sánh các cấu hình model trên holdout theo thời gian.

Chạy từ thư mục gốc repo:

    python experiments/experiment.py
    python experiments/experiment.py --model hgb --target absolute
    python experiments/experiment.py --rolling 7 14 28
    python experiments/experiment.py --alpha 10 --test-days 365

Script KHÔNG ghi đè model production (models/model.joblib). Muốn cập nhật model
thì chạy `python train.py`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

# Cho phép chạy trực tiếp `python experiments/experiment.py` từ thư mục gốc repo.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_daily_temperature, time_split  # noqa: E402
from src.features import make_features  # noqa: E402


def build_model(name: str, alpha: float) -> Pipeline:
    if name == "ridge":
        return make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    if name == "hgb":
        return make_pipeline(
            HistGradientBoostingRegressor(
                max_iter=300,
                learning_rate=0.05,
                random_state=0,
            )
        )
    raise ValueError(f"model không hợp lệ: {name}")


def make_target(df: pd.DataFrame, mode: str) -> pd.Series:
    if mode == "delta":
        return df["y"] - df["lag_1"]
    if mode == "absolute":
        return df["y"]
    raise ValueError(f"target không hợp lệ: {mode}")


def predict(
    model: Pipeline, df: pd.DataFrame, features: list[str], mode: str
) -> pd.Series:
    raw = model.predict(df[features])
    if mode == "delta":
        raw = df["lag_1"].to_numpy() + raw
    return pd.Series(raw, index=df.index)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Đánh giá model dự báo nhiệt độ trên holdout theo thời gian."
    )
    parser.add_argument(
        "--model",
        choices=["ridge", "hgb"],
        default="ridge",
        help="Loại model (mặc định: ridge)",
    )
    parser.add_argument(
        "--target",
        choices=["delta", "absolute"],
        default="delta",
        help="Học y - lag_1 (delta) hay học trực tiếp y (absolute). Mặc định: delta",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1.0,
        help="alpha của Ridge (mặc định: 1.0)",
    )
    parser.add_argument(
        "--rolling",
        type=int,
        nargs="+",
        default=None,
        metavar="W",
        help="Thêm rolling mean/std với các cửa sổ W (vd: --rolling 7 14 28)",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=365,
        help="Số ngày holdout ở cuối chuỗi (mặc định: 365)",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    if args.test_days <= 0:
        raise SystemExit("--test-days phải > 0")
    if args.alpha < 0:
        raise SystemExit("--alpha phải >= 0")

    roll_windows: tuple[int, ...] = ()
    if args.rolling:
        roll_windows = tuple(sorted(set(args.rolling)))
        if any(w < 2 for w in roll_windows):
            raise SystemExit("--rolling yêu cầu mọi cửa sổ >= 2")

    s = load_daily_temperature()
    _, test_s = time_split(s, test_days=args.test_days)

    feat = make_features(s, roll_windows=roll_windows)
    features = [c for c in feat.columns if c != "y"]

    cutoff = test_s.index.min()
    train_df = feat.loc[feat.index < cutoff].dropna()
    test_df = feat.loc[feat.index >= cutoff].dropna()

    if len(train_df) < 100 or test_df.empty:
        raise SystemExit("Dữ liệu train/test không đủ để đánh giá")

    model = build_model(args.model, args.alpha)
    model.fit(train_df[features], make_target(train_df, args.target))
    pred = predict(model, test_df, features, args.target)

    mae = mean_absolute_error(test_df["y"], pred)
    rmse = mean_squared_error(test_df["y"], pred) ** 0.5
    base_mae = mean_absolute_error(test_df["y"], test_df["lag_1"])
    base_rmse = mean_squared_error(test_df["y"], test_df["lag_1"]) ** 0.5

    print(
        f"Cấu hình: model={args.model} target={args.target} "
        f"alpha={args.alpha} rolling={roll_windows} test_days={args.test_days}"
    )
    print(
        f"Train: {len(train_df)} mẫu ({train_df.index.min().date()} → "
        f"{train_df.index.max().date()})"
    )
    print(
        f"Test : {len(test_df)} mẫu ({test_df.index.min().date()} → "
        f"{test_df.index.max().date()})"
    )
    print(f"Số feature: {len(features)} -> {features}")
    print()
    print(f"{'Model':<28}{'MAE':>8}{'RMSE':>10}")
    print(f"{'Baseline persistence':<28}{base_mae:>8.3f}{base_rmse:>10.3f}")
    print(f"{'Model':<28}{mae:>8.3f}{rmse:>10.3f}")
    print()
    delta_pct = (base_mae - mae) / base_mae * 100
    print(f"Cải thiện MAE so với baseline: {delta_pct:+.1f}%")
    print()
    print("5 dự báo mẫu:")
    print(f"{'date':<12}{'actual':>9}{'pred':>9}{'err':>9}")
    sample = test_df.assign(pred=pred).head(5)
    for idx, row in sample.iterrows():
        print(
            f"{idx.strftime('%Y-%m-%d'):<12}{row['y']:>9.2f}"
            f"{row['pred']:>9.2f}{row['pred'] - row['y']:>9.2f}"
        )


if __name__ == "__main__":
    main()
