from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import load_daily_temperature, time_split
from src.features import make_features

logger = logging.getLogger(__name__)

MODEL_PATH = Path("models/model.joblib")

# "delta": model học y - lag_1 (khuyến nghị, tốt hơn hẳn dự báo trực tiếp y).
# "absolute": model học trực tiếp y.
TARGET_MODE = "delta"
RIDGE_ALPHA = 1.0


def build_model() -> Pipeline:
    # Ridge + StandardScaler thắng HGB trên dữ liệu này (MAE 1.681 vs ~1.93),
    # đồng thời nhanh, nhẹ và ổn định hơn với chuỗi ngắn.
    return make_pipeline(StandardScaler(), Ridge(alpha=RIDGE_ALPHA))


def _target(df: pd.DataFrame, target_mode: str) -> pd.Series:
    if target_mode == "delta":
        return df["y"] - df["lag_1"]
    if target_mode == "absolute":
        return df["y"]
    raise ValueError(f"target_mode không hợp lệ: {target_mode}")


def _predict(model, df: pd.DataFrame, features: list[str], target_mode: str) -> pd.Series:
    raw = model.predict(df[features])
    if target_mode == "delta":
        raw = df["lag_1"].to_numpy() + raw
    return pd.Series(raw, index=df.index)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    s = load_daily_temperature()
    _, test_s = time_split(s, test_days=365)

    feat = make_features(s)
    features = [c for c in feat.columns if c != "y"]

    cutoff = test_s.index.min()
    train_df = feat.loc[feat.index < cutoff].dropna()
    test_df = feat.loc[feat.index >= cutoff].dropna()

    if len(train_df) < 100 or test_df.empty:
        raise RuntimeError("Dữ liệu train/test không đủ để huấn luyện")

    # Validation theo thời gian (90 ngày cuối của train) — KHÔNG random split
    val_cut = train_df.index.max() - pd.Timedelta(days=90)
    fit_df = train_df.loc[train_df.index <= val_cut]
    val_df = train_df.loc[train_df.index > val_cut]

    model = build_model()
    model.fit(fit_df[features], _target(fit_df, TARGET_MODE))

    for name, part in (("val", val_df), ("test", test_df)):
        if part.empty:
            continue
        pred = _predict(model, part, features, TARGET_MODE)
        logger.info(
            "%s MAE=%.3f RMSE=%.3f",
            name,
            mean_absolute_error(part["y"], pred),
            mean_squared_error(part["y"], pred) ** 0.5,
        )

    # Baseline persistence: dự báo hôm nay = nhiệt độ hôm qua
    logger.info(
        "baseline persistence MAE=%.3f",
        mean_absolute_error(test_df["y"], test_df["lag_1"]),
    )

    # Refit trên TOÀN BỘ dữ liệu (kể cả holdout) cho production.
    # Chỉ số val/test ở trên là của model chưa thấy holdout, nên vẫn đánh giá khách quan.
    full_df = feat.dropna()
    model.fit(full_df[features], _target(full_df, TARGET_MODE))

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "features": features,
            "target_mode": TARGET_MODE,
            "trained_until": full_df.index.max(),
        },
        MODEL_PATH,
    )
    logger.info(
        "Đã lưu model: %s (%d features, target_mode=%s, trained_until=%s)",
        MODEL_PATH,
        len(features),
        TARGET_MODE,
        full_df.index.max(),
    )


if __name__ == "__main__":
    main()
