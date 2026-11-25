from __future__ import annotations

import logging
from pathlib import Path

import joblib
import pandas as pd

from src.features import make_inference_features

logger = logging.getLogger(__name__)

DEFAULT_MODEL_PATH = Path("models/model.joblib")


class Forecaster:
    """Bọc model + danh sách feature để tránh train/serve skew."""

    def __init__(self, model_path: Path = DEFAULT_MODEL_PATH) -> None:
        bundle = joblib.load(model_path)
        if not isinstance(bundle, dict) or "model" not in bundle or "features" not in bundle:
            raise ValueError(f"Model bundle không hợp lệ: {model_path}")

        self.model = bundle["model"]
        self.features: list[str] = list(bundle["features"])
        self.target_mode: str = bundle.get("target_mode", "absolute")
        self.trained_until: pd.Timestamp | None = bundle.get("trained_until")

        if self.target_mode not in ("absolute", "delta"):
            raise ValueError(f"target_mode không hợp lệ trong bundle: {self.target_mode}")
        if self.target_mode == "delta" and "lag_1" not in self.features:
            raise ValueError("target_mode=delta yêu cầu feature 'lag_1'")

        logger.info(
            "Đã load model (%d features, target_mode=%s), trained_until=%s",
            len(self.features),
            self.target_mode,
            self.trained_until,
        )

    def predict(self, history: pd.Series, horizon: int = 1) -> pd.Series:
        """Dự báo đệ quy horizon ngày tiếp theo sau history."""
        if isinstance(horizon, bool) or not isinstance(horizon, int):
            raise TypeError("horizon phải là int")
        if horizon < 1:
            raise ValueError("horizon phải >= 1")

        if self.trained_until is not None and history.index.max() > self.trained_until:
            logger.warning("History vượt quá trained_until; dự báo có thể kém tin cậy")

        work = history.copy()
        dates: list[pd.Timestamp] = []
        values: list[float] = []

        for _ in range(horizon):
            feats = make_inference_features(work)

            missing = [c for c in self.features if c not in feats.columns]
            if missing:
                raise ValueError(f"Thiếu feature: {missing}")

            row = feats[self.features]
            if row.isna().any().any():
                raise ValueError("History không đủ dài/liên tục để tạo feature")

            raw = float(self.model.predict(row)[0])
            if self.target_mode == "delta":
                raw += float(row["lag_1"].iloc[0])
            y_hat = raw
            next_idx = feats.index[0]

            dates.append(next_idx)
            values.append(y_hat)
            work = pd.concat(
                [work, pd.Series([y_hat], index=[next_idx], name=work.name)]
            )

        return pd.Series(values, index=pd.DatetimeIndex(dates), name="pred")
