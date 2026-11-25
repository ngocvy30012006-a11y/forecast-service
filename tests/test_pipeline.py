from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data import time_split
from src.features import MIN_HISTORY, make_features, make_inference_features
from src.forecast import Forecaster


def _series(n: int = 100) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    return pd.Series(np.arange(n, dtype=float), index=idx, name="temp")


@pytest.mark.parametrize("bad", [0, -1, -365])
def test_time_split_rejects_non_positive(bad: int) -> None:
    with pytest.raises(ValueError):
        time_split(_series(), test_days=bad)


def test_time_split_rejects_too_large() -> None:
    with pytest.raises(ValueError):
        time_split(_series(10), test_days=365)


def test_time_split_sizes() -> None:
    train, test = time_split(_series(100), test_days=30)
    assert len(train) == 70
    assert len(test) == 30


def test_features_no_leakage() -> None:
    s = _series(100)
    f = make_features(s)
    assert f["lag_1"].iloc[10] == f["y"].iloc[9]


def test_features_too_short() -> None:
    with pytest.raises(ValueError):
        make_features(_series(MIN_HISTORY - 1))


def test_inference_features_single_row() -> None:
    s = _series(100)
    f = make_inference_features(s)
    assert len(f) == 1
    assert f.index[0] == s.index.max() + pd.Timedelta(days=1)
    assert not f.drop(columns="y").isna().any().any()


def test_forecaster_delta_roundtrip(tmp_path) -> None:
    n = 800
    idx = pd.date_range("2018-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    s = pd.Series(
        10 + 10 * np.sin(2 * np.pi * np.arange(n) / 365.25) + rng.normal(0, 1, n),
        index=idx,
        name="temp",
    )
    feat = make_features(s)
    features = [c for c in feat.columns if c != "y"]
    df = feat.dropna()

    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(df[features], df["y"] - df["lag_1"])

    path = tmp_path / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "features": features,
            "target_mode": "delta",
            "trained_until": df.index.max(),
        },
        path,
    )

    forecaster = Forecaster(path)
    preds = forecaster.predict(s, horizon=5)
    assert len(preds) == 5
    assert preds.index[0] == s.index.max() + pd.Timedelta(days=1)
    assert np.isfinite(preds.to_numpy()).all()
    assert preds.name == "pred"

    with pytest.raises(ValueError):
        forecaster.predict(s, horizon=0)
    with pytest.raises(TypeError):
        forecaster.predict(s, horizon=1.5)
