from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import app as app_module
from src.features import make_features
from src.forecast import Forecaster


def _synthetic_series(n: int = 400) -> pd.Series:
    idx = pd.date_range("2020-01-01", periods=n, freq="D")
    rng = np.random.default_rng(0)
    values = (
        10
        + 8 * np.sin(2 * np.pi * np.arange(n) / 365.25)
        + rng.normal(0, 1, n)
    )
    return pd.Series(values, index=idx, name="temp")


def _write_model(path, s: pd.Series) -> None:
    feat = make_features(s).dropna()
    features = [c for c in feat.columns if c != "y"]
    model = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
    model.fit(feat[features], feat["y"] - feat["lag_1"])
    joblib.dump(
        {
            "model": model,
            "features": features,
            "target_mode": "delta",
            "trained_until": feat.index.max(),
        },
        path,
    )


@pytest.fixture()
def client(monkeypatch, tmp_path) -> TestClient:
    s = _synthetic_series()
    path = tmp_path / "model.joblib"
    _write_model(path, s)

    # app.py load model + data lúc startup; thay bằng dữ liệu/model tạm để test
    # không phụ thuộc dataset 13MB hay models/model.joblib (đều bị gitignore).
    monkeypatch.setattr(app_module, "Forecaster", lambda: Forecaster(path))
    monkeypatch.setattr(app_module, "load_daily_temperature", lambda: s)

    with TestClient(app_module.app) as test_client:
        yield test_client


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_loaded": True}


def test_forecast_horizon(client: TestClient) -> None:
    r = client.get("/forecast", params={"horizon": 3})
    assert r.status_code == 200

    body = r.json()
    assert body["horizon"] == 3
    assert len(body["predictions"]) == 3

    dates = [p["date"] for p in body["predictions"]]
    assert dates == sorted(dates)
    assert dates[0] > "2020-01-01"
    assert all(isinstance(p["temperature"], float) for p in body["predictions"])


def test_forecast_default_horizon(client: TestClient) -> None:
    r = client.get("/forecast")
    assert r.status_code == 200
    assert len(r.json()["predictions"]) == 7


@pytest.mark.parametrize("horizon", [0, -1, 31])
def test_forecast_rejects_out_of_range(client: TestClient, horizon: int) -> None:
    assert client.get("/forecast", params={"horizon": horizon}).status_code == 422


def test_forecast_unavailable_before_startup() -> None:
    # Không enter context manager -> lifespan không chạy -> state rỗng -> 503
    test_client = TestClient(app_module.app)
    r = test_client.get("/forecast", params={"horizon": 1})
    assert r.status_code == 503
