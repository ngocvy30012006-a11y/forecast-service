from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Annotated

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from src.data import load_daily_temperature
from src.forecast import Forecaster

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

state: dict = {}


@asynccontextmanager
async def lifespan(_: FastAPI):
    # Load model + dữ liệu MỘT LẦN lúc startup, không load theo request
    state["forecaster"] = Forecaster()
    state["history"] = load_daily_temperature()
    logger.info("Service sẵn sàng")
    try:
        yield
    finally:
        state.clear()


app = FastAPI(title="Forecast Service", version="1.0.0", lifespan=lifespan)


class ForecastPoint(BaseModel):
    date: str
    temperature: float


class ForecastResponse(BaseModel):
    horizon: int
    predictions: list[ForecastPoint]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": "forecaster" in state}


@app.get("/forecast", response_model=ForecastResponse)
def forecast(horizon: Annotated[int, Query(ge=1, le=30)] = 7) -> ForecastResponse:
    forecaster: Forecaster | None = state.get("forecaster")
    history: pd.Series | None = state.get("history")
    if forecaster is None or history is None:
        raise HTTPException(status_code=503, detail="Service chưa sẵn sàng")

    try:
        preds = forecaster.predict(history, horizon=horizon)
    except (ValueError, TypeError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    return ForecastResponse(
        horizon=horizon,
        predictions=[
            ForecastPoint(date=idx.strftime("%Y-%m-%d"), temperature=float(val))
            for idx, val in preds.items()
        ],
    )
