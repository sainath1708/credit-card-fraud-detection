from __future__ import annotations

import json
import logging
import os
import time
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from schemas import (
    BatchPredictInput,
    BatchPredictResponse,
    HealthResponse,
    PredictionResponse,
    TransactionInput,
)

LOGGER_NAME = "fraud_api"
logger = logging.getLogger(LOGGER_NAME)
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = Path(os.getenv("ARTIFACT_DIR", BASE_DIR / "artifacts"))
MODEL_FILENAME = "fraud_random_forest_model.joblib"
SCALER_FILENAME = "scaler.joblib"
METADATA_FILENAME = "model_metadata.json"


@dataclass(frozen=True)
class ModelBundle:
    model: Any
    scaler: Any
    feature_names: list[str]
    optimal_threshold: float
    model_version: str
    training_timestamp: str | None
    metadata: dict


def load_bundle(artifact_dir: Path = ARTIFACT_DIR) -> ModelBundle:
    paths = {
        "model": artifact_dir / MODEL_FILENAME,
        "scaler": artifact_dir / SCALER_FILENAME,
        "metadata": artifact_dir / METADATA_FILENAME,
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Missing model artifact(s): "
            + ", ".join(missing)
            + ". Run `python train_pipeline.py` first."
        )

    model = joblib.load(paths["model"])
    scaler = joblib.load(paths["scaler"])
    metadata = json.loads(paths["metadata"].read_text(encoding="utf-8"))

    required = {"optimal_threshold", "feature_names", "model_version"}
    absent = sorted(required.difference(metadata))
    if absent:
        raise ValueError(f"model_metadata.json is missing key(s): {', '.join(absent)}")

    feature_names = list(metadata["feature_names"])
    threshold = float(metadata["optimal_threshold"])
    if not 0.0 < threshold < 1.0:
        raise ValueError(f"optimal_threshold must be in (0, 1); got {threshold}")

    if getattr(model, "n_features_in_", len(feature_names)) != len(feature_names):
        raise ValueError(
            f"Model expects {model.n_features_in_} features but metadata lists "
            f"{len(feature_names)}."
        )
    if getattr(scaler, "n_features_in_", 2) != 2:
        raise ValueError("scaler.joblib must be fitted on exactly ['Time', 'Amount'].")

    logger.info(
        "Loaded model version %s (%d features, threshold=%.4f) from %s",
        metadata["model_version"],
        len(feature_names),
        threshold,
        artifact_dir.resolve(),
    )
    return ModelBundle(
        model=model,
        scaler=scaler,
        feature_names=feature_names,
        optimal_threshold=threshold,
        model_version=str(metadata["model_version"]),
        training_timestamp=metadata.get("training_timestamp_utc"),
        metadata=metadata,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = None
    app.state.load_error = None
    app.state.started_at = time.time()

    try:
        app.state.bundle = load_bundle()
    except (FileNotFoundError, ValueError, OSError) as exc:
        app.state.load_error = str(exc)
        logger.error("Model failed to load: %s", exc)
        logger.error("API will start in degraded mode; /predict returns 503.")
    except Exception as exc:
        app.state.load_error = f"{type(exc).__name__}: {exc}"
        logger.exception("Unexpected error while loading model artifacts")
        logger.error("API will start in degraded mode; /predict returns 503.")

    yield

    app.state.bundle = None
    logger.info("Shutdown complete; model released.")


app = FastAPI(
    title="Credit Card Fraud Detection API",
    description=(
        "Real-time scoring for credit-card transactions. Probabilities come from a "
        "Random Forest trained on SMOTE-balanced data; the decision threshold is "
        "optimised against a business cost matrix (FN=$100, FP=$2)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

_origins = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "*").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins,
    allow_credentials=_origins != ["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


def _get_bundle(request: Request) -> ModelBundle:
    bundle: ModelBundle | None = getattr(request.app.state, "bundle", None)
    if bundle is None:
        detail = getattr(request.app.state, "load_error", None) or "model not loaded"
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Inference unavailable: {detail}",
        )
    return bundle


def _score(bundle: ModelBundle, payloads: list[TransactionInput]) -> np.ndarray:
    frame = pd.DataFrame([payload.model_dump() for payload in payloads])

    scaled_values = bundle.scaler.transform(frame[["Time", "Amount"]])
    frame[["Time", "Amount"]] = scaled_values
    frame = frame.rename(columns={"Time": "scaled_time", "Amount": "scaled_amount"})

    missing = [name for name in bundle.feature_names if name not in frame.columns]
    if missing:
        raise RuntimeError(f"Feature construction failed; missing {missing}")

    probabilities = bundle.model.predict_proba(frame[bundle.feature_names])[:, 1]
    return np.asarray(probabilities, dtype=float)


def _risk_level(probability: float, threshold: float) -> str:
    if probability >= threshold:
        return "HIGH"
    if probability >= threshold * 0.5:
        return "MEDIUM"
    return "LOW"


def _to_response(
    probability: float, threshold: float, execution_time_ms: float
) -> PredictionResponse:
    return PredictionResponse(
        is_fraud=bool(probability >= threshold),
        fraud_probability=round(float(probability), 6),
        decision_threshold_used=round(float(threshold), 6),
        risk_level=_risk_level(probability, threshold),
        execution_time_ms=round(float(execution_time_ms), 3),
    )


@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "credit-card-fraud-detection",
        "docs": "/docs",
        "health": "/health",
    }


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={503: {"model": HealthResponse, "description": "Model unavailable"}},
    tags=["ops"],
)
def health(request: Request, response: Response) -> HealthResponse:
    bundle: ModelBundle | None = getattr(request.app.state, "bundle", None)
    if bundle is None:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
        return HealthResponse(status="degraded", model_loaded=False)
    return HealthResponse(
        status="ok",
        model_loaded=True,
        model_version=bundle.model_version,
        decision_threshold=bundle.optimal_threshold,
    )


@app.post(
    "/predict",
    response_model=PredictionResponse,
    responses={
        422: {"description": "Validation error"},
        503: {"description": "Model not loaded"},
    },
    tags=["inference"],
)
def predict(payload: TransactionInput, request: Request) -> PredictionResponse:
    bundle = _get_bundle(request)
    started = time.perf_counter()

    try:
        probability = float(_score(bundle, [payload])[0])
    except Exception as exc:
        logger.exception("Inference failed for /predict")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Scoring failed",
        ) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    response = _to_response(probability, bundle.optimal_threshold, elapsed_ms)
    logger.info(
        "prediction is_fraud=%s risk=%s latency_ms=%.2f",
        response.is_fraud,
        response.risk_level,
        elapsed_ms,
    )
    return response


@app.post(
    "/predict-batch",
    response_model=BatchPredictResponse,
    responses={
        422: {"description": "Validation error (1-100 transactions)"},
        503: {"description": "Model not loaded"},
    },
    tags=["inference"],
)
def predict_batch(
    payload: BatchPredictInput, request: Request
) -> BatchPredictResponse:
    bundle = _get_bundle(request)
    started = time.perf_counter()

    try:
        probabilities = _score(bundle, payload.transactions)
    except Exception as exc:
        logger.exception("Inference failed for /predict-batch")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Batch scoring failed",
        ) from exc

    elapsed_ms = (time.perf_counter() - started) * 1000
    per_item_ms = elapsed_ms / len(probabilities)

    predictions = [
        _to_response(probability, bundle.optimal_threshold, per_item_ms)
        for probability in probabilities
    ]

    flagged = sum(prediction.is_fraud for prediction in predictions)
    logger.info(
        "batch scored count=%d flagged=%d latency_ms=%.2f",
        len(predictions),
        flagged,
        elapsed_ms,
    )
    return BatchPredictResponse(
        count=len(predictions),
        predictions=predictions,
        execution_time_ms=round(elapsed_ms, 3),
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "Internal server error"},
    )


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", "8000")),
        log_level=os.getenv("LOG_LEVEL", "info").lower(),
    )
