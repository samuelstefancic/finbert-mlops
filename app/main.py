"""
Sentiment Prediction API.

Endpoints:
    GET  /health          — service health status
    POST /predict         — single headline sentiment
    POST /predict/batch   — batch headline sentiment
    GET  /metrics         — Prometheus metrics
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest
from pydantic import BaseModel
from utils import load_classifier

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
logger = logging.getLogger("finbert-api")


def log(level: str, message: str, **kwargs) -> None:
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": level,
        "message": message,
        **kwargs,
    }
    logger.log(getattr(logging, level), json.dumps(entry))


PREDICTION_REQUESTS = Counter(
    "prediction_requests_total", "Completed predictions", ["sentiment"]
)
PREDICTION_LATENCY = Histogram(
    "prediction_latency_ms", "Inference latency per request in milliseconds",
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000),
)
PREDICTION_ERRORS = Counter("prediction_errors_total", "Failed inference requests")

load_dotenv()

classifiers = {}


class PredictRequest(BaseModel):
    text: str


class PredictBatchRequest(BaseModel):
    texts: list[str]


class PredictionResult(BaseModel):
    text: str
    sentiment: str
    confidence: float
    latency_ms: float


@asynccontextmanager
async def lifespan(app: FastAPI):
    log("INFO", "Loading model...")
    classifiers["sentiment"] = load_classifier()
    log("INFO", "Model loaded successfully")
    try:
        yield
    finally:
        classifiers.clear()
        log("INFO", "Model unloaded")


app = FastAPI(title="Sentiment Analysis API", lifespan=lifespan)


def run_predictions(texts: list[str]) -> list[PredictionResult]:
    try:
        if any(len(text) > 2000 for text in texts):
            log("WARNING", "Input exceeds 2000 characters; truncation enabled")
        started = time.perf_counter()
        outputs = classifiers["sentiment"](texts, truncation=True)
        latency_ms = (time.perf_counter() - started) * 1000
        results = [
            PredictionResult(
                text=text,
                sentiment=output["label"],
                confidence=float(output["score"]),
                latency_ms=latency_ms,
            )
            for text, output in zip(texts, outputs)
        ]
        PREDICTION_LATENCY.observe(latency_ms)
        for result in results:
            PREDICTION_REQUESTS.labels(sentiment=result.sentiment).inc()
            log(
                "INFO",
                "Prediction completed",
                sentiment=result.sentiment,
                confidence=result.confidence,
                latency_ms=result.latency_ms,
            )
        return results
    except Exception as error:
        log("ERROR", "Prediction failed", error_type=type(error).__name__)
        PREDICTION_ERRORS.inc()
        raise


@app.get("/metrics")
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


def require_model() -> None:
    if classifiers.get("sentiment") is None:
        raise HTTPException(status_code=503, detail="Model is not loaded")


@app.get("/health")
def health():
    require_model()
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResult)
def predict(request: PredictRequest):
    require_model()
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="Text must not be empty")
    return run_predictions([request.text])[0]


@app.post("/predict/batch", response_model=list[PredictionResult])
def predict_batch(request: PredictBatchRequest):
    require_model()
    if not request.texts or any(not text.strip() for text in request.texts):
        raise HTTPException(status_code=422, detail="Texts must contain non-empty headlines")
    return run_predictions(request.texts)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv("API_HOST", "0.0.0.0"),
        port=int(os.getenv("API_PORT", 8000)),
    )
