"""Integration tests for the Sentiment Analysis API."""

import pytest
from fastapi.testclient import TestClient

from app.main import classifiers


@pytest.mark.parametrize("text", [
    "The company reported record profits this quarter.",
    "The company reported a substantial loss.",
    "The company published its annual report.",
])
def test_predict_returns_valid_response(client: TestClient, text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    result = response.json()
    assert set(result) == {"text", "sentiment", "confidence", "latency_ms"}
    assert result["text"] == text
    assert result["sentiment"] in {"positive", "neutral", "negative"}
    assert 0 <= result["confidence"] <= 1
    assert result["latency_ms"] >= 0


def test_predict_batch(client: TestClient):
    texts = ["Profits increased this quarter.", "Revenue declined sharply."]
    response = client.post("/predict/batch", json={"texts": texts})
    assert response.status_code == 200
    results = response.json()
    assert len(results) == len(texts)
    for result, text in zip(results, texts):
        assert set(result) == {"text", "sentiment", "confidence", "latency_ms"}
        assert result["text"] == text
        assert result["sentiment"] in {"positive", "neutral", "negative"}
        assert 0 <= result["confidence"] <= 1
        assert result["latency_ms"] >= 0


def test_health(client: TestClient):
    assert client.get("/health").json() == {"status": "ok"}


@pytest.mark.parametrize("path,payload", [
    ("/predict", {"text": ""}),
    ("/predict", {"text": "   "}),
    ("/predict/batch", {"texts": []}),
    ("/predict/batch", {"texts": ["Valid headline", ""]}),
])
def test_empty_inputs_return_422(client: TestClient, path, payload):
    assert client.post(path, json=payload).status_code == 422


@pytest.mark.parametrize("path,payload", [
    ("/health", None),
    ("/predict", {"text": "The company reported record profits."}),
    ("/predict/batch", {"texts": ["The company reported record profits."]}),
])
def test_unloaded_model_returns_503(client: TestClient, path, payload):
    model = classifiers.pop("sentiment")
    try:
        response = client.get(path) if payload is None else client.post(path, json=payload)
        assert response.status_code == 503
    finally:
        classifiers["sentiment"] = model
