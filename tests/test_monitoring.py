"""Checks for monitoring and the model promotion gate."""

import json
from types import SimpleNamespace

import pytest

from app import main as api
from monitoring import stream
from scripts import promote


def metric_value(name, labels=None):
    from prometheus_client.parser import text_string_to_metric_families

    for family in text_string_to_metric_families(api.generate_latest().decode()):
        for sample in family.samples:
            if sample.name == name and sample.labels == (labels or {}):
                return sample.value
    return 0


def test_metrics_exposes_predictions(client):
    text = "The company reported record profits."
    before = metric_value("prediction_latency_ms_count")
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 200
    assert metric_value("prediction_latency_ms_count") == before + 1
    metrics = client.get("/metrics")
    assert metrics.status_code == 200
    assert metrics.headers["content-type"].startswith("text/plain")
    assert "prediction_requests_total" in metrics.text


def test_prediction_errors_are_counted_and_logged(client, monkeypatch, caplog):
    def fail(*args, **kwargs):
        raise RuntimeError("inference failed")

    before = metric_value("prediction_errors_total")
    monkeypatch.setitem(api.classifiers, "sentiment", fail)
    with pytest.raises(RuntimeError, match="inference failed"):
        client.post("/predict", json={"text": "The company reported profits."})
    assert metric_value("prediction_errors_total") == before + 1
    entry = json.loads(caplog.records[-1].message)
    assert entry["level"] == "ERROR"
    assert entry["error_type"] == "RuntimeError"
    assert "timestamp" in entry


def test_monitoring_window_aggregates_metrics(monkeypatch):
    recorded = {}
    monkeypatch.setattr(
        stream.mlflow, "log_metrics",
        lambda metrics, step: recorded.update(metrics=metrics, step=step),
    )
    window = [
        {"sentiment": label, "confidence": confidence, "latency_ms": latency}
        for label, confidence, latency in [
            ("positive", 0.8, 10),
            ("positive", 0.6, 20),
            ("negative", 0.9, 30),
            ("neutral", 0.7, 40),
        ]
    ]
    stream.log_window(window, 3)
    assert recorded["step"] == 3
    assert recorded["metrics"] == pytest.approx({
        "pct_positive": 50,
        "pct_negative": 25,
        "pct_neutral": 25,
        "avg_confidence": 0.75,
        "avg_latency_ms": 25,
    })


@pytest.mark.parametrize("f1,expected", [(0.55, False), (0.56, True), (0.70, True)])
def test_promotion_respects_threshold(monkeypatch, f1, expected):
    promoted = []
    version = SimpleNamespace(version="1", run_id="evaluated-run")
    client = SimpleNamespace(
        search_model_versions=lambda query: [version],
        set_registered_model_alias=lambda *args: promoted.append(args),
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setenv("MODEL_NAME", "finbert")
    monkeypatch.setattr(promote, "MlflowClient", lambda: client)
    monkeypatch.setattr(promote, "load_params", lambda: {"f1_threshold": 0.56})
    monkeypatch.setattr(promote, "get_latest_f1", lambda *args: ("evaluated-run", f1))
    promote.main()
    assert bool(promoted) is expected
    if expected:
        assert promoted == [("finbert", "production", "1")]


def test_promotion_rejects_unmatched_model_version(monkeypatch):
    client = SimpleNamespace(
        search_model_versions=lambda query: [
            SimpleNamespace(version="2", run_id="another-run")
        ],
    )
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    monkeypatch.setattr(promote, "MlflowClient", lambda: client)
    monkeypatch.setattr(promote, "load_params", lambda: {"f1_threshold": 0.56})
    monkeypatch.setattr(promote, "get_latest_f1", lambda *args: ("evaluated-run", 0.7))
    with pytest.raises(RuntimeError, match="does not match"):
        promote.main()
