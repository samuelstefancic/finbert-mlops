"""Send the prepared stream to the API and log window metrics in MLflow."""

import os
import time

import mlflow
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = os.getenv("API_URL", f"http://127.0.0.1:{os.getenv('API_PORT', '8000')}")
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "50"))
SLEEP_MS = float(os.getenv("SLEEP_MS", "100"))


def predict(text: str) -> dict:
    response = requests.post(f"{API_URL}/predict", json={"text": text}, timeout=30)
    response.raise_for_status()
    return response.json()


def log_window(window: list[dict], window_idx: int) -> None:
    """Log percentages, confidence and latency for a completed observation window."""
    if not window:
        return
    count = len(window)
    metrics = {
        f"pct_{sentiment}": 100 * sum(
            item["sentiment"] == sentiment for item in window
        ) / count
        for sentiment in ("positive", "negative", "neutral")
    }
    metrics["avg_confidence"] = sum(item["confidence"] for item in window) / count
    metrics["avg_latency_ms"] = sum(item["latency_ms"] for item in window) / count
    mlflow.log_metrics(metrics, step=window_idx)
    print(f"Window {window_idx}: {count} predictions, {metrics}", flush=True)


def main():
    if WINDOW_SIZE <= 0 or SLEEP_MS < 0:
        raise ValueError("WINDOW_SIZE must be positive and SLEEP_MS non-negative")
    texts = pd.read_csv("data/stream.csv")["text"].tolist()
    if not texts:
        raise ValueError("The monitoring stream is empty")
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    with mlflow.start_run(run_name="production-monitoring"):
        mlflow.set_tag("task", "monitoring")
        mlflow.log_params({"window_size": WINDOW_SIZE, "api_url": API_URL})
        window = []
        window_idx = 0
        for text in texts:
            window.append(predict(text))
            if len(window) == WINDOW_SIZE:
                log_window(window, window_idx)
                window = []
                window_idx += 1
            time.sleep(SLEEP_MS / 1000)
        if window:
            log_window(window, window_idx)


if __name__ == "__main__":
    main()
