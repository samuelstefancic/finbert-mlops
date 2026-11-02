"""
Evaluate sentiment model on the test set and register it
in the MLflow Model Registry.

Run:
    python scripts/evaluate.py
"""

import os

import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from transformers import pipeline

import mlflow
import mlflow.transformers


load_dotenv()


def load_test_data(test_path: str) -> tuple[list[str], list[str]]:
    df = pd.read_csv(test_path)
    return df["text"].tolist(), df["label"].tolist()


def build_classifier(model_id: str):
    print(f"Loading {model_id}...")
    return pipeline(
        "text-classification",
        model=model_id,
        tokenizer=model_id,
        truncation=True,
        max_length=512,
    )


def run_inference(classifier, texts: list[str]) -> list[str]:
    print(f"Running inference on {len(texts)} samples...")
    results = classifier(texts, batch_size=32)
    return [r["label"] for r in results]


def compute_metrics(y_true: list[str], y_pred: list[str]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
        "precision_weighted": precision_score(y_true, y_pred, average="weighted"),
        "recall_weighted": recall_score(y_true, y_pred, average="weighted"),
    }


def main():
    model_id = os.getenv("HF_MODEL_ID", "baptle/FinBERT_market_based")
    test_path = os.path.join("data", "test.csv")

    texts, y_true = load_test_data(test_path)
    classifier = build_classifier(model_id)
    y_pred = run_inference(classifier, texts)
    metrics = compute_metrics(y_true, y_pred)

    print("\nEvaluation Results:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))

    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation"))
    with mlflow.start_run(run_name="finbert-evaluation"):
        mlflow.set_tag("task", "evaluation")
        mlflow.log_params({"model_id": model_id, "test_samples": len(texts)})
        mlflow.log_metrics(metrics)
        mlflow.log_text(classification_report(y_true, y_pred), "classification_report.txt")
        mlflow.transformers.log_model(
            transformers_model=classifier,
            artifact_path="model",
            registered_model_name=os.getenv("MODEL_NAME", "finbert"),
            input_example=texts[:1],
        )


if __name__ == "__main__":
    main()
