"""Fail the quality gate when text sentiment or prediction distributions drift."""

import json
import math
import os
from pathlib import Path
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import pandas as pd
import yaml
from deepchecks.nlp import TextData
from deepchecks.nlp.checks import PredictionDrift, PropertyDrift
from dotenv import load_dotenv

from app.utils import load_classifier

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["deepchecks"]


def run_predictions(classifier, texts: list[str]) -> list[str]:
    results = classifier(texts, batch_size=32, truncation=True, max_length=512)
    return [result["label"] for result in results]


def main():
    params = load_params()
    property_drift_threshold = float(params["property_drift_threshold"])
    prediction_drift_threshold = float(params["prediction_drift_threshold"])

    stream_texts = pd.read_csv("data/stream.csv")["text"].tolist()
    test_texts = pd.read_csv("data/test.csv")["text"].tolist()
    if not stream_texts or not test_texts:
        raise ValueError("Both datasets must contain text")

    stream_dataset = TextData(
        raw_text=stream_texts, task_type="text_classification", name="stream"
    )
    test_dataset = TextData(
        raw_text=test_texts, task_type="text_classification", name="reference"
    )
    for dataset in (stream_dataset, test_dataset):
        dataset.calculate_builtin_properties(
            include_properties=["Sentiment"],
            ignore_non_english_samples_for_english_properties=False,
        )

    property_result = PropertyDrift(properties=["Sentiment"]).run(
        train_dataset=test_dataset, test_dataset=stream_dataset, with_display=False
    )
    property_score = float(property_result.value["Sentiment"]["Drift score"])

    classifier = load_classifier()
    test_predictions = run_predictions(classifier, test_texts)
    stream_predictions = run_predictions(classifier, stream_texts)
    prediction_result = PredictionDrift(drift_mode="prediction").run(
        train_dataset=test_dataset,
        test_dataset=stream_dataset,
        train_predictions=test_predictions,
        test_predictions=stream_predictions,
        with_display=False,
    )
    prediction_score = float(prediction_result.value["Drift score"])
    scores = {"property_drift": property_score, "prediction_drift": prediction_score}
    if not all(math.isfinite(value) for value in scores.values()):
        raise RuntimeError("Drift checks did not produce finite scores")
    Path("reports").mkdir(exist_ok=True)
    Path("reports/drift.json").write_text(json.dumps(scores, indent=2))
    print(json.dumps(scores, indent=2))
    failures = []
    if property_score > property_drift_threshold:
        failures.append(f"Sentiment drift {property_score:.4f} > {property_drift_threshold}")
    if prediction_score > prediction_drift_threshold:
        failures.append(f"Prediction drift {prediction_score:.4f} > {prediction_drift_threshold}")
    if failures:
        raise SystemExit("Quality gate failed: " + "; ".join(failures))
    print("Quality gate passed")


if __name__ == "__main__":
    main()
