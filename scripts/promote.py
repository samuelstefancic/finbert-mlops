"""Promote the latest evaluated model when its weighted F1 meets the threshold."""

import os

import mlflow
import yaml
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient

load_dotenv()


def load_params() -> dict:
    with open("params.yaml") as f:
        return yaml.safe_load(f)["promote"]


def get_latest_f1(client: MlflowClient, experiment_name: str) -> tuple[str, float]:
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise RuntimeError(f"Experiment not found: {experiment_name}")
    runs = client.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED' AND metrics.f1_weighted >= 0",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    if not runs:
        raise RuntimeError("No completed evaluation run with f1_weighted")
    return runs[0].info.run_id, runs[0].data.metrics["f1_weighted"]


def promote(client: MlflowClient, model_name: str):
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No registered version for {model_name}")
    latest = max(versions, key=lambda version: int(version.version))
    client.set_registered_model_alias(model_name, "production", latest.version)
    print(f"Promoted {model_name} version {latest.version} to production")


def main():
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    client = MlflowClient()
    threshold = float(load_params()["f1_threshold"])
    model_name = os.getenv("MODEL_NAME", "finbert")
    run_id, f1 = get_latest_f1(
        client, os.getenv("MLFLOW_EXPERIMENT_NAME", "finbert-evaluation")
    )
    if f1 < threshold:
        print(f"Not promoted: weighted F1 {f1:.4f} < threshold {threshold:.4f}")
        return
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        raise RuntimeError(f"No registered version for {model_name}")
    latest = max(versions, key=lambda version: int(version.version))
    if latest.run_id != run_id:
        raise RuntimeError("Latest registered version does not match the evaluated run")
    promote(client, model_name)


if __name__ == "__main__":
    main()
