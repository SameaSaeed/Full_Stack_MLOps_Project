#!/usr/bin/env python3
import argparse
import json
import logging
import os
import platform
import subprocess
from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import sklearn
import xgboost as xgb
import yaml
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

# -----------------------------
# Configure logging
# -----------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train and register final model with MLflow + optional DVC")
    parser.add_argument("--config", type=str, required=True, help="Path to model_config.yaml")
    parser.add_argument(
        "--data", type=str, required=True, help="Path to processed CSV dataset (DVC tracked path)"
    )
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--mlflow-tracking-uri", type=str, default=None, help="MLflow tracking URI")
    parser.add_argument("--dvc", action="store_true", help="If set, use DVC to pull data and track model artifact")
    parser.add_argument("--dvc-remote", type=str, default=None, help="Optional DVC remote name to push to (e.g. 'storage')")
    return parser.parse_args()


# -----------------------------
# Model factory
# -----------------------------
def get_model_instance(name, params):
    model_map = {
        "LinearRegression": LinearRegression,
        "RandomForest": RandomForestRegressor,
        "GradientBoosting": GradientBoostingRegressor,
        "XGBoost": xgb.XGBRegressor,
    }
    if name not in model_map:
        raise ValueError(f"Unsupported model: {name}")
    return model_map[name](**params)


# -----------------------------
# Helpers for DVC (optional)
# -----------------------------
def dvc_pull(path: str):
    logger.info("Running: dvc pull %s", path)
    cmd = ["dvc", "pull", path] if path else ["dvc", "pull"]
    subprocess.run(cmd, check=True)


def dvc_add_and_push(path: str, remote: str | None = None):
    # dvc add <path>
    logger.info("Running: dvc add %s", path)
    subprocess.run(["dvc", "add", path], check=True)

    # git add the .dvc file
    dvc_meta = f"{path}.dvc"
    logger.info("Running: git add %s", dvc_meta)
    subprocess.run(["git", "add", dvc_meta], check=True)

    # commit - allow skipping if nothing to commit
    logger.info("Running: git commit -m 'Add model artifact via DVC' || true")
    try:
        subprocess.run(["git", "commit", "-m", f"Add model artifact {Path(path).name}"], check=True)
    except subprocess.CalledProcessError:
        # It's fine if there's nothing to commit (e.g., same file/version)
        logger.info("Git commit returned non-zero (maybe nothing to commit). Continuing.")

    # dvc push (optionally to a named remote)
    push_cmd = ["dvc", "push"]
    if remote:
        push_cmd += ["-r", remote]
    logger.info("Running: %s", " ".join(push_cmd))
    subprocess.run(push_cmd, check=True)


# -----------------------------
# Main logic
# -----------------------------
def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]

    # Configure MLflow
    if args.mlflow_tracking_uri:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
    # set experiment regardless of URI; mlflow will create a local experiment if needed
    mlflow.set_experiment(model_cfg.get("name", "default-experiment"))

    # If DVC mode, pull dataset
    if args.dvc:
        logger.info("DVC mode enabled - pulling data from remote if needed")
        dvc_pull(args.data)

    # Load data
    logger.info("Loading dataset from: %s", args.data)
    df = pd.read_csv(args.data)
    target = model_cfg["target_variable"]

    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=model_cfg.get("test_size", 0.2), random_state=42)

    # Create model
    model = get_model_instance(model_cfg["best_model"], model_cfg.get("parameters", {}))

    # Deterministic save path so DVC knows the artifact location
    models_dir = Path(args.models_dir)
    artifact_name = model_cfg.get("artifact_name", "trained_model.pkl")
    models_dir.mkdir(parents=True, exist_ok=True)
    save_path = models_dir / artifact_name

    with mlflow.start_run(run_name="final_training") as run:
        run_id = run.info.run_id
        logger.info("Training model: %s | run_id=%s", model_cfg["best_model"], run_id)

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        logger.info("Metrics -- MAE: %.4f | R2: %.4f", mae, r2)

        # Log params & metrics to MLflow
        mlflow.log_params(model_cfg.get("parameters", {}))
        mlflow.log_metrics({"mae": mae, "r2": r2})

        # Log model to MLflow (artifact store)
        mlflow.sklearn.log_model(model, "tuned_model")

        # Save model locally (deterministic path)
        joblib.dump(model, save_path)
        logger.info("Saved model to %s", save_path)

        # Save metrics.json for DVC metrics tracking
        metrics = {"mae": mae, "r2": r2}
        metrics_path = models_dir / "metrics.json"
        with open(metrics_path, "w") as mf:
            json.dump(metrics, mf, indent=2)
        logger.info("Saved metrics to %s", metrics_path)

        # Register model in MLflow Model Registry
        model_name = model_cfg.get("name", "my_model")
        model_uri = f"runs:/{run_id}/tuned_model"
        client = MlflowClient()
        try:
            client.create_registered_model(model_name)
            logger.info("Created new registered model: %s", model_name)
        except MlflowException:
            logger.info("Registered model %s already exists (or create failed).", model_name)

        try:
            mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
            logger.info("Created model version %s for model %s", mv.version, model_name)
        except MlflowException as e:
            logger.error("Failed to create model version: %s", e)
            raise

        # Transition to Staging (best-effort)
        try:
            client.transition_model_version_stage(name=model_name, version=mv.version, stage="Staging")
            logger.info("Transitioned model %s version %s to 'Staging'", model_name, mv.version)
        except MlflowException as e:
            logger.warning("Could not transition model version stage: %s", e)

        # Add metadata / tags to registered model
        try:
            client.update_registered_model(name=model_name, description=f"Trained model for target: {target}")
            # Set tags on the registered model (not model version)
            client.set_registered_model_tag(model_name, "algorithm", model_cfg["best_model"])
            client.set_registered_model_tag(model_name, "artifact_path", str(save_path))
            client.set_registered_model_tag(model_name, "training_run_id", run_id)
            client.set_registered_model_tag(model_name, "python_version", platform.python_version())
            client.set_registered_model_tag(model_name, "scikit_learn_version", sklearn.__version__)
            client.set_registered_model_tag(model_name, "xgboost_version", xgb.__version__)
            client.set_registered_model_tag(model_name, "pandas_version", pd.__version__)
            client.set_registered_model_tag(model_name, "numpy_version", np.__version__)
        except MlflowException as e:
            logger.warning("Unable to update registered model tags/description: %s", e)

    # After run: optionally track model via DVC (add + push)
    if args.dvc:
        try:
            dvc_add_and_push(str(save_path), remote=args.dvc_remote)
        except subprocess.CalledProcessError as e:
            logger.error("Error while running DVC commands: %s", e)
            raise

    logger.info("Training pipeline finished successfully. Model at: %s", save_path)

metrics = {"mae": mae, "r2": r2}
with open("metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)

if __name__ == "__main__":
    args = parse_args()
    main(args)
