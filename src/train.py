#!/usr/bin/env python3
import argparse
import json
import logging
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
    parser.add_argument("--data", type=str, required=True, help="Path to processed CSV dataset (DVC tracked path)")
    parser.add_argument("--models-dir", type=str, required=True, help="Directory to save trained model")
    parser.add_argument("--dvc", action="store_true", help="If set, use DVC to pull data and track model artifact")
    parser.add_argument("--dvc-remote", type=str, default=None, help="Optional DVC remote name to push to")
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
# DVC helpers
# -----------------------------
def dvc_pull(path: str | None = None):
    cmd = ["dvc", "pull"]
    if path:
        cmd.append(path)
    logger.info("Running: %s", " ".join(cmd))
    subprocess.run(cmd, check=True)

def dvc_add_and_push(path: str, remote: str | None = None):
    try:
        logger.info("Running: dvc add %s", path)
        subprocess.run(["dvc", "add", path], check=True)
    except subprocess.CalledProcessError:
        logger.warning("DVC add failed (maybe file already tracked): %s", path)

    # Git add the .dvc file if it exists
    dvc_meta = f"{path}.dvc"
    if Path(dvc_meta).exists():
        logger.info("Running: git add %s", dvc_meta)
        subprocess.run(["git", "add", dvc_meta], check=True)

        try:
            subprocess.run(["git", "commit", "-m", f"Add model artifact {Path(path).name}"], check=True)
        except subprocess.CalledProcessError:
            logger.info("Nothing new to commit for %s", dvc_meta)

    # DVC push (optional remote)
    push_cmd = ["dvc", "push"]
    if remote:
        push_cmd += ["-r", remote]
    try:
        logger.info("Running: %s", " ".join(push_cmd))
        subprocess.run(push_cmd, check=True)
    except subprocess.CalledProcessError as e:
        logger.warning("DVC push failed (maybe nothing to push): %s", e)

# -----------------------------
# Main logic
# -----------------------------
def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_cfg = config["model"]

    # Ensure models directory exists
    models_dir = Path(args.models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)

    # Set MLflow tracking to models/mlruns
    mlruns_path = models_dir / "mlruns"
    mlflow.set_tracking_uri(f"file://{mlruns_path.resolve()}")
    mlflow.set_experiment(model_cfg.get("name", "default-experiment"))
    logger.info("MLflow tracking URI set to %s", mlruns_path)

    # Pull dataset via DVC if enabled
    if args.dvc:
        logger.info("DVC mode enabled, pulling dataset")
        dvc_pull(args.data)

    # Load dataset
    df = pd.read_csv(args.data)
    target = model_cfg["target_variable"]
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=model_cfg.get("test_size", 0.2), random_state=42
    )

    # Instantiate model
    model = get_model_instance(model_cfg["best_model"], model_cfg.get("parameters", {}))

    artifact_name = model_cfg.get("artifact_name", "trained_model.pkl")
    save_path = models_dir / artifact_name

    # -----------------------------
    # MLflow run
    # -----------------------------
    with mlflow.start_run(run_name="final_training") as run:
        run_id = run.info.run_id
        logger.info("Training model: %s | run_id=%s", model_cfg["best_model"], run_id)

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Compute metrics
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))
        logger.info("Metrics -- MAE: %.4f | R2: %.4f", mae, r2)

        # Log params, metrics, and model
        mlflow.log_params(model_cfg.get("parameters", {}))
        mlflow.log_metrics({"mae": mae, "r2": r2})
        mlflow.sklearn.log_model(model, "tuned_model")

        # Save model locally
        joblib.dump(model, save_path)
        logger.info("Saved model to %s", save_path)

        # Save metrics.json for DVC
        metrics_path = models_dir / "metrics.json"
        with open(metrics_path, "w") as mf:
            json.dump({"mae": mae, "r2": r2}, mf, indent=2)
        logger.info("Saved metrics to %s", metrics_path)

        # Register model
        model_name = model_cfg.get("name", "my_model")
        client = MlflowClient()
        model_uri = f"runs:/{run_id}/tuned_model"
        try:
            client.create_registered_model(model_name)
            logger.info("Created registered model: %s", model_name)
        except MlflowException:
            logger.info("Registered model %s already exists.", model_name)

        try:
            mv = client.create_model_version(name=model_name, source=model_uri, run_id=run_id)
            logger.info("Created model version %s", mv.version)
        except MlflowException as e:
            logger.error("Failed to create model version: %s", e)
            raise

        # Transition to staging
        try:
            client.transition_model_version_stage(name=model_name, version=mv.version, stage="Staging")
        except MlflowException as e:
            logger.warning("Could not transition stage: %s", e)

        # Add metadata/tags
        try:
            client.update_registered_model(name=model_name, description=f"Trained model for target: {target}")
            tags = {
                "algorithm": model_cfg["best_model"],
                "artifact_path": str(save_path),
                "training_run_id": run_id,
                "python_version": platform.python_version(),
                "scikit_learn_version": sklearn.__version__,
                "xgboost_version": xgb.__version__,
                "pandas_version": pd.__version__,
                "numpy_version": np.__version__,
            }
            for k, v in tags.items():
                client.set_registered_model_tag(model_name, k, v)
        except MlflowException as e:
            logger.warning("Could not update model tags: %s", e)

    # Push model via DVC if enabled
    if args.dvc:
        dvc_add_and_push(str(save_path), remote=args.dvc_remote)

    logger.info("Training pipeline finished. Model saved at: %s", save_path)

# -----------------------------
# Entry point
# -----------------------------
if __name__ == "__main__":
    args = parse_args()
    main(args)
