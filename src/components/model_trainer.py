# src/components/model_trainer.py
import os
import sys
import pickle
import socket
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score, fbeta_score
from dotenv import load_dotenv

from src.entity.config import ModelTrainerConfig
from src.entity.artifacts import DataTransformationArtifacts, ModelTrainerArtifacts
from src.exception import CustomException
from src.logger import logging

# ML models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


def is_mlflow_server_running(host="127.0.0.1", port=5000) -> bool:
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        load_dotenv()

    def initiate_model_training(self, transformation_artifact: DataTransformationArtifacts) -> ModelTrainerArtifacts:
        try:
            logging.info("Starting model training...")

            # ---------------- Load Transformed Data ----------------
            train_data = np.load(transformation_artifact.transformed_train_path, allow_pickle=True)
            test_data = np.load(transformation_artifact.transformed_test_path, allow_pickle=True)

            X_train, y_train = train_data["X_train"], train_data["y_train"]
            X_test, y_test = test_data["X_test"], test_data["y_test"]
            logging.info(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

            # ---------------- MLflow Setup ----------------
            if is_mlflow_server_running():
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
                logging.info(f"Connected to MLflow server → {tracking_uri}")
            else:
                tracking_uri = os.getenv("MLFLOW_LOCAL_URI", f"file:///{os.getcwd()}/mlruns")
                logging.info(f"No MLflow server → using local tracking dir: {tracking_uri}")

            mlflow.set_tracking_uri(tracking_uri)
            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME_3", "fraud_detection_experiment_3")
            mlflow.set_experiment(experiment_name)

            # ---------------- Candidate Models ----------------
            models = {
                "RandomForest": RandomForestClassifier(random_state=self.config.random_state, n_jobs=-1),
                "GradientBoosting": GradientBoostingClassifier(random_state=self.config.random_state),
                "DecisionTree": DecisionTreeClassifier(random_state=self.config.random_state),
                "LogisticRegression": LogisticRegression(random_state=self.config.random_state, max_iter=1000),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(random_state=self.config.random_state, probability=True),
                "XGBoost": XGBClassifier(random_state=self.config.random_state, use_label_encoder=False, eval_metric="logloss"),
            }

            best_model = None
            best_score = -1
            best_name = None

            # ---------------- Train & Evaluate ----------------
            for name, model in models.items():
                logging.info(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                f1 = f1_score(y_test, y_pred, zero_division=0)
                f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)
                logging.info(f"{name} → F1: {f1:.4f}, F2: {f2:.4f}")

                with mlflow.start_run(run_name=name):
                    mlflow.log_param("model_name", name)
                    mlflow.log_param("resampling_strategy", "smote_tomek")
                    mlflow.log_metric("f1_score", f1)
                    mlflow.log_metric("f2_score", f2)
                    mlflow.sklearn.log_model(model, artifact_path=f"{name}_model")

                # Select best model based on F2
                if f2 > best_score:
                    best_score = f2
                    best_model = model
                    best_name = name

            # ---------------- Save Best Model ----------------
            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
            with open(self.config.trained_model_path, "wb") as f:
                pickle.dump(best_model, f)

            logging.info(f"Best Model: {best_name} with F2-score={best_score:.4f}")

            return ModelTrainerArtifacts(
                trained_model_path=self.config.trained_model_path,
                training_score=best_model.score(X_train, y_train),
                test_score=best_score,
            )

        except Exception as e:
            logging.error("Error in model training", exc_info=True)
            raise CustomException(e, sys)
