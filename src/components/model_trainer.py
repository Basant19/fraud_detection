import os
import sys
import pickle
import socket
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.metrics import f1_score
from dotenv import load_dotenv  

from src.entity.config import ModelTrainerConfig
from src.entity.artifacts import DataTransformationArtifacts, ModelTrainerArtifacts
from src.exception import CustomException
from src.logger import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def is_mlflow_server_running(host="127.0.0.1", port=5000):
    """Check if MLflow tracking server is running."""
    try:
        with socket.create_connection((host, port), timeout=2):
            return True
    except OSError:
        return False


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        load_dotenv()  # load .env file

    def initiate_model_training(
        self, transformation_artifact: DataTransformationArtifacts
    ) -> ModelTrainerArtifacts:
        try:
            logging.info("Starting model training...")

            # Load transformed data
            train_data = np.load(transformation_artifact.transformed_train_path, allow_pickle=True)
            test_data = np.load(transformation_artifact.transformed_test_path, allow_pickle=True)

            X_train, y_train = train_data["X_train"], train_data["y_train"]
            X_test, y_test = test_data["X_test"], test_data["y_test"]

            # Configure MLflow tracking URI
            if is_mlflow_server_running():
                tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
                logging.info(f"Connected to MLflow server → {tracking_uri}")
            else:
                tracking_uri = os.getenv("MLFLOW_LOCAL_URI", "file:///D:/fraud_detection/mlruns")
                logging.info(f"No MLflow server → using local tracking dir: {tracking_uri}")

            mlflow.set_tracking_uri(tracking_uri)

            experiment_name = os.getenv("MLFLOW_EXPERIMENT_NAME", "fraud_detection_experiment")
            mlflow.set_experiment(experiment_name)

            # Define candidate models
            models = {
                "RandomForest": RandomForestClassifier(random_state=self.config.random_state, n_jobs=-1),
                "GradientBoosting": GradientBoostingClassifier(random_state=self.config.random_state),
                "DecisionTree": DecisionTreeClassifier(random_state=self.config.random_state),
                "KNN": KNeighborsClassifier(),
                "SVM": SVC(random_state=self.config.random_state, probability=True),
            }

            best_model = None
            best_score = -1
            best_name = None

            # Train and evaluate
            for name, model in models.items():
                logging.info(f"Training {name}...")
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                f1 = f1_score(y_test, y_pred, zero_division=0)
                logging.info(f"{name} f1_score: {f1:.4f}")

                # MLflow logging
                with mlflow.start_run(run_name=name):
                    mlflow.log_param("model", name)
                    mlflow.log_metric("f1_score", f1)
                    mlflow.sklearn.log_model(model, artifact_path=f"{name}_model")

                if f1 > best_score:
                    best_score = f1
                    best_model = model
                    best_name = name

            # Save best model locally
            os.makedirs(os.path.dirname(self.config.trained_model_path), exist_ok=True)
            with open(self.config.trained_model_path, "wb") as f:
                pickle.dump(best_model, f)

            logging.info(f"Best Model: {best_name} with f1_score={best_score:.4f}")

            return ModelTrainerArtifacts(
                trained_model_path=self.config.trained_model_path,
                training_score=best_model.score(X_train, y_train),
                test_score=best_score,
            )

        except Exception as e:
            logging.error("Error in model training", exc_info=True)
            raise CustomException(e, sys)
