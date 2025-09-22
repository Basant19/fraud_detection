# D:\fraud_detection\src\components\model_evaluation.py

import os
import sys
import json
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    precision_score, recall_score, f1_score, fbeta_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, RocCurveDisplay, PrecisionRecallDisplay
)

from src.entity.artifacts import (
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
    ModelEvaluationArtifacts,
)
from src.entity.config import ModelEvaluationConfig
from src.exception import CustomException
from src.logger import logging


class ModelEvaluator:
    def __init__(self, config: ModelEvaluationConfig, best_model_path: str):
        """
        Args:
            config: ModelEvaluationConfig containing paths and acceptance criteria
            best_model_path: Path of trained best model to load
        """
        self.config = config
        self.evaluation_report_path = config.evaluation_report_path
        self.min_recall = config.min_recall
        self.min_f2 = config.min_f2
        self.best_model_path = best_model_path

    def evaluate_model(self,
                       transformation_artifact: DataTransformationArtifacts,
                       trainer_artifact: ModelTrainerArtifacts
                       ) -> ModelEvaluationArtifacts:
        try:
            logging.info("🔍 Starting model evaluation...")

            # ---------------- Load Artifacts ----------------
            test_data = np.load(transformation_artifact.transformed_test_path, allow_pickle=True)
            X_test, y_test = test_data["X_test"], test_data["y_test"]

            with open(trainer_artifact.trained_model_path, "rb") as f:
                model = pickle.load(f)

            logging.info(f"Loaded best model from {trainer_artifact.trained_model_path}")

            # ---------------- Predictions ----------------
            y_pred = model.predict(X_test)
            y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # ---------------- Metrics ----------------
            precision = precision_score(y_test, y_pred, zero_division=0)
            recall = recall_score(y_test, y_pred, zero_division=0)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            f2 = fbeta_score(y_test, y_pred, beta=2, zero_division=0)

            roc_auc = roc_auc_score(y_test, y_prob) if y_prob is not None else None
            pr_auc = average_precision_score(y_test, y_prob) if y_prob is not None else None

            cm = confusion_matrix(y_test, y_pred).tolist()  # Convert to list for JSON

            metrics = {
                "precision": precision,
                "recall": recall,
                "f1_score": f1,
                "f2_score": f2,
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "confusion_matrix": cm,
            }

            # ---------------- Save Report ----------------
            os.makedirs(os.path.dirname(self.evaluation_report_path), exist_ok=True)
            with open(self.evaluation_report_path, "w") as f:
                json.dump(metrics, f, indent=4)

            logging.info(f"✅ Evaluation report saved → {self.evaluation_report_path}")

            # ---------------- Visualizations ----------------
            if y_prob is not None:
                roc_disp = RocCurveDisplay.from_predictions(y_test, y_prob)
                roc_disp.figure_.savefig(
                    os.path.join(os.path.dirname(self.evaluation_report_path), "roc_curve.png")
                )
                plt.close(roc_disp.figure_)

                pr_disp = PrecisionRecallDisplay.from_predictions(y_test, y_prob)
                pr_disp.figure_.savefig(
                    os.path.join(os.path.dirname(self.evaluation_report_path), "pr_curve.png")
                )
                plt.close(pr_disp.figure_)

                logging.info("ROC & PR curves saved.")

            # ---------------- Acceptance Criteria ----------------
            acceptance_criteria = {
                "min_recall": self.min_recall,
                "min_f2": self.min_f2
            }
            is_accepted = (recall >= self.min_recall) and (f2 >= self.min_f2)

            # ---------------- Artifact ----------------
            return ModelEvaluationArtifacts(
                evaluation_report_path=self.evaluation_report_path,
                best_model_path=trainer_artifact.trained_model_path,
                is_model_accepted=is_accepted,
                acceptance_criteria=acceptance_criteria
            )

        except Exception as e:
            logging.error("❌ Error during model evaluation", exc_info=True)
            raise CustomException(e, sys)
