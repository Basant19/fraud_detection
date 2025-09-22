#D:\fraud_detection\src\components\model_prediction.py

import sys

import os
import json
import pickle
import numpy as np
import pandas as pd
from typing import Union
from datetime import datetime

from src.entity.config import ModelPredictionConfig
from src.entity.artifacts import ModelPredictionArtifacts
from src.logger import logging
from src.exception import CustomException


class ModelPrediction:
    """
    Handles model inference (fraud detection) for single or batch transactions.
    Ensures preprocessing and feature engineering consistency.
    """

    def __init__(self, config: ModelPredictionConfig = None):
        self.config = config or ModelPredictionConfig.get_default_config()

        # Use dedicated subfolder for predictions
        self.prediction_dir = os.path.join("artifacts", "model_prediction")
        os.makedirs(self.prediction_dir, exist_ok=True)

        # Load model, preprocessor, feature names
        self.model = self._load_pickle(self.config.trained_model_path, "model")
        self.preprocessor = self._load_pickle(self.config.preprocessor_path, "preprocessor")
        self.feature_names = self._load_feature_names(self.config.feature_names_path)

    def _load_pickle(self, path: str, name: str):
        try:
            with open(path, "rb") as f:
                obj = pickle.load(f)
            logging.info(f"✅ Loaded {name} from {path}")
            return obj
        except Exception as e:
            raise CustomException(e, sys) from e

    def _load_feature_names(self, path: str):
        try:
            with open(path, "r") as f:
                names = json.load(f)
            logging.info(f"✅ Loaded feature names from {path}")
            return names
        except Exception as e:
            raise CustomException(e, sys) from e

    def predict(self, input_data: Union[pd.DataFrame, dict, list]) -> ModelPredictionArtifacts:
        """
        Run fraud detection prediction.
        Args:
            input_data: transaction(s) as dict, list[dict], or pandas DataFrame.
        Returns:
            ModelPredictionArtifacts with predictions, probabilities, and report path.
        """

        # Convert input into DataFrame
        if isinstance(input_data, dict):
            df = pd.DataFrame([input_data])
        elif isinstance(input_data, list):
            df = pd.DataFrame(input_data)
        elif isinstance(input_data, pd.DataFrame):
            df = input_data.copy()
        else:
            raise CustomException("Unsupported input type for prediction")

        logging.info(f"Received input data with shape: {df.shape}")

        # Preprocess data
        try:
            transformed = self.preprocessor.transform(df)
            logging.info(f"✅ Applied preprocessing to input data")
        except Exception as e:
            raise CustomException(f"Preprocessing failed: {e}", sys) from e

        # Generate predictions & probabilities
        try:
            predictions = self.model.predict(transformed)
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(transformed)[:, 1]
            else:
                probabilities = np.zeros(len(predictions))
            logging.info("✅ Generated predictions and probabilities")
        except Exception as e:
            raise CustomException(f"Prediction failed: {e}", sys) from e

        # Save results to prediction report
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.prediction_dir, f"prediction_report_{timestamp}.json")

        report = {
            "timestamp": timestamp,
            "n_samples": len(predictions),
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
        }

        try:
            with open(report_path, "w") as f:
                json.dump(report, f, indent=4)
            logging.info(f"✅ Prediction report saved to {report_path}")
        except Exception as e:
            raise CustomException(f"Failed to save prediction report: {e}", sys) from e

        # Return artifacts
        return ModelPredictionArtifacts(
            predictions=predictions,
            probabilities=probabilities,
            prediction_report_path=report_path
        )
