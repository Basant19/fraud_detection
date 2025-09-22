# D:\fraud_detection\src\pipeline\predict_pipeline.py

import os
import sys
import pandas as pd
from src.entity.config import ModelPredictionConfig, DataTransformationConfig
from src.components.model_prediction import ModelPrediction
from src.components.data_transformation import DataTransformation
from src.logger import logging
from src.exception import CustomException


def start_prediction_pipeline(input_data: pd.DataFrame):
    """
    Predict pipeline:
    - Applies feature engineering & preprocessing exactly as in training
    - Does NOT apply SMOTE/Tomek (no resampling in prediction)
    - Generates predictions & probabilities using trained model
    """

    try:
        # ----------------------
        # 1. Load Prediction Config
        # ----------------------
        prediction_config = ModelPredictionConfig.get_default_config()
        logging.info("✅ Model Prediction Config Loaded")

        # ----------------------
        # 2. Apply Feature Engineering & Preprocessing
        # ----------------------
        # Use DataTransformation but skip resampling
        transformation_config = DataTransformationConfig.get_default_config()
        transformer = DataTransformation(transformation_config, resampling_strategy="none")

        # We need a mock ingestion artifact for DataTransformation
        from src.entity.artifacts import DataIngestionArtifacts
        ingestion_artifact = DataIngestionArtifacts(
            train_file_path="", test_file_path="", raw_file_path="", ingestion_metadata_path=""
        )

        # Apply feature engineering only
        try:
            # Access the private feature_engineering function from DataTransformation
            # We'll simulate the same transformations
            drop_cols = ["TransactionID", "MerchantID", "CustomerID", "Name", "Address", "LastLogin"]

            def feature_engineering(df):
                df = df.copy()
                if "Timestamp" in df.columns:
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                    df["Hour"] = df["Timestamp"].dt.hour.fillna(-1).astype(int)
                if "LastLogin" in df.columns:
                    df["LastLogin"] = pd.to_datetime(df["LastLogin"], errors="coerce")
                if "Timestamp" in df.columns and "LastLogin" in df.columns:
                    df["gap"] = (df["Timestamp"] - df["LastLogin"]).dt.days.abs()
                    df["gap"] = df["gap"].fillna(df["gap"].median())
                else:
                    df["gap"] = 0
                df = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["Timestamp", "LastLogin"], errors="ignore")
                return df

            df_input_fe = feature_engineering(input_data)
            logging.info(f"✅ Feature engineering applied: resulting shape {df_input_fe.shape}")

        except Exception as e:
            raise CustomException(f"Feature engineering failed: {e}", sys) from e

        # ----------------------
        # 3. Initialize ModelPrediction component
        # ----------------------
        predictor = ModelPrediction(prediction_config)
        logging.info("✅ ModelPrediction component initialized")

        # ----------------------
        # 4. Predict on transformed input
        # ----------------------
        artifacts = predictor.predict(df_input_fe)
        logging.info("✅ Prediction completed")

        # ----------------------
        # 5. Print summary
        # ----------------------
        print("Prediction Report Path:", artifacts.prediction_report_path)
        print("Predictions:", artifacts.predictions)
        print("Probabilities:", artifacts.probabilities)

        return artifacts

    except Exception as e:
        raise CustomException(f"Prediction pipeline failed: {e}", sys) from e


if __name__ == "__main__":
    # Example usage with CSV input
    try:
        sample_file = os.path.join("artifacts", "sample_input.csv")
        if not os.path.exists(sample_file):
            raise FileNotFoundError(f"{sample_file} does not exist!")

        df_input = pd.read_csv(sample_file)
        start_prediction_pipeline(df_input)

    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}", exc_info=True)
        sys.exit(1)
