# D:\fraud_detection\test\test_model_prediction.py
import os
import json
import pandas as pd

from src.entity.config import ModelPredictionConfig
from src.components.model_prediction import ModelPrediction


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature engineering as training.
    Creates 'Hour' and 'gap' from Timestamp and LastLogin.
    """
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

    return df


if __name__ == "__main__":
    # ---------------- Step 1: Load default config ----------------
    config = ModelPredictionConfig.get_default_config()

    # ---------------- Step 2: Initialize ModelPrediction ----------------
    predictor = ModelPrediction(config)

    # ---------------- Step 3: Load or create input data ----------------
    test_csv_path = r"D:\fraud_detection\artifacts\test.csv"
    if os.path.exists(test_csv_path):
        df = pd.read_csv(test_csv_path)
        input_data = df.sample(5, random_state=42)
    else:
        input_data = {
            "TransactionID": 1,
            "FraudIndicator": 0,
            "Category": "purchase",
            "TransactionAmount": 120.50,
            "AnomalyScore": 0.1,
            "Timestamp": "2025-09-22 12:00:00",
            "LastLogin": "2025-09-21 11:00:00",
            "account_age_days": 365,
            "device_type": "mobile",
            "location": "New York",
            "Age": 30,
            "Address": "123 Main St",
            "AccountBalance": 5000.0,
            "SuspiciousFlag": 0,
        }
        input_data = pd.DataFrame([input_data])

    # ---------------- Step 4: Apply feature engineering ----------------
    input_data = apply_feature_engineering(input_data)

    # DO NOT encode 'Category' manually! ColumnTransformer will handle it

    print("\n🔍 Input Data for Prediction:")
    print(input_data.head())

    # ---------------- Step 5: Run Prediction ----------------
    artifacts = predictor.predict(input_data)

    print("\n✅ Prediction successful")
    print("Predictions:", artifacts.predictions[:10])
    print("Probabilities:", artifacts.probabilities[:10])
    print("Report Path:", artifacts.prediction_report_path)

    # ---------------- Step 6: Verify saved report ----------------
    with open(artifacts.prediction_report_path, "r") as f:
        report = json.load(f)

    print("\n📊 Prediction Report Content:")
    print(json.dumps(report, indent=4))
