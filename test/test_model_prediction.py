import os
import json
import pickle
import pandas as pd

from src.entity.config import ModelPredictionConfig
from src.components.model_prediction import ModelPrediction


if __name__ == "__main__":
    # ---------------- Step 1: Load default config ----------------
    config = ModelPredictionConfig.get_default_config()

    # ---------------- Step 2: Initialize ModelPrediction ----------------
    predictor = ModelPrediction(config)

    # ---------------- Step 3: Create mock input data ----------------
    # Either load test.csv OR build a small sample
    test_csv_path = r"D:\fraud_detection\artifacts\test.csv"
    if os.path.exists(test_csv_path):
        df = pd.read_csv(test_csv_path)
        input_data = df.sample(5, random_state=42)  # pick 5 rows
    else:
        # Minimal mock example (must match feature names)
        input_data = {
            "amount": 120.50,
            "transaction_type": "purchase",
            "account_age_days": 365,
            "device_type": "mobile",
            "location": "New York",
        }

    print("\n🔍 Input Data for Prediction:")
    print(input_data)

    # ---------------- Step 4: Run Prediction ----------------
    artifacts = predictor.predict(input_data)

    print("\n✅ Prediction successful")
    print("Predictions:", artifacts.predictions[:10])
    print("Probabilities:", artifacts.probabilities[:10])
    print("Report Path:", artifacts.prediction_report_path)

    # ---------------- Step 5: Verify saved report ----------------
    with open(artifacts.prediction_report_path, "r") as f:
        report = json.load(f)

    print("\n📊 Prediction Report Content:")
    print(json.dumps(report, indent=4))
