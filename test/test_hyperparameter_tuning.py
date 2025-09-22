# D:\fraud_detection\test\test_hyperparameter_tuning.py

import os
import pickle
import json
import numpy as np

from src.entity.config import DataTransformationConfig, HyperparameterTuningConfig
from src.entity.artifacts import DataIngestionArtifacts
from src.components.data_transformation import DataTransformation
from src.components.hyperparameter_tuning import HyperparameterTuner


if __name__ == "__main__":
    # Step 1: Mock ingestion artifacts (assumes ingestion already ran)
    ingestion_artifact = DataIngestionArtifacts(
        train_file_path=r"D:\fraud_detection\artifacts\train.csv",
        test_file_path=r"D:\fraud_detection\artifacts\test.csv",
        raw_file_path=r"D:\fraud_detection\artifacts\raw_data\raw.csv",
        ingestion_metadata_path=r"D:\fraud_detection\artifacts\raw_data\ingestion_metadata.json"
    )

    # Step 2: Run transformation
    transformation_config = DataTransformationConfig.get_default_config()
    transformation = DataTransformation(transformation_config)
    transformation_artifact = transformation.initiate_data_transformation(ingestion_artifact)

    print("✅ Transformation successful")
    print("Transformation Artifacts:", transformation_artifact)

    # Step 3: Run hyperparameter tuning
    tuning_config = HyperparameterTuningConfig.get_default_config()
    tuner = HyperparameterTuner(tuning_config)
    tuning_artifact = tuner.initiate_hyperparameter_tuning(transformation_artifact)

    # Step 4: Print results
    print("\n✅ Hyperparameter tuning successful")
    print("Tuned model path:", tuning_artifact.tuned_model_path)
    print("Best params path:", tuning_artifact.best_params_path)
    print("Best CV Score (F2):", tuning_artifact.best_score)

    # Step 5: Load tuned model
    with open(tuning_artifact.tuned_model_path, "rb") as f:
        tuned_model = pickle.load(f)

    print("\nLoaded tuned model type:", type(tuned_model))

    # Step 6: Load best params JSON
    with open(tuning_artifact.best_params_path, "r") as f:
        best_params = json.load(f)

    print("Best hyperparameters:", best_params)

    # Step 7: Quick prediction check
    test_data = np.load(transformation_artifact.transformed_test_path, allow_pickle=True)
    X_test = test_data["X_test"]
    print("Prediction sample:", tuned_model.predict(X_test[:5]))
