# D:\fraud_detection\test\test_model_evaluation.py

import os
import json
import pickle
import numpy as np

from src.entity.config import DataTransformationConfig, ModelTrainerConfig
from src.entity.artifacts import (
    DataIngestionArtifacts,
    DataTransformationArtifacts,
    ModelTrainerArtifacts,
)
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluator


if __name__ == "__main__":
    # ---------------- Step 1: Mock ingestion artifacts ----------------
    ingestion_artifact = DataIngestionArtifacts(
        train_file_path=r"D:\fraud_detection\artifacts\train.csv",
        test_file_path=r"D:\fraud_detection\artifacts\test.csv",
        raw_file_path=r"D:\fraud_detection\artifacts\raw_data\raw.csv",
        ingestion_metadata_path=r"D:\fraud_detection\artifacts\raw_data\ingestion_metadata.json"
    )

    # ---------------- Step 2: Transformation ----------------
    transformation_config = DataTransformationConfig.get_default_config()
    transformation = DataTransformation(transformation_config)
    transformation_artifact = transformation.initiate_data_transformation(ingestion_artifact)

    print("✅ Transformation successful")
    print("Transformation Artifacts:", transformation_artifact)

    # ---------------- Step 3: Train a model (if not already trained) ----------------
    trainer_config = ModelTrainerConfig.get_default_config()
    trainer = ModelTrainer(trainer_config)
    trainer_artifact = trainer.initiate_model_training(transformation_artifact)

    print("\n✅ Model training successful")
    print("ModelTrainerArtifacts:", trainer_artifact)

    # ---------------- Step 4: Run model evaluation ----------------
    evaluation_report_path = r"D:\fraud_detection\artifacts\evaluation\report.json"
    best_model_path = trainer_artifact.trained_model_path

    evaluator = ModelEvaluator(evaluation_report_path, best_model_path)
    evaluation_artifact = evaluator.evaluate_model(
        transformation_artifact, trainer_artifact
    )

    print("\n✅ Model evaluation successful")
    print("Evaluation Artifact:", evaluation_artifact)

    # ---------------- Step 5: Print report ----------------
    with open(evaluation_report_path, "r") as f:
        report = json.load(f)

    print("\n📊 Evaluation Report:")
    for metric, value in report.items():
        print(f"{metric}: {value}")

    # ---------------- Step 6: Quick prediction check ----------------
    test_data = np.load(transformation_artifact.transformed_test_path, allow_pickle=True)
    X_test = test_data["X_test"]

    with open(best_model_path, "rb") as f:
        best_model = pickle.load(f)

    print("\nPrediction sample:", best_model.predict(X_test[:5]))
