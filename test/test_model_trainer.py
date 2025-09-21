import os
import numpy as np
from src.entity.config import DataTransformationConfig, ModelTrainerConfig
from src.entity.artifacts import DataIngestionArtifacts
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

if __name__ == "__main__":
    # Step 1: Mock ingestion artifacts
    ingestion_artifact = DataIngestionArtifacts(
        train_file_path=r"D:\fraud_detection\artifacts\train.csv",
        test_file_path=r"D:\fraud_detection\artifacts\test.csv",
        raw_file_path=r"D:\fraud_detection\artifacts\raw_data\raw.csv"
    )

    # Step 2: Run transformation
    transformation_config = DataTransformationConfig.get_default_config()
    transformation = DataTransformation(transformation_config)
    transformation_artifact = transformation.initiate_data_transformation(ingestion_artifact)

    print("✅ Transformation successful")
    print("Transformation Artifacts:", transformation_artifact)

    # Step 3: Run model trainer
    model_trainer_config = ModelTrainerConfig.get_default_config()
    trainer = ModelTrainer(model_trainer_config)
    model_trainer_artifact = trainer.initiate_model_trainer(transformation_artifact)

    # Step 4: Print results
    print("\n✅ Model training successful")
    print("Best trained model path:", model_trainer_artifact.trained_model_path)
    print("Training Score:", model_trainer_artifact.training_score)
    print("Test Score (F1):", model_trainer_artifact.test_score)

    # Step 5: Load the saved model and verify
    import pickle
    with open(model_trainer_artifact.trained_model_path, "rb") as f:
        model = pickle.load(f)

    print("\nLoaded model type:", type(model))
    print("Prediction sample:", model.predict(
        np.load(transformation_artifact.transformed_test_path)["X_test"][:5]
    ))
