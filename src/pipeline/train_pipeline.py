# D:\fraud_detection\src\pipeline\train_pipeline.py

from src.entity.config import (
    DataIngestionConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    HyperparameterTuningConfig,
    ModelEvaluationConfig,
)
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.hyperparameter_tuning import HyperparameterTuner
from src.components.model_evaluation import ModelEvaluator


def start_training_pipeline():
    # ----------------------
    # 1. Data Ingestion
    # ----------------------
    ingestion_config = DataIngestionConfig.get_default_config()
    data_ingestion = DataIngestion(ingestion_config)
    ingestion_artifacts = data_ingestion.initiate_data_ingestion()

    print("✅ Data Ingestion Completed")
    print(f"Train Data: {ingestion_artifacts.train_file_path}")
    print(f" Test Data: {ingestion_artifacts.test_file_path}")
    print(f" Raw Data: {ingestion_artifacts.raw_file_path}")
    print(f" Metadata: {ingestion_artifacts.ingestion_metadata_path}")

    # ----------------------
    # 2. Data Transformation
    # ----------------------
    transformation_config = DataTransformationConfig.get_default_config()
    data_transformation = DataTransformation(transformation_config, resampling_strategy="smote_tomek")

    transformation_artifacts = data_transformation.initiate_data_transformation(
        ingestion_artifact=ingestion_artifacts
    )

    print("✅ Data Transformation Completed")
    print(f"Transformed Train: {transformation_artifacts.transformed_train_path}")
    print(f" Transformed Test: {transformation_artifacts.transformed_test_path}")
    print(f" Preprocessor: {transformation_artifacts.preprocessor_path}")
    print(f" Feature Names: {transformation_artifacts.feature_names_path}")

    # ----------------------
    # 3. Baseline Model Training
    # ----------------------
    trainer_config = ModelTrainerConfig.get_default_config()
    model_trainer = ModelTrainer(trainer_config)

    model_artifacts = model_trainer.initiate_model_training(
        transformation_artifact=transformation_artifacts
    )

    print("✅ Model Training Completed")
    print(f"Best Baseline Model: {model_artifacts.trained_model_path}")
    print(f" Training Score: {model_artifacts.training_score:.4f}")
    print(f" Test (F2) Score: {model_artifacts.test_score:.4f}")

    # ----------------------
    # 4. Hyperparameter Tuning
    # ----------------------
    tuning_config = HyperparameterTuningConfig.get_default_config()
    tuner = HyperparameterTuner(tuning_config)

    tuning_artifacts = tuner.initiate_hyperparameter_tuning(
        transformation_artifact=transformation_artifacts
    )

    print("✅ Hyperparameter Tuning Completed")
    print(f"Tuned Model Path: {tuning_artifacts.tuned_model_path}")
    print(f" Best Params Path: {tuning_artifacts.best_params_path}")
    print(f" Best CV Score (F2): {tuning_artifacts.best_score:.4f}")

    # ----------------------
    # ----------------------
    # 5. Model Evaluation
    # ----------------------
    evaluation_config = ModelEvaluationConfig.get_default_config()
    # Use the baseline trained model or the tuned model for evaluation
    best_model_path = tuning_artifacts.tuned_model_path  # or model_artifacts.trained_model_path
    evaluator = ModelEvaluator(evaluation_config, best_model_path)

    evaluation_artifacts = evaluator.evaluate_model(
    transformation_artifact=transformation_artifacts,
    trainer_artifact=model_artifacts
    )




    print("✅ Model Evaluation Completed")
    print(f"Evaluation Report: {evaluation_artifacts.evaluation_report_path}")
    print(f" Best Model Path: {evaluation_artifacts.best_model_path}")
    print(f" Is Model Accepted? {evaluation_artifacts.is_model_accepted}")
    print(f" Acceptance Criteria: {evaluation_artifacts.acceptance_criteria}")


if __name__ == "__main__":
    start_training_pipeline()
