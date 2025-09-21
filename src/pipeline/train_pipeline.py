from src.entity.config import DataIngestionConfig, DataTransformationConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer, ModelTrainerConfig


def start_training_pipeline():
    # 1. Data Ingestion
    ingestion_config = DataIngestionConfig.get_default_config()
    data_ingestion = DataIngestion(ingestion_config)
    ingestion_artifacts = data_ingestion.initiate_data_ingestion()

    print(f"✅ Data Ingestion Completed")
    print(f"Train Data: {ingestion_artifacts.train_file_path}")
    print(f" Test Data: {ingestion_artifacts.test_file_path}")
    print(f" Raw Data: {ingestion_artifacts.raw_file_path}")

    # 2. Data Transformation
    transformation_config = DataTransformationConfig.get_default_config()
    data_transformation = DataTransformation(transformation_config)
    transformation_artifacts = data_transformation.initiate_data_transformation(
        ingestion_artifacts
    )

    print(f"\n✅ Data Transformation Completed")
    print(f"Preprocessor Object Path: {transformation_artifacts.preprocessor_path}")
    print(f"Transformed Train Path: {transformation_artifacts.transformed_train_path}")
    print(f"Transformed Test Path: {transformation_artifacts.transformed_test_path}")

    # 3. Model Training
    trainer_config = ModelTrainerConfig.get_default_config()
    model_trainer = ModelTrainer(trainer_config)

    trainer_artifacts = model_trainer.initiate_model_training(transformation_artifacts)

    print(f"\n✅ Model Training Completed")
    print(f"Best Model Path: {trainer_artifacts.best_model_path}")
    print(f"Metrics: {trainer_artifacts.metrics}")




if __name__ == "__main__":
    start_training_pipeline()
