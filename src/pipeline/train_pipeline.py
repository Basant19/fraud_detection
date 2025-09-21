# D:\fraud_detection\src\pipeline\train_pipeline.py

from src.entity.config import DataIngestionConfig, DataTransformationConfig
from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation

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
    data_transformation = DataTransformation(transformation_config)

    # Pass the full ingestion_artifacts object, not individual paths
    transformation_artifacts = data_transformation.initiate_data_transformation(
        ingestion_artifact=ingestion_artifacts
    )

    print("✅ Data Transformation Completed")
    print(f"Transformed Train: {transformation_artifacts.transformed_train_path}")
    print(f" Transformed Test: {transformation_artifacts.transformed_test_path}")
    print(f" Preprocessor: {transformation_artifacts.preprocessor_path}")
    print(f" Feature Names: {transformation_artifacts.feature_names_path}")


if __name__ == "__main__":
    start_training_pipeline()
