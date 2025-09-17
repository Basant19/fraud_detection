from src.entity.config import DataIngestionConfig, DataTransformationConfig
from src.components.data_ingestion import DataIngestion

def start_training_pipeline():
    # 1. Data Ingestion
    ingestion_config = DataIngestionConfig.get_default_config()
    data_ingestion = DataIngestion(ingestion_config)
    ingestion_artifacts = data_ingestion.initiate_data_ingestion()

    print(f"Train Data: {ingestion_artifacts.train_file_path}")
    print(f" Test Data: {ingestion_artifacts.test_file_path}")
    print(f" Raw Data: {ingestion_artifacts.raw_file_path}")


if __name__ == "__main__":
    start_training_pipeline()