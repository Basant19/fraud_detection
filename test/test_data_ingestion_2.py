from src.entity.config import DataIngestionConfig
from src.components.data_ingestion import DataIngestion

if __name__ == "__main__":
    config = DataIngestionConfig.get_default_config()
    ingestion = DataIngestion(config)
    artifacts = ingestion.initiate_data_ingestion()
    print(artifacts)
