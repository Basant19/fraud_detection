from src.entity.config import DataTransformationConfig
from src.entity.artifacts import DataIngestionArtifacts
from src.components.data_transformation import DataTransformation
import numpy as np

if __name__ == "__main__":
    # Mock ingestion artifacts (instead of running DataIngestion again)
    ingestion_artifact = DataIngestionArtifacts(
        train_file_path=r"D:\fraud_detection\artifacts\train.csv",
        test_file_path=r"D:\fraud_detection\artifacts\test.csv",
        raw_file_path=r"D:\fraud_detection\artifacts\raw_data\raw.csv"
    )

    # Load transformation config
    config = DataTransformationConfig.get_default_config()

    # Run transformation
    transformation = DataTransformation(config)
    artifacts = transformation.initiate_data_transformation(ingestion_artifact)

    print("✅ Transformation successful")
    print("Artifacts:", artifacts)

    # Verify loading the saved .npz files
    train_data = np.load(artifacts.transformed_train_path, allow_pickle=True)
    test_data = np.load(artifacts.transformed_test_path, allow_pickle=True)

    print("Train arrays:", train_data.files)
    print("Train X shape:", train_data["X_train"].shape)
    print("Train y shape:", train_data["y_train"].shape)

    print("Test arrays:", test_data.files)
    print("Test X shape:", test_data["X_test"].shape)
    print("Test y shape:", test_data["y_test"].shape)
