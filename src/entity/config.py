
# src/entity/config.py
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DataIngestionConfig:
    # Connection + source
    mongo_uri: str = None
    db_name: str = None
    data_source: str = None           # "mongo" or "local"
    local_data_path: str = None       # CSV path used when data_source == "local"
    sample_limit: int = None

    # Artifact paths (populated by get_default_config)
    raw_data_dir: str = None
    raw_file_path: str = None
    train_file_path: str = None
    test_file_path: str = None
    ingestion_metadata_path: str = None

    # Behavior
    test_size: float = None
    min_for_stratify: int = None

    # Collections (for Mongo ingestion)
    collections: dict = field(default_factory=dict)

    def __post_init__(self):
        # Fill values from .env if not provided explicitly
        if self.mongo_uri is None:
            self.mongo_uri = os.getenv("MONGO_URI")

        if self.db_name is None:
            self.db_name = os.getenv("DB_NAME")

        if self.data_source is None:
            self.data_source = os.getenv("DATA_SOURCE", "mongo").lower()

        if self.local_data_path is None:
            self.local_data_path = os.getenv("LOCAL_DATA_PATH", None)

        if self.sample_limit is None:
            sample_limit_env = os.getenv("SAMPLE_LIMIT")
            self.sample_limit = int(sample_limit_env) if sample_limit_env else None

        if self.test_size is None:
            self.test_size = float(os.getenv("TEST_SIZE", 0.2))

        if self.min_for_stratify is None:
            self.min_for_stratify = int(os.getenv("MIN_FOR_STRATIFY", 30))

        if not self.collections:
            self.collections = {
                "transaction_records": os.getenv("COLLECTION_TRANSACTION_RECORDS"),
                "transaction_metadata": os.getenv("COLLECTION_TRANSACTION_METADATA"),
                "customer_data": os.getenv("COLLECTION_CUSTOMER_DATA"),
                "account_activity": os.getenv("COLLECTION_ACCOUNT_ACTIVITY"),
                "fraud_indicators": os.getenv("COLLECTION_FRAUD_INDICATORS"),
                "suspicious_activity": os.getenv("COLLECTION_SUSPICIOUS_ACTIVITY"),
                "amount_data": os.getenv("COLLECTION_AMOUNT_DATA"),
                "anomaly_scores": os.getenv("COLLECTION_ANOMALY_SCORES"),
                "merchant_data": os.getenv("COLLECTION_MERCHANT_DATA"),
                "transaction_category_labels": os.getenv("COLLECTION_TRANSACTION_CATEGORY_LABELS"),
            }

    @staticmethod
    def get_default_config(base_dir: str = "artifacts") -> "DataIngestionConfig":
        """
        Returns a config with artifact paths included.
        Use this helper to get consistent artifact locations.
        """
        config = DataIngestionConfig()

        config.raw_data_dir = os.path.join(base_dir, "raw_data")
        config.raw_file_path = os.path.join(config.raw_data_dir, "raw.csv")
        config.ingestion_metadata_path = os.path.join(config.raw_data_dir, "ingestion_metadata.json")
        config.train_file_path = os.path.join(base_dir, "train.csv")
        config.test_file_path = os.path.join(base_dir, "test.csv")

        # Keep defaults from env or class-level values
        config.test_size = getattr(config, "test_size", 0.2)
        config.min_for_stratify = getattr(config, "min_for_stratify", 30)

        return config

@dataclass
class DataTransformationConfig:
    preprocessor_path: str = None
    random_state: int = None
    transformed_train_path: str = None
    transformed_test_path: str = None
    feature_names_path: str = None  

    def __post_init__(self):
        if self.preprocessor_path is None:
            self.preprocessor_path = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl")

        if self.random_state is None:
            self.random_state = int(os.getenv("RANDOM_STATE", 42))

        if self.transformed_train_path is None:
            self.transformed_train_path = os.getenv("TRANSFORMED_TRAIN_PATH", "artifacts/transformed_train.npz")

        if self.transformed_test_path is None:
            self.transformed_test_path = os.getenv("TRANSFORMED_TEST_PATH", "artifacts/transformed_test.npz")

        if self.feature_names_path is None:
            self.feature_names_path = os.getenv("FEATURE_NAMES_PATH", "artifacts/feature_names.json")

    @staticmethod
    def get_default_config(base_dir: str = "artifacts") -> "DataTransformationConfig":
        return DataTransformationConfig(
            preprocessor_path=os.path.join(base_dir, "preprocessor.pkl"),
            transformed_train_path=os.path.join(base_dir, "transformed_train.npz"),
            transformed_test_path=os.path.join(base_dir, "transformed_test.npz"),
            feature_names_path=os.path.join(base_dir, "feature_names.json"),
            random_state=int(os.getenv("RANDOM_STATE", 42))
        )


'''
@dataclass
class ModelTrainerConfig:
    trained_model_path: str = None
    random_state: int = None

    def __post_init__(self):
        if self.trained_model_path is None:
            self.trained_model_path = os.getenv("TRAINED_MODEL_PATH", "artifacts/best_model.pkl")

        if self.random_state is None:
            self.random_state = int(os.getenv("RANDOM_STATE", 42))

    @staticmethod
    def get_default_config(base_dir: str = "artifacts") -> "ModelTrainerConfig":
        return ModelTrainerConfig(
            trained_model_path=os.path.join(base_dir, "best_model.pkl"),
            random_state=int(os.getenv("RANDOM_STATE", 42))
        )
'''