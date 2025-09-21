import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

@dataclass
class DataIngestionConfig:
    mongo_uri: str = None
    db_name: str = None
    data_path: str = None
    collections: dict = field(default_factory=dict)

    def __post_init__(self):
        """
        Fills in values from .env if not provided explicitly.
        This way, you can override them manually if needed.
        """
        if self.mongo_uri is None:
            self.mongo_uri = os.getenv("MONGO_URI")

        if self.db_name is None:
            self.db_name = os.getenv("DB_NAME")

        if self.data_path is None:
            self.data_path = os.getenv("DATA_PATH", "data/")

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
        Returns a config with artifact paths included
        """
        config = DataIngestionConfig()
        config.raw_data_dir = os.path.join(base_dir, "raw_data")
        config.raw_file_path = os.path.join(config.raw_data_dir, "raw.csv")
        config.train_file_path = os.path.join(base_dir, "train.csv")
        config.test_file_path = os.path.join(base_dir, "test.csv")
        config.test_size = 0.2
        return config


@dataclass
class DataTransformationConfig:
    preprocessor_path: str = None
    random_state: int = None
    transformed_train_path: str = None
    transformed_test_path: str = None

    def __post_init__(self):
        if self.preprocessor_path is None:
            self.preprocessor_path = os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl")

        if self.random_state is None:
            self.random_state = int(os.getenv("RANDOM_STATE", 42))

        if self.transformed_train_path is None:
            self.transformed_train_path = os.getenv("TRANSFORMED_TRAIN_PATH", "artifacts/transformed_train.npz")

        if self.transformed_test_path is None:
            self.transformed_test_path = os.getenv("TRANSFORMED_TEST_PATH", "artifacts/transformed_test.npz")

    @staticmethod
    def get_default_config(base_dir: str = "artifacts") -> "DataTransformationConfig":
        return DataTransformationConfig(
            preprocessor_path=os.path.join(base_dir, "preprocessor.pkl"),
            transformed_train_path=os.path.join(base_dir, "transformed_train.npz"),
            transformed_test_path=os.path.join(base_dir, "transformed_test.npz"),
            random_state=int(os.getenv("RANDOM_STATE", 42))
        )

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
