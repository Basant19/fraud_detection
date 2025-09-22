# D:\fraud_detection\src\entity\config.py
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# Data Ingestion Config
# ----------------------------
@dataclass
class DataIngestionConfig:
    # Connection + source
    mongo_uri: str = None
    db_name: str = None
    data_source: str = None
    local_data_path: str = None
    sample_limit: int = None

    # Artifact paths
    raw_data_dir: str = None
    raw_file_path: str = None
    train_file_path: str = None
    test_file_path: str = None
    ingestion_metadata_path: str = None

    # Behavior
    test_size: float = None
    min_for_stratify: int = None

    # Collections
    collections: dict = field(default_factory=dict)

    def __post_init__(self):
        self.mongo_uri = self.mongo_uri or os.getenv("MONGO_URI")
        self.db_name = self.db_name or os.getenv("DB_NAME")
        self.data_source = (self.data_source or os.getenv("DATA_SOURCE", "mongo")).lower()
        self.local_data_path = self.local_data_path or os.getenv("LOCAL_DATA_PATH")
        self.sample_limit = self.sample_limit or (
            int(os.getenv("SAMPLE_LIMIT")) if os.getenv("SAMPLE_LIMIT") else None
        )
        self.test_size = self.test_size or float(os.getenv("TEST_SIZE", 0.2))
        self.min_for_stratify = self.min_for_stratify or int(os.getenv("MIN_FOR_STRATIFY", 30))

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
        cfg = DataIngestionConfig()
        cfg.raw_data_dir = os.path.join(base_dir, "raw_data")
        cfg.raw_file_path = os.path.join(cfg.raw_data_dir, "raw.csv")
        cfg.ingestion_metadata_path = os.path.join(cfg.raw_data_dir, "ingestion_metadata.json")
        cfg.train_file_path = os.path.join(base_dir, "train.csv")
        cfg.test_file_path = os.path.join(base_dir, "test.csv")
        return cfg


# ----------------------------
# Data Transformation Config
# ----------------------------
@dataclass
class DataTransformationConfig:
    preprocessor_path: str = None
    transformed_train_path: str = None
    transformed_test_path: str = None
    feature_names_path: str = None
    random_state: int = None

    def __post_init__(self):
        self.preprocessor_path = self.preprocessor_path or os.getenv("PREPROCESSOR_PATH", "artifacts/preprocessor.pkl")
        self.transformed_train_path = self.transformed_train_path or os.getenv("TRANSFORMED_TRAIN_PATH", "artifacts/transformed_train.npz")
        self.transformed_test_path = self.transformed_test_path or os.getenv("TRANSFORMED_TEST_PATH", "artifacts/transformed_test.npz")
        self.feature_names_path = self.feature_names_path or os.getenv("FEATURE_NAMES_PATH", "artifacts/feature_names.json")
        self.random_state = self.random_state or int(os.getenv("RANDOM_STATE", 42))

    @staticmethod
    def get_default_config(base_dir: str = "artifacts") -> "DataTransformationConfig":
        return DataTransformationConfig(
            preprocessor_path=os.path.join(base_dir, "preprocessor.pkl"),
            transformed_train_path=os.path.join(base_dir, "transformed_train.npz"),
            transformed_test_path=os.path.join(base_dir, "transformed_test.npz"),
            feature_names_path=os.path.join(base_dir, "feature_names.json"),
            random_state=int(os.getenv("RANDOM_STATE", 42))
        )


# ----------------------------
# Model Trainer Config
# ----------------------------
@dataclass
class ModelTrainerConfig:
    trained_model_path: str = None
    random_state: int = None

    def __post_init__(self):
        self.trained_model_path = self.trained_model_path or os.getenv("TRAINED_MODEL_PATH", "artifacts/best_model.pkl")
        self.random_state = self.random_state or int(os.getenv("RANDOM_STATE", 42))

    @staticmethod
    def get_default_config(base_dir: str = "artifacts") -> "ModelTrainerConfig":
        return ModelTrainerConfig(
            trained_model_path=os.path.join(base_dir, "best_model.pkl"),
            random_state=int(os.getenv("RANDOM_STATE", 42))
        )


# ----------------------------
# Hyperparameter Tuning Config
# ----------------------------
@dataclass
class HyperparameterTuningConfig:
    tuned_model_path: str = None
    best_params_path: str = None
    search_strategy: str = None
    random_state: int = None

    def __post_init__(self):
        self.tuned_model_path = self.tuned_model_path or os.getenv("TUNED_MODEL_PATH", "artifacts/tuned_model.pkl")
        self.best_params_path = self.best_params_path or os.getenv("BEST_PARAMS_PATH", "artifacts/best_params.json")
        self.search_strategy = (self.search_strategy or os.getenv("SEARCH_STRATEGY", "optuna")).lower()
        self.random_state = self.random_state or int(os.getenv("RANDOM_STATE", 42))

    @staticmethod
    def get_default_config(base_dir: str = "artifacts") -> "HyperparameterTuningConfig":
        return HyperparameterTuningConfig(
            tuned_model_path=os.path.join(base_dir, "tuned_model.pkl"),
            best_params_path=os.path.join(base_dir, "best_params.json"),
            search_strategy=os.getenv("SEARCH_STRATEGY", "optuna").lower(),
            random_state=int(os.getenv("RANDOM_STATE", 42))
        )


# ----------------------------
# Model Evaluation Config
# ----------------------------
@dataclass
class ModelEvaluationConfig:
    evaluation_report_path: str = None
    min_recall: float = 0.15
    min_f2: float = 0.15

    @staticmethod
    def get_default_config(base_dir: str = "artifacts") -> "ModelEvaluationConfig":
        return ModelEvaluationConfig(
            evaluation_report_path=os.path.join(base_dir, "evaluation_report.json"),
            min_recall=float(os.getenv("MIN_RECALL", 0.15)),
            min_f2=float(os.getenv("MIN_F2", 0.15)),
        )

# ----------------------------
# Model Prediction Config
# ----------------------------
@dataclass
class ModelPredictionConfig:
    trained_model_path: str = None
    preprocessor_path: str = None
    feature_names_path: str = None
    prediction_dir: str = None   # NEW: store predictions in artifacts/model_prediction

    def __post_init__(self):
        self.trained_model_path = self.trained_model_path or os.getenv(
            "TRAINED_MODEL_PATH", "artifacts/best_model.pkl"
        )
        self.preprocessor_path = self.preprocessor_path or os.getenv(
            "PREPROCESSOR_PATH", "artifacts/preprocessor.pkl"
        )
        self.feature_names_path = self.feature_names_path or os.getenv(
            "FEATURE_NAMES_PATH", "artifacts/feature_names.json"
        )
        # Default prediction dir
        self.prediction_dir = self.prediction_dir or os.getenv(
            "PREDICTION_DIR", "artifacts/model_prediction"
        )
        os.makedirs(self.prediction_dir, exist_ok=True)

    @staticmethod
    def get_default_config(base_dir: str = "artifacts") -> "ModelPredictionConfig":
        prediction_dir = os.path.join(base_dir, "model_prediction")
        os.makedirs(prediction_dir, exist_ok=True)

        return ModelPredictionConfig(
            trained_model_path=os.path.join(base_dir, "best_model.pkl"),
            preprocessor_path=os.path.join(base_dir, "preprocessor.pkl"),
            feature_names_path=os.path.join(base_dir, "feature_names.json"),
            prediction_dir=prediction_dir,
        )

