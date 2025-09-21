#D:\fraud_detection\src\entity\artifacts.py
from dataclasses import dataclass

# Ingestion Stage Artifacts
@dataclass
class DataIngestionArtifacts:
    train_file_path: str
    test_file_path: str
    raw_file_path: str
    ingestion_metadata_path: str


# Transformation Stage Artifacts
@dataclass
class DataTransformationArtifacts:
    transformed_train_path: str
    transformed_test_path: str
    preprocessor_path: str
    feature_names_path: str   





# Model Training Stage Artifacts

@dataclass
class ModelTrainerArtifacts:
    trained_model_path: str
    training_score: float
    test_score: float





@dataclass
class HyperparameterTuningArtifacts:
    tuned_model_path: str
    best_params_path: str
    best_score: float



'''
# ----------------------------
# Model Evaluation Stage Artifacts
# ----------------------------
@dataclass
class ModelEvaluationArtifacts:
    evaluation_report_path: str
    best_model_path: str
    is_model_accepted: bool
    '''
