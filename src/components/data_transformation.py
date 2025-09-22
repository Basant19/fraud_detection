# src/components/data_transformation.py
import os
import sys
import pandas as pd
import numpy as np
import pickle
import json
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

from src.entity.config import DataTransformationConfig
from src.entity.artifacts import DataIngestionArtifacts, DataTransformationArtifacts
from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, resampling_strategy: str = "smote_tomek"):
        self.config = config
        self.resampling_strategy = resampling_strategy.lower()

    def initiate_data_transformation(self, ingestion_artifact: DataIngestionArtifacts) -> DataTransformationArtifacts:
        try:
            logging.info("Starting data transformation...")

            # Load train and test data
            train_df = pd.read_csv(ingestion_artifact.train_file_path)
            test_df = pd.read_csv(ingestion_artifact.test_file_path)
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # ---------------- Feature Engineering ----------------
            drop_cols = ["TransactionID", "MerchantID", "CustomerID", "Name", "Address", "LastLogin"]

            def feature_engineering(df):
                df = df.copy()

                # Timestamp handling
                if "Timestamp" in df.columns:
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                    df["Hour"] = df["Timestamp"].dt.hour.fillna(-1).astype(int)
                if "LastLogin" in df.columns:
                    df["LastLogin"] = pd.to_datetime(df["LastLogin"], errors="coerce")
                if "Timestamp" in df.columns and "LastLogin" in df.columns:
                    df["gap"] = (df["Timestamp"] - df["LastLogin"]).dt.days.abs()
                    df["gap"] = df["gap"].fillna(df["gap"].median())
                else:
                    df["gap"] = 0

                # Drop identifiers
                df = df.drop(columns=[c for c in drop_cols if c in df.columns] + ["Timestamp", "LastLogin"], errors="ignore")
                return df

            train_df = feature_engineering(train_df)
            test_df = feature_engineering(test_df)

            # ---------------- Split Features & Target ----------------
            if "FraudIndicator" not in train_df.columns:
                raise ValueError("FraudIndicator (target) not found in dataset")

            X_train = train_df.drop(columns=["FraudIndicator"])
            y_train = train_df["FraudIndicator"]
            X_test = test_df.drop(columns=["FraudIndicator"])
            y_test = test_df["FraudIndicator"]

            # Identify numeric and categorical columns
            numeric_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
            categorical_cols = X_train.select_dtypes(include=["object"]).columns.tolist()

            # ---------------- Preprocessing Pipeline ----------------
            numeric_pipeline = Pipeline([
                ("scaler", MinMaxScaler())
            ])

            categorical_pipeline = Pipeline([
                ("onehot", OneHotEncoder(handle_unknown="ignore"))
            ]) if categorical_cols else None

            if categorical_pipeline:
                preprocessor = ColumnTransformer([
                    ("num", numeric_pipeline, numeric_cols),
                    ("cat", categorical_pipeline, categorical_cols)
                ])
            else:
                preprocessor = numeric_pipeline  # only numeric data

            # Fit-transform on train, transform test
            X_train_transformed = preprocessor.fit_transform(X_train)
            X_test_transformed = preprocessor.transform(X_test)

            # ---------------- Resampling ----------------
            if self.resampling_strategy == "smote":
                sampler = SMOTE(random_state=self.config.random_state)
            elif self.resampling_strategy == "tomek":
                sampler = TomekLinks()
            elif self.resampling_strategy == "smote_tomek":
                sampler = SMOTETomek(random_state=self.config.random_state)
            else:
                raise ValueError(f"Unknown resampling strategy: {self.resampling_strategy}")

            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_transformed, y_train)
            logging.info(
                f"After {self.resampling_strategy.upper()} -> Train: {X_train_resampled.shape}, Test: {X_test_transformed.shape}"
            )

            # ---------------- Save Artifacts ----------------
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)
            with open(self.config.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

            np.savez(self.config.transformed_train_path, X_train=X_train_resampled, y_train=y_train_resampled)
            np.savez(self.config.transformed_test_path, X_test=X_test_transformed, y_test=y_test)

            # Save feature names
            feature_names = numeric_cols + categorical_cols
            with open(self.config.feature_names_path, "w") as f:
                json.dump(feature_names, f, indent=2)

            logging.info("Data transformation completed successfully.")

            return DataTransformationArtifacts(
                transformed_train_path=self.config.transformed_train_path,
                transformed_test_path=self.config.transformed_test_path,
                preprocessor_path=self.config.preprocessor_path,
                feature_names_path=self.config.feature_names_path,
            )

        except Exception as e:
            logging.error("Error in data transformation", exc_info=True)
            raise CustomException(e, sys)
