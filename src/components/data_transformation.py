import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE

from src.entity.config import DataTransformationConfig
from src.entity.artifacts import DataIngestionArtifacts, DataTransformationArtifacts
from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config

    def initiate_data_transformation(
        self, ingestion_artifact: DataIngestionArtifacts
    ) -> DataTransformationArtifacts:
        try:
            logging.info("Starting data transformation...")

            # Load train and test data (from ingestion artifacts)
            train_df = pd.read_csv(ingestion_artifact.train_file_path)
            test_df = pd.read_csv(ingestion_artifact.test_file_path)

            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # ---------------- Feature Engineering ----------------
            drop_cols = ["TransactionID", "MerchantID", "CustomerID", "Name", "Address", "LastLogin"]

            def feature_engineering(df):
                df = df.copy()

                # Timestamp handling
                df["Timestamp1"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                df["LastLogin_dt"] = pd.to_datetime(df["LastLogin"], errors="coerce")
                df["Hour"] = df["Timestamp1"].dt.hour.fillna(-1).astype(int)
                df["gap"] = (df["Timestamp1"] - df["LastLogin_dt"]).dt.days.abs()
                df["gap"] = df["gap"].fillna(df["gap"].median())

                # Drop identifiers + redundant timestamp cols
                df = df.drop(
                    columns=[c for c in drop_cols if c in df.columns]
                    + ["Timestamp", "Timestamp1", "LastLogin", "LastLogin_dt"],
                    errors="ignore",
                )

                # Fill missing numericals
                for col in df.select_dtypes(include=["number"]).columns:
                    df[col] = df[col].fillna(df[col].median())

                # Encode categorical "Category"
                le = LabelEncoder()
                if "Category" in df.columns:
                    df["Category"] = df["Category"].fillna("Unknown")
                    df["Category_enc"] = le.fit_transform(df["Category"])
                    df = df.drop(columns=["Category"], errors="ignore")

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

            # Numeric-only (after encoding)
            X_train = X_train.select_dtypes(include=[np.number]).copy()
            X_test = X_test.select_dtypes(include=[np.number]).copy()

            # ---------------- Scaling ----------------
            scaler = MinMaxScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            # ---------------- SMOTE ----------------
            smote = SMOTE(random_state=self.config.random_state)
            X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

            logging.info(
                f"After SMOTE -> Train: {X_train_smote.shape}, Test: {X_test_scaled.shape}"
            )

            # ---------------- Save Artifacts ----------------
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)

            # Save preprocessor
            with open(self.config.preprocessor_path, "wb") as f:
                pickle.dump(scaler, f)

            # Save transformed data using np.savez
            np.savez(
                self.config.transformed_train_path,
                X_train=X_train_smote,
                y_train=y_train_smote
            )
            np.savez(
                self.config.transformed_test_path,
                X_test=X_test_scaled,
                y_test=y_test
            )

            logging.info("Data transformation completed successfully.")

            return DataTransformationArtifacts(
                transformed_train_path=self.config.transformed_train_path,
                transformed_test_path=self.config.transformed_test_path,
                preprocessor_path=self.config.preprocessor_path,
            )

        except Exception as e:
            logging.error("Error in data transformation", exc_info=True)
            raise CustomException(e, sys)
