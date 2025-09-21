# src/components/data_transformation.py
import os
import sys
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

from src.entity.config import DataTransformationConfig
from src.entity.artifacts import DataIngestionArtifacts, DataTransformationArtifacts
from src.exception import CustomException
from src.logger import logging


class DataTransformation:
    def __init__(self, config: DataTransformationConfig, resampling_strategy: str = "smote"):
        """
        resampling_strategy: "smote", "tomek", "smote_tomek"
        """
        self.config = config
        self.label_encoder = LabelEncoder()  # For consistent encoding
        self.resampling_strategy = resampling_strategy.lower()

    def initiate_data_transformation(
        self, ingestion_artifact: DataIngestionArtifacts
    ) -> DataTransformationArtifacts:
        try:
            logging.info("Starting data transformation...")

            # Load train and test data
            train_df = pd.read_csv(ingestion_artifact.train_file_path)
            test_df = pd.read_csv(ingestion_artifact.test_file_path)
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

            # ---------------- Feature Engineering ----------------
            drop_cols = ["TransactionID", "MerchantID", "CustomerID", "Name", "Address", "LastLogin"]

            def feature_engineering(df, fit_encoder=False):
                df = df.copy()

                # Timestamp handling
                if "Timestamp" in df.columns:
                    df["Timestamp1"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                    df["Hour"] = df["Timestamp1"].dt.hour.fillna(-1).astype(int)
                if "LastLogin" in df.columns:
                    df["LastLogin_dt"] = pd.to_datetime(df["LastLogin"], errors="coerce")
                if "Timestamp1" in df.columns and "LastLogin_dt" in df.columns:
                    df["gap"] = (df["Timestamp1"] - df["LastLogin_dt"]).dt.days.abs()
                    df["gap"] = df["gap"].fillna(df["gap"].median())
                else:
                    df["gap"] = 0  # fallback if columns missing

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
                if "Category" in df.columns:
                    df["Category"] = df["Category"].fillna("Unknown")
                    if fit_encoder:
                        df["Category_enc"] = self.label_encoder.fit_transform(df["Category"])
                    else:
                        df["Category_enc"] = self.label_encoder.transform(df["Category"])
                    df = df.drop(columns=["Category"], errors="ignore")

                return df

            # Apply feature engineering (fit encoder on train, transform on test)
            train_df = feature_engineering(train_df, fit_encoder=True)
            test_df = feature_engineering(test_df, fit_encoder=False)

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

            # ---------------- Resampling ----------------
            if self.resampling_strategy == "smote":
                sampler = SMOTE(random_state=self.config.random_state)
            elif self.resampling_strategy == "tomek":
                sampler = TomekLinks()
            elif self.resampling_strategy == "smote_tomek":
                sampler = SMOTETomek(random_state=self.config.random_state)
            else:
                raise ValueError(f"Unknown resampling strategy: {self.resampling_strategy}")

            X_train_resampled, y_train_resampled = sampler.fit_resample(X_train_scaled, y_train)
            logging.info(
                f"After {self.resampling_strategy.upper()} -> Train: {X_train_resampled.shape}, Test: {X_test_scaled.shape}"
            )

            # ---------------- Save Artifacts ----------------
            os.makedirs(os.path.dirname(self.config.preprocessor_path), exist_ok=True)

            # Save preprocessing objects (scaler + label encoder) as a dict
            preprocessor = {"scaler": scaler, "label_encoder": self.label_encoder,"feature_names": X_train.columns.tolist()}
            with open(self.config.preprocessor_path, "wb") as f:
                pickle.dump(preprocessor, f)

            # Save transformed data using np.savez
            np.savez(
                self.config.transformed_train_path,
                X_train=X_train_resampled,
                y_train=y_train_resampled
            )
            np.savez(
                self.config.transformed_test_path,
                X_test=X_test_scaled,
                y_test=y_test
            )

            logging.info("Data transformation completed successfully.")
            # Save feature names to JSON
            import json
            with open(self.config.feature_names_path, "w") as f:

                json.dump(X_train.columns.tolist(), f, indent=2)

            return DataTransformationArtifacts(
                transformed_train_path=self.config.transformed_train_path,
                transformed_test_path=self.config.transformed_test_path,
                preprocessor_path=self.config.preprocessor_path,
                feature_names_path=self.config.feature_names_path,
            )

        except Exception as e:
            logging.error("Error in data transformation", exc_info=True)
            raise CustomException(e, sys)
