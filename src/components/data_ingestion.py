import os
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from pymongo import MongoClient

from src.logger import get_logger
from src.exception import CustomException
from src.entity.config import DataIngestionConfig
from src.entity.artifacts import DataIngestionArtifacts

logger = get_logger(__name__)


def fetch_collection_as_df(db, collection_name, sample_limit=None):
    """
    Fetch MongoDB collection as a pandas DataFrame
    """
    cursor = db[collection_name].find()
    if sample_limit:
        cursor = cursor.limit(sample_limit)
    df = pd.DataFrame(list(cursor))
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])
    return df


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def initiate_data_ingestion(self) -> DataIngestionArtifacts:
        """
        Reads from MongoDB, merges collections, creates raw.csv, 
        then splits into train/test.
        """
        logger.info("Starting data ingestion process...")

        try:
            # Connect to MongoDB
            client = MongoClient(self.config.mongo_uri)
            db = client[self.config.db_name]
            logger.info(f"Connected to MongoDB: {self.config.db_name}")

            # Load collections into DataFrames
            account = fetch_collection_as_df(db, self.config.collections["account_activity"])
            customer = fetch_collection_as_df(db, self.config.collections["customer_data"])
            fraud = fetch_collection_as_df(db, self.config.collections["fraud_indicators"])
            suspicion = fetch_collection_as_df(db, self.config.collections["suspicious_activity"])
            merchant = fetch_collection_as_df(db, self.config.collections["merchant_data"])
            tran_cat = fetch_collection_as_df(db, self.config.collections["transaction_category_labels"])
            amount = fetch_collection_as_df(db, self.config.collections["amount_data"])
            anomaly = fetch_collection_as_df(db, self.config.collections["anomaly_scores"])
            tran_data = fetch_collection_as_df(db, self.config.collections["transaction_metadata"])
            tran_rec = fetch_collection_as_df(db, self.config.collections["transaction_records"])

            # Merge customer related
            customer_data = pd.merge(customer, account, on="CustomerID", how="left")
            customer_data = pd.merge(customer_data, suspicion, on="CustomerID", how="left")

            # Merge transaction pieces
            transaction_data1 = pd.merge(fraud, tran_cat, on="TransactionID", how="left")
            transaction_data2 = pd.merge(amount, anomaly, on="TransactionID", how="left")
            transaction_data3 = pd.merge(tran_data, tran_rec, on="TransactionID", how="left")

            transaction_data = pd.merge(transaction_data1, transaction_data2, on="TransactionID", how="left")
            transaction_data = pd.merge(transaction_data, transaction_data3, on="TransactionID", how="left")

            # Final dataset
            data = pd.merge(transaction_data, customer_data, on="CustomerID", how="left")
            logger.info(f"Final merged data shape: {data.shape}")

            # Create directories
            os.makedirs(self.config.raw_data_dir, exist_ok=True)

            # Save raw file
            data.to_csv(self.config.raw_file_path, index=False)
            logger.info(f"Raw data saved at {self.config.raw_file_path}")

            # Split dataset
            train_set, test_set = train_test_split(
                data, test_size=self.config.test_size, random_state=42
            )

            os.makedirs(os.path.dirname(self.config.train_file_path), exist_ok=True)

            train_set.to_csv(self.config.train_file_path, index=False)
            test_set.to_csv(self.config.test_file_path, index=False)

            logger.info(
                f"Data ingestion completed: train ({train_set.shape}), test ({test_set.shape})"
            )

            return DataIngestionArtifacts(
                train_file_path=self.config.train_file_path,
                test_file_path=self.config.test_file_path,
                raw_file_path=self.config.raw_file_path,
            )

        except Exception as e:
            logger.error("Error in data ingestion", exc_info=True)
            raise CustomException(e, sys)
