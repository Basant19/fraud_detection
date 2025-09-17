import os
import unittest
import tempfile
import pandas as pd

from src.entity.config import DataIngestionConfig
from src.components.data_ingestion import DataIngestion


class TestDataIngestion(unittest.TestCase):

    def setUp(self):
        """
        Create temporary directory and sample CSV for testing
        """
        self.test_dir = tempfile.TemporaryDirectory()
        raw_dir = os.path.join(self.test_dir.name, "raw_data")
        os.makedirs(raw_dir, exist_ok=True)

        self.raw_file = os.path.join(raw_dir, "raw.csv")
        df = pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "amount": [100, 200, 300, 400, 500],
            "label": [0, 1, 0, 1, 0],
        })
        df.to_csv(self.raw_file, index=False)

        # Prepare config
        self.config = DataIngestionConfig.get_default_config(base_dir=self.test_dir.name)
        self.config.raw_file_path = self.raw_file

    def tearDown(self):
        """
        Cleanup temporary directory
        """
        self.test_dir.cleanup()

    def test_data_ingestion_creates_train_test(self):
        ingestion = DataIngestion(config=self.config)
        artifacts = ingestion.initiate_data_ingestion()

        # Check files exist
        self.assertTrue(os.path.exists(artifacts.train_file_path))
        self.assertTrue(os.path.exists(artifacts.test_file_path))

        # Check dataset split
        train_df = pd.read_csv(artifacts.train_file_path)
        test_df = pd.read_csv(artifacts.test_file_path)

        self.assertEqual(len(train_df) + len(test_df), 5)
        self.assertFalse(train_df.empty)
        self.assertFalse(test_df.empty)


if __name__ == "__main__":
    unittest.main()
