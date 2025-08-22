import os
import pandas as pd
from src.logger import logging
from src.exception import CustomException
import sys

class DataIngestion:
    def __init__(self, raw_data_dir="data", artifact_dir="artifacts"):
        # Get absolute path to project root (two levels up from this file)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        self.raw_data_dir = os.path.join(project_root, raw_data_dir)
        self.artifact_dir = os.path.join(project_root, artifact_dir)

    def initiate_data_ingestion(self):
        try:
            logging.info("===== Data Ingestion Started =====")

            filenames = ["features.csv", "train.csv", "stores.csv", "test.csv"]
            filenames = [f.strip().replace("\x0c", "") for f in filenames]

            # Build absolute paths
            features_path = os.path.join(self.raw_data_dir, filenames[0])
            train_path = os.path.join(self.raw_data_dir, filenames[1])
            stores_path = os.path.join(self.raw_data_dir, filenames[2])
            test_path = os.path.join(self.raw_data_dir, filenames[3])

            logging.info(f"Looking for data in: {self.raw_data_dir}")

            features = pd.read_csv(features_path)
            train = pd.read_csv(train_path)
            stores = pd.read_csv(stores_path)
            test = pd.read_csv(test_path)

            logging.info("Raw datasets loaded successfully")

            feature_store = features.merge(stores, how='left', on='Store')

            train_df = train.merge(feature_store, how='left', on=['Store', 'Date', 'IsHoliday']) \
                            .sort_values(by=['Store', 'Dept', 'Date']) \
                            .reset_index(drop=True)

            test_df = test.merge(feature_store, how='left', on=['Store', 'Date', 'IsHoliday']) \
                          .sort_values(by=['Store', 'Dept', 'Date']) \
                          .reset_index(drop=True)

            os.makedirs(self.artifact_dir, exist_ok=True)
            train_df.to_csv(os.path.join(self.artifact_dir, "train_merged.csv"), index=False)
            test_df.to_csv(os.path.join(self.artifact_dir, "test_merged.csv"), index=False)

            logging.info("===== Data Ingestion Completed =====")
            return (
                os.path.join(self.artifact_dir, "train_merged.csv"),
                os.path.join(self.artifact_dir, "test_merged.csv"),
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataIngestion()
    obj.initiate_data_ingestion()
