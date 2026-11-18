import os
import sys
import numpy as np
import pandas as pd
from src.Airbnb.logger import logging
from src.Airbnb.exception import customexception
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from pathlib import Path

class DataIngestionConfig:
    base_dir: str = "airbnb-price-prediction"
    raw_train_path: str = os.path.join(base_dir, "Airbnb_train.csv")
    raw_test_path: str = os.path.join(base_dir, "Airbnb_test.csv")
    train_data_path: str = os.path.join(base_dir, "train_split.csv")
    val_data_path: str = os.path.join(base_dir, "val_split.csv")


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data ingestion started")
        try:
            data = pd.read_csv(self.ingestion_config.raw_train_path)
            logging.info("Read the labeled training file")

            data["log_price"] = np.log(data["price"])

            # price -> log_price (all prices are > 0)
            # data["log_price"] = np.log(data["price"])
            # data = data.drop(columns=["price"])   # avoid target leakage

            train_data, val_data = train_test_split(data, test_size=0.2, random_state=42)

            train_data.to_csv(self.ingestion_config.train_data_path, index=False)
            val_data.to_csv(self.ingestion_config.val_data_path, index=False)
            logging.info("Train/validation split saved")

            return (self.ingestion_config.train_data_path,
                    self.ingestion_config.val_data_path)
        except Exception as e:
            logging.info("Exception occurred while ingesting the data")
            raise customexception(e, sys)