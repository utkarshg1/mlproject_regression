# Import all the required libraries
import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass

# Initialize Data Ingestion Configuration
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts','train.csv')
    test_data_path: str = os.path.join('artifacts','test.csv')
    raw_data_path: str = os.path.join('artifacts','data.csv')

# Create a class for Data Ingestion
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initate_data_ingestion(self):
        logging.info('Data ingestion method Started')
        try:
            df = pd.read_csv('notebook/data/gemstone.csv')
            logging.info('Dataset read as pandas Dataframe')

            os.makedirs(os.path.dirname(self.ingestion_config.raw_data_path),exist_ok=True)

            df.to_csv(self.ingestion_config.raw_data_path,index=False)
            
            logging.info('Train Test Split Initiated')
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)

            train_set.to_csv(self.ingestion_config.train_data_path,index=False)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False)

            logging.info('Ingestion of Data is completed')

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )

        except Exception as e:
            logging.info('Exception occured at Data Ingestion stage')
            raise CustomException(e, sys)
    
# Run Data ingestion
if __name__ == '__main__':
    obj = DataIngestion()
    obj.initate_data_ingestion()

