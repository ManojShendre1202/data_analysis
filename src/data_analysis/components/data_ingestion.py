import os
import sys
from data_analysis.utils.exception import CustomException
from data_analysis.utils.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('data','train.csv')
    test_data_path:str=os.path.join('data','test.csv')

class DataIngestion:
    def __init__(self):
        self.ingestion_config=DataIngestionConfig()

    def Start_data_ingestion(self):
        try:
            ##reading the data from mysql
            df=pd.read_csv('C:/datascienceprojects/data_analysis/data/raw.csv')

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)

            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Data Ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            raise CustomException(e,sys)