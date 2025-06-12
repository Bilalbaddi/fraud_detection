from fraud_detection.exception.exception import fraud_detection_exception
from fraud_detection.logger.logging import logging
from fraud_detection.constant import training_pipeline
from datetime import datetime
import os
import sys
import numpy as np
import pandas as pd
from fraud_detection.constant import training_pipeline
from fraud_detection.entity.config_entity import TrainingPipelineConfig
from fraud_detection.entity.config_entity import DataIngestionConfig
from fraud_detection.entity.artifact_entity import DataIngestionArtifact
from dotenv import load_dotenv
load_dotenv()
import pymongo
import certifi
ca = certifi.where()
MONGO_DB_URL = os.getenv('MONGO_DB_URL')
from sklearn.model_selection import train_test_split



class DataIngestion:
    def __init__(self,data_ingestion_config:DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
    def export_collection_as_dataframe(self):
        try:
            collection = self.data_ingestion_config.collection_name
            database = self.data_ingestion_config.database_name
            self.client = pymongo.MongoClient(MONGO_DB_URL)
            collection = self.client[database][collection]
            df = pd.DataFrame(list(collection.find()))
            if "_id" in df.columns.to_list():
                df = df.drop(columns=["_id"],axis=1)
            print(df.head(2))
            return df
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
    def feature_engineering(self,df:pd.DataFrame):
        try:
            '''  '''
            df['TX_DATETIME'] =  df['TX_DATETIME'].astype(str)
            df['date'] = df['TX_DATETIME'].str.split(' ').str[0]
            df['time'] = df['TX_DATETIME'].str.split(' ').str[1]
            df = df.drop('TX_DATETIME',axis = 1)

            df['Year'] = df['date'].str.split('-').str[0]
            df['Month'] = df['date'].str.split('-').str[1]
            df['Day'] = df['date'].str.split('-').str[2]
            df = df.drop('date',axis = 1)

            df['Hour'] = df['time'].str.split(':').str[0]
            df['Minutes'] = df['time'].str.split(':').str[1]
            df['Seconds'] = df['time'].str.split(':').str[2]
            df = df.drop('time',axis =1)

            df['Day']=df['Day'].astype(int)
            df['Month']=df['Month'].astype(int)
            df['Year']=df['Year'].astype(int)
            df['Hour'] = df['Hour'].astype(int)
            df['Minutes'] = df['Minutes'].astype(int)
            df['Seconds'] = df['Seconds'].astype(int)
            df['CUSTOMER_ID'] = df['CUSTOMER_ID'].astype(int)
            df['TERMINAL_ID'] = df['TERMINAL_ID'].astype(int)
            df['TX_TIME_SECONDS'] = df['TX_TIME_SECONDS'].astype(int)
            df['TX_TIME_DAYS'] = df['TX_TIME_DAYS'].astype(int)
            return df
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
    def export_data_to_feature_store(self,df:pd.DataFrame):
        try:
            feature_store_file_path = self.data_ingestion_config.data_ingestion_feature_store_file_path
            dir_path = os.path.dirname(feature_store_file_path)
            os.makedirs(dir_path,exist_ok=True)
            df.to_csv(feature_store_file_path,index=False,header=True)
            logging.info(f"Data exported to feature store at {feature_store_file_path}")
            return df
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
        
    def split_data_as_train_test_split(self,df:pd.DataFrame):
        try:

            train_set,test_set = train_test_split(df,test_size=self.data_ingestion_config.train_test_split_ratio,random_state=42)
            logging.info("Performed tarin test split on the dataframe")

            train_file_path = self.data_ingestion_config.training_file_path
            test_file_path = self.data_ingestion_config.test_file_path
            dir_path = os.path.dirname(train_file_path)
            os.makedirs(dir_path,exist_ok=True)
            logging.info("directory is created")

            train_set.to_csv(train_file_path,index=False,header = True)
            test_set.to_csv(test_file_path,index= False,header= True)
            logging.info("Exported train and test data to respective files")
            

            return train_set,test_set
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
        
    def initiate_data_ingestion(self):
        try:
            df = self.export_collection_as_dataframe()
            clean_df = self.feature_engineering(df=df)
            clean_df = self.export_data_to_feature_store(clean_df)
            self.split_data_as_train_test_split(clean_df)
            logging.info("Data ingestion completed successfully")
            data_ingestion_artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.test_file_path
            )
            return data_ingestion_artifact
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
        
        
    