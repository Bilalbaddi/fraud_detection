from dotenv import load_dotenv
load_dotenv()
import numpy as np
import pandas as pd
from fraud_detection.logger.logging import logging
from fraud_detection.exception.exception import fraud_detection_exception
import json
import pymongo
import certifi
ca = certifi.where()
import os
import sys
import pickle
from fraud_detection.utils.main_utils.utils import save_csv_object

mongo_db_url = os.getenv('MONGO_DB_URL')
print(mongo_db_url)

class NetworkDataExtract:
    def __init__(self):
        try:
            pass
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
    def pickle_to_csv(self,file_path):

        all_data = []

        for filename in sorted(os.listdir(file_path)):
            if filename.endswith(".pkl"):
                try:
                    with open(os.path.join(file_path, filename), 'rb') as f:
                        # Use protocol=4 for better compatibility
                        df = pickle.load(f)
                        if isinstance(df, pd.DataFrame):
                            all_data.append(df)
                except ModuleNotFoundError:
                    # Alternative approach if pickle loading fails
                    print(f"Using alternative loading method for {filename}")
                    with open(os.path.join(file_path, filename), 'rb') as f:
                        # Try to load with pandas directly
                        df = pd.read_pickle(os.path.join(file_path, filename))
                        all_data.append(df)


        if all_data:
            final_df = pd.concat(all_data, ignore_index=True)
            print(final_df.shape)
            print(final_df.head())
        else:
            print("No DataFrames were successfully loaded")

        
        save_csv_object('csv_data/fraud.csv',final_df)

    def csv_to_json(self,file_path):
        try:
            df = pd.read_csv(file_path)
            df.reset_index(drop=True,inplace=True)
            records = list(json.loads(df.T.to_json()).values())
            return records
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
        
    def insert_data_to_mongodb(self,records,database,collection):
        try:
            self.records = records
            self.database = database
            self.collection = collection
            self.client = pymongo.MongoClient(mongo_db_url)
            self.database = self.client[self.database]
            self.collection = self.database[self.collection]
            self.collection.insert_many(self.records)
            return(len(self.records))
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
        
if __name__ == "__main__":
    pickle_file_path = 'data'
    FILE_PATH = "csv_data/fraud.csv"
    DATABASE = "frauddb"
    collection = "fraud_data"
    networkobj = NetworkDataExtract()
    networkobj.pickle_to_csv(pickle_file_path)
    records = networkobj.csv_to_json(file_path=FILE_PATH)
    print(records)
    no_of_records = networkobj.insert_data_to_mongodb(records, DATABASE, collection)
    print(no_of_records)

        

    
    
     
        