import sys
import os
import pandas as pd
import numpy as np

'''
constant for tarinig pipeline
'''
ARTIFACT_DIR : str = "Artifacts"
FILE_NAME : str = "fraud.csv"
TRAIN_FILE_NAME : str = "train.csv"
TEST_FILE_NAME : str = "test.csv"
TARGET_COLUMN = 'TX_FRAUD'
PIPELINE_NAME : str = 'fraud_detection'





'''
Data Ingestion constant
'''

DATA_INGESTION_DIR : str= 'fraud_detection'
DATA_INGESTION_FEATURE_STORE_FILE_DIR :str= 'feature_store'
DATA_INGESTION_INGESTED_DIR_NAME : str = 'ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO : float = 0.25
DATA_INGESTION_COLLECTION_NAME : str = 'fraud_data'
DATA_INGESTION_DATABASE_NAME :str = 'frauddb'
