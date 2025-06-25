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

SCHEMA_FILE_PATH = os.path.join("Data_schema", "schema.yaml")

SAVED_MODEL_DIR :str = os.path.join("saved_models")
SAVED_MODEL_NAME : str = "model.pkl"




'''
Data Ingestion constant
'''

DATA_INGESTION_DIR : str= 'fraud_detection'
DATA_INGESTION_FEATURE_STORE_FILE_DIR :str= 'feature_store'
DATA_INGESTION_INGESTED_DIR_NAME : str = 'ingested'
DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO : float = 0.25
DATA_INGESTION_COLLECTION_NAME : str = 'fraud_data'
DATA_INGESTION_DATABASE_NAME :str = 'frauddb'




'''
data Validation constant
'''

DATA_VALIDATION_DIR_NAME : str= 'data_validation'
DATA_VALIDATION_VALID_DIR_NAME :str= 'valid'
DATA_VALIDATION_INVALID_DIR_NAME : str= 'invalid'
DATA_VALIDATION_DRIFT_REPORT_DIR :str = 'drift_report'
DATA_VALIDATION_DRIFT_REPORT_FILE_NAME :  str = 'report.yaml'


'''
data Transformation constant
'''
DATA_TRANSFORMATION_DIR_NAME : str = "data_transformation"
DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR : str = "transformed"
DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR : str = "transformed_object"
PREPROCESSING_OBJECT_FILE_NAME :str = "preprocessing.pkl"
DATA_TRANSFORMATION_TRAINED_FILE_PATH : str = 'train.npy'
DATA_TRANSFORMATION_TEST_FILE_PATH : str = 'test.npy'

DATA_TRANSFORMATION_TRAINED_FILE_PATH : str = 'train.npy'
DATA_TRANSFORMATION_TEST_FILE_PATH : str = 'test.npy'


'''
Model tariner constant
'''

MODEL_TRAINER_DIR_NAME : str = "model_trainer"
MODEL_TRAINER_TRAINED_MODEL_DIR : str = "trained_model"
MODEL_TRAINER_TRAINED_MODEL_NAME : str = "model.pkl"
MODEL_TRAINER_EXPECTED_SCORE : float = 0.6
MODEL_TRAINER_OVERFITTING_UNDERFITTING_THRESHOLD : float = 0.05