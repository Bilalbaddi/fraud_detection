from fraud_detection.entity.artifact_entity import DataValidationArtifact
from fraud_detection.entity.artifact_entity import DataIngestionArtifact
from fraud_detection.exception.exception import fraud_detection_exception
from fraud_detection.logger.logging import logging
from fraud_detection.entity.config_entity import DataValidationConfig
from fraud_detection.constant.training_pipeline import SCHEMA_FILE_PATH
import pandas as pd
import os
from scipy.stats import ks_2samp
import sys
from fraud_detection.utils.main_utils.utils import read_yaml_file,write_yaml_file


class DataValidation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,
                 data_validation_config:DataValidationConfig):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self.schema = read_yaml_file(SCHEMA_FILE_PATH)
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
        
    def check_columns(self,df:pd.DataFrame)-> bool:
        try:
            no_of_columns = len(self.schema['columns'])
            logging.info(f"Number of columns in the schema: {no_of_columns}")
            logging.info(f"Number of columns in the dataframe: {len(df.columns)}")
            if no_of_columns == len(df.columns):
                return True
            else:
                return False
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
        
    def validate_numerical_columns(self,df:pd.DataFrame)->bool:
        try:
            no_of_numerical_columns = len(self.schema['numerical_columns'])
            logging.info(f"Number of columns in the schema: {no_of_numerical_columns}")
            logging.info(f"Number of columns in the dataframe: {len(df.select_dtypes(include=['number']).columns)}")
            if no_of_numerical_columns == len(df.select_dtypes(include=['number']).columns):
                return True
            else:
                return False

        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
        
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise fraud_detection_exception(e, sys) from e
        
    def detect_dataset_drift(self,base_df,current_df,threshold = 0.05)-> bool:
        try:
            status = True
            report = {}

            for column in base_df.columns:
                d1 = base_df[column]
                d2 = current_df[column]
                is_same_dist = ks_2samp(d1,d2)
                if threshold <+ is_same_dist.pvalue:
                    is_found = False
                else:
                    is_found = True
                    status = False
                report.update({column:{
                    "p_value": is_same_dist.pvalue,
                    "is_found": is_found
                }})
            drift_report_file_path  = self.data_validation_config.drift_report_file_path
            dir_path = os.path.dirname(drift_report_file_path)
            os.makedirs(dir_path, exist_ok=True)

            write_yaml_file(file_path=drift_report_file_path,content=report)
                
        except Exception as e:
           raise fraud_detection_exception(e,sys) from e
        
    def initiate_data_validation(self)->DataValidationArtifact:
        try:
            train_file_path = self.data_ingestion_artifact.trained_file_path
            test_file_path = self.data_ingestion_artifact.test_file_path

            train_df = DataValidation.read_data(train_file_path)
            test_df = DataValidation.read_data(test_file_path)

            status = self.check_columns(train_df)
            if not status:
               print(f" Number of columns in the train file (train_df does not match with schema self._schema")
            status = self.check_columns(test_df)
            if not status:
                print(f" Number of columns in the test file (test_df does not match with schema self._schema")

            status = self.validate_numerical_columns(train_df)
            if not status:
               print(f" Number of columns in the train file (train_df does not match with schema self._schema")
            status = self.validate_numerical_columns(test_df)
            if not status:
                print(f" Number of columns in the test file (test_df does not match with schema self._schema")
            status = self.detect_dataset_drift(base_df=train_df,current_df=test_df)
            dir_path = os.path.dirname(self.data_validation_config.valid_test_dir)
            os.makedirs(dir_path, exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_dir, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_dir, index=False)

            data_validation_artifact = DataValidationArtifact(
                validation_status=status,
                valid_train_file_path=  self.data_ingestion_artifact.trained_file_path,
                valid_test_file_path= self.data_ingestion_artifact.test_file_path,
                invalid_train_file_path= None,
                invalid_test_file_path= None,
                drift_report_file_path=  self.data_validation_config.drift_report_file_path
                
            )

            return data_validation_artifact
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
            
    