import os
import sys
import pandas as pd
from fraud_detection.exception.exception import fraud_detection_exception
from fraud_detection.logger.logging import logging
import numpy as np
from fraud_detection.constant.training_pipeline import TARGET_COLUMN
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from fraud_detection.entity.artifact_entity import(
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)
from fraud_detection.entity.config_entity import DataTransformationConfig

from fraud_detection.utils.main_utils.utils import save_numpy_array,save_object
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from  imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


class DataTansformation:
    def __init__(self,data_validation_artifact: DataValidationArtifact,
                 data_transformation_config : DataTransformationConfig):
        try:
            self.data_validation_artifact : DataValidationArtifact = data_validation_artifact
            self.data_transformation_config : DataTransformationConfig = data_transformation_config
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
    @staticmethod
    def read_data(file_path)-> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise fraud_detection_exception(e, sys) from e
        
    def get_data_transformation(self):
        try:
            numerical_columns = ['TRANSACTION_ID','CUSTOMER_ID','TERMINAL_ID','TX_AMOUNT','TX_TIME_SECONDS','TX_TIME_DAYS',
                                'TX_FRAUD_SCENARIO','Year','Month','Day','Hour','Minutes','Seconds']
            scaler : StandardScaler = StandardScaler()
            pca :PCA= PCA(n_components = 5)
            preprocessor : Pipeline = Pipeline(
                steps=[
                    ('scaler',scaler),
                    ('pca',pca)

            ])
            logging.info("Data transformation pipeline created successfully")
            return preprocessor
            
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
        
    def Balancing_the_data(self):
        try:
            oversample = SMOTE(random_state=42)
            return oversample
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
        
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            train_df =  self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)
            logging.info('train and test data has been resd sucessfully')

            input_train_feature_df = train_df.drop(columns=TARGET_COLUMN,axis=1)
            target_train_feature_df  = train_df[TARGET_COLUMN]

            input_test_feature_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_test_feature_df = test_df[TARGET_COLUMN]

            preprocessor = self.get_data_transformation()
            preprocessor_obj = preprocessor.fit(input_train_feature_df)
            transfored_input_train_feature = preprocessor_obj.transform(input_train_feature_df)
            transfored_input_test_feature = preprocessor_obj.transform(input_test_feature_df)
            logging.info('Pca nad std scaler has been app,lied')
            

            smote = self.Balancing_the_data()
            balanced_input_train_feature,balanced_target_train_feature = smote.fit_resample(transfored_input_train_feature,target_train_feature_df)
            logging.info('balancing has done sucessfully')

            train_arr = np.c_[balanced_input_train_feature,np.array(balanced_target_train_feature)]
            test_arr = np.c_[transfored_input_test_feature,np.array(target_test_feature_df)]

            save_numpy_array(file_path=self.data_transformation_config.train_transformed_file_path,array=train_arr)
            save_numpy_array(file_path=self.data_transformation_config.test_transformed_file_path,array=test_arr)
            save_object(file_path=self.data_transformation_config.transformed_object_file_path,obj=preprocessor_obj)
            save_object("final_model/preprocessing.pkl",preprocessor_obj)


            data_transformation_artificat = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                train_transformed_file_path= self.data_transformation_config.train_transformed_file_path,
                test_transformed_file_path = self.data_transformation_config.test_transformed_file_path
            )

            return data_transformation_artificat
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
        
    