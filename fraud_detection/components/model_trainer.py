import os
import sys
import pandas as pd
import numpy as np
from fraud_detection.exception.exception import fraud_detection_exception
from fraud_detection.logger.logging import logging
from fraud_detection.entity.config_entity import ModelTrainerConfig
from fraud_detection.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact
)
from fraud_detection.utils.main_utils.utils import  save_object,load_object
from fraud_detection.utils.main_utils.utils import  load_numpy_array,evaluate_models
from fraud_detection.utils.ml_utils.metric.classification_report import get_classification_report
from fraud_detection.utils.ml_utils.model.estimator import frudmodel

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    AdaBoostClassifier
)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier


class ModelTrainer:
    def __init__(self,data_transformation_artifact: DataTransformationArtifact,
                 model_trainer_config: ModelTrainerConfig):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.model_trainer_config = model_trainer_config
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
        
    def train_model(self,x_train,y_train,x_test,y_test):
        
            models = {
            "RandomForestClassifier" : RandomForestClassifier(),
            "GradientBoostingClassifier" : GradientBoostingClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier(),
            "LogisticRegression" : LogisticRegression(),
            "DecisionTreeClassifier" : DecisionTreeClassifier(),
            "KNeighborsClassifier" : KNeighborsClassifier(),
            "SVC" : SVC()
        }
        #     params={
        #     "DecisionTreeClassifier": {
        #         'criterion':['gini', 'entropy', 'log_loss'],
        #         'splitter':['best','random'],
        #         'max_features':['sqrt','log2'],
        #     },
        #     "RandomForestClassifier":{
        #         'criterion':['gini', 'entropy', 'log_loss'],
                
        #         'max_features':['sqrt','log2',None],
        #         'n_estimators': [8,16,32,128,256]
        #     },
        #     "GradientBoostingClassifier":{
        #         'loss':['log_loss', 'exponential'],
        #         'learning_rate':[.1,.01,.05,.001],
        #         'subsample':[0.6,0.7,0.75,0.85,0.9],
        #         'criterion':['squared_error', 'friedman_mse'],
        #         'max_features':['auto','sqrt','log2'],
        #         'n_estimators': [8,16,32,64,128,256]
        #     },
        #     "LogisticRegression":{},
        #     "AdaBoostClassifier":{
        #         'learning_rate':[.1,.01,.001],
        #         'n_estimators': [8,16,32,64,128,256]
        #     },
        #     "SVC":{
        #         "kernel" : ['linear','poly','rbf','sigmoid'],
        #         "C" : [1.0,2.0,3.0,4.0]


        #     },
            
        #     "KNeighborsClassifier":{
        #         "n_neighbors" : [1,2,3,4,5,6,7,8,9],
        #         "weights" : ['uniform', 'distance'],
        #         "algorithm" : ['auto','ball_tree','kd_tree','brute']
        #         }           
        # }
            model_report : dict = evaluate_models(x_train=x_train,y_train=y_train,x_test=x_test,y_test= y_test,models=models)
            best_model_scores = max(sorted(model_report.values()))
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_scores)
            ]
            best_model = models[best_model_name]

            y_train_pred = best_model.predict(x_train)
            classification_train_matrix = get_classification_report(y_true=y_train,y_pred=y_train_pred)


            y_test_pred = best_model.predict(x_test)
            classification_test_matrix = get_classification_report(y_true=y_test,y_pred=y_test_pred)

            preprocessor = load_object(file_path=self.data_transformation_artifact.transformed_object_file_path)
            model_dir_path = os.path.dirname(self.model_trainer_config.trained_model_file_path)
            os.makedirs(model_dir_path,exist_ok=True)


            Network_model = frudmodel(preprocessor=preprocessor,model=best_model)
            save_object(self.model_trainer_config.trained_model_file_path,obj=Network_model)
            save_object('final_model/model.pkl',best_model)
            

            model_trainer_artifact = ModelTrainerArtifact(
            trained_model_file_path=self.model_trainer_config.trained_model_file_path,
            train_metric_artifact=classification_train_matrix,
            test_metric_artifact=classification_test_matrix
            )
            logging.info(f"model trainer artifact {model_trainer_artifact}")
            return model_trainer_artifact
    
    

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            train_arr_file_path = self.data_transformation_artifact.train_transformed_file_path
            test_arr_file_path = self.data_transformation_artifact.test_transformed_file_path

            train_arr = load_numpy_array(train_arr_file_path)
            test_arr = load_numpy_array(test_arr_file_path)

            x_train,y_train,x_test,y_test=(
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1],
            )

            model_trainer_artifact = self.train_model(x_train=x_train,y_train=y_train,x_test=x_test,y_test=y_test)
            return model_trainer_artifact
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e

        


    