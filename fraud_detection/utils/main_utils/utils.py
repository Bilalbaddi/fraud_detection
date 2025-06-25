import yaml
import dill
from fraud_detection.exception.exception import fraud_detection_exception
from fraud_detection.logger.logging import logging
import sys
import os
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score



def save_csv_object(file_path:str, data) -> None:

    """
    Saves a DataFrame or list of dictionaries to a CSV file.

    Parameters:
    - data: pd.DataFrame or list[dict]
    - file_path: str, full path including filename (e.g., 'data/output/myfile.csv')
    """
    try:
        # Ensure folder exists
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Convert list of dicts to DataFrame if needed
        if isinstance(data, list):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            raise ValueError("Data must be a Pandas DataFrame or a list of dictionaries.")

        # Save as CSV
        data.to_csv(file_path, index=False)
        print(f"✅ File saved successfully at: {file_path}")

    except Exception as e:
        print(f"❌ Failed to save CSV: {e}")

def read_yaml_file(file_path : str)->dict:
    try:
        with open(file_path,'rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise fraud_detection_exception(e,sys) from e
    
    
def write_yaml_file(file_path:str,content:object,replace : bool = False)-> None:
    try:
        if replace :
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path,'w') as file_obj:
                yaml.dump(content, file_obj)
    except Exception as e:
        raise fraud_detection_exception(e, sys) from e
    
def save_numpy_array(file_path:str,array :np.ndarray) ->None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            np.save(file,array)
    except Exception as e:
        raise fraud_detection_exception(e,sys) from e

def save_object(file_path : str,obj:object) ->None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            pickle.dump(obj,file)
    except Exception as e:
        raise fraud_detection_exception(e,sys) from e
    

def load_object(file_path:str)->object:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} Not exists")
        with open(file_path,'rb') as file:
            return pickle.load(file)
    except Exception as e:
        raise fraud_detection_exception(e,sys) from e
    

def load_numpy_array(file_path: str)-> np.array:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"The file {file_path} Not exists")
        with open(file_path,'rb') as file:
            return np.load(file)
        
    except Exception as e :
        raise fraud_detection_exception(e,sys) from e
    
def evaluate_models(x_train,y_train,x_test,y_test,models):
    reports = {}
    try:
        for i in range (len(list(models))):
            model = list(models.values())[i]
            # para = params[list(models.keys())[i]]

            # rs = RandomizedSearchCV(model,para,cv=5,verbose=2,n_jobs=-1)
            # rs.fit(x_train,y_train)

            # model.set_params(**rs.best_params_)
            model.fit(x_train,y_train)

            y_train_pred = model.predict(x_train)
            y_test_pred = model.predict(x_test)

            train_model_score = r2_score(y_train,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)


            reports[list(models.keys())[i]] = test_model_score

            return reports
    except Exception as e:
        raise fraud_detection_exception(e,sys) from e