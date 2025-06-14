import yaml
import dill
from fraud_detection.exception.exception import fraud_detection_exception
from fraud_detection.logger.logging import logging
import sys
import os
import pickle
import numpy as np
import pandas as pd



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
    


