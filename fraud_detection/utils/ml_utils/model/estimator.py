from fraud_detection.constant.training_pipeline import SAVED_MODEL_DIR, SAVED_MODEL_NAME

from fraud_detection.exception.exception import fraud_detection_exception
from fraud_detection.utils.main_utils.utils import load_object, save_object
import sys
import os


class frudmodel:
    def __init__(self,preprocessor,model):
        self.preprocessor = preprocessor
        self.model = model
    def predict(self,x):
        try:
            x_transform = self.preprocessor.transform(x)
            y_hat = self.model.predict(x_transform)
            return y_hat
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
