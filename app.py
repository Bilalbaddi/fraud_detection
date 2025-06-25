from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from fraud_detection.utils.ml_utils.model.estimator import frudmodel
from fraud_detection.pipeline.predictionpipeline import CoustomData
from fraud_detection.utils.main_utils.utils import load_object
from fraud_detection.exception.exception import fraud_detection_exception
import sys


application = Flask(__name__)
app = application

@app.route('/')
def Index():
    return render_template('index.html')

@app.route('/predictdata', methods=['GET','POST'])
def predict_datapoint():
    if request.method  == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CoustomData(
                transaction_id= int(request.form.get('TRANSACTION_ID')),
                coustomer_id= int(request.form.get('CUSTOMER_ID')),
                terminal_id= int(request.form.get('TERMINAL_ID')),
                tx_amount= float(request.form.get('TX_AMOUNT')),
                tx_time_seconds= int(request.form.get('TX_TIME_SECONDS')),
                tx_time_days = int(request.form.get('TX_TIME_DAYS')),
                tx_fraud_scenario=  int(request.form.get('TX_FRAUD_SCENARIO')),
                year=  int(request.form.get('Year')),
                month=  int(request.form.get('Month')),
                day= int(request.form.get('Day')),
                hour= int(request.form.get('Hour')),
                minutes= int(request.form.get('Minutes')),
                seconds= int(request.form.get('Seconds')),
            )
            df = data.get_data_as_dataframe()
            print(df)
            preprocessor = load_object('final_model/preprocessing.pkl')
            final_model = load_object('final_model/model.pkl')

            model = frudmodel(preprocessor=preprocessor,model=final_model)
            results = model.predict(df)
            print(results)
            
            prediction_value = results[0] if len(results) > 0 else None
            
            # Format the result to two decimal places if it's a float
            formatted_result = round(prediction_value, 2) if isinstance(prediction_value, float) else prediction_value
            print("Formatted Prediction result:", formatted_result)  # Debugging output
            return render_template('home.html', results=formatted_result)
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e

       
if __name__ == "__main__":
    app.run(host='0.0.0.0',debug=True)