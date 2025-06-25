import pandas as pd
import numpy as np
from fraud_detection.exception.exception import fraud_detection_exception
from fraud_detection.logger.logging import logging
import sys

class CoustomData:
    def __init__(self,
                 transaction_id :int,
                 coustomer_id :int,
                 terminal_id: int,
                 tx_amount :float,
                 tx_time_seconds:int,
                 tx_time_days : int,
                 tx_fraud_scenario : int,
                 year : int,
                 month: int,
                 day: int,
                 hour: int,
                 minutes : int,
                 seconds : int):
        try:
            self.transaction_id = transaction_id
            self.coustomer_id = coustomer_id
            self.terminal_id = terminal_id
            self.tx_amount = tx_amount
            self.tx_time_seconds = tx_time_seconds
            self.tx_time_days = tx_time_days
            self.tx_fraud_scenario = tx_fraud_scenario
            self.year = year
            self.month = month
            self.day = day
            self.hour = hour
            self.minutes = minutes
            self.seconds = seconds
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e
    def get_data_as_dataframe(self) -> dict:
        try:
            custom_data_input_dict = {
                
                "TRANSACTION_ID" : [self.transaction_id],
                "CUSTOMER_ID" : [self.coustomer_id],
                "TERMINAL_ID" : [self.terminal_id],
                "TX_AMOUNT" : [self.tx_amount],
                "TX_TIME_SECONDS" : [self.tx_time_seconds],
                "TX_TIME_DAYS" : [self.tx_time_days],
                "TX_FRAUD_SCENARIO" : [self.tx_fraud_scenario],
                "Year" : [self.year],
                "Month" : [self.month],
                "Day" : [self.day],
                "Hour" : [self.hour],
                "Minutes" : [self.minutes],
                "Seconds" : [self.seconds]
            }
            return pd.DataFrame(custom_data_input_dict)
        except Exception as e:
            raise fraud_detection_exception(e,sys) from e