from fraud_detection.exception.exception import fraud_detection_exception
from fraud_detection.logger.logging import logging
from fraud_detection.components.data_ingestion import DataIngestion
from fraud_detection.components.data_validation import DataValidation
from fraud_detection.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig
import sys

if __name__ == "__main__":
    try:
        trainingpipelineconfig = TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        dataingestion = DataIngestion(dataingestionconfig)
        logging.info("Starting data ingestion process")
        dataingestionartifacts = dataingestion.initiate_data_ingestion()
        logging.info("Data ingestion process completed successfully")
        print(dataingestionartifacts)


        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation = DataValidation(data_validation_config=  data_validation_config,data_ingestion_artifact=dataingestionartifacts)
        logging.info("Starting data validation process")
        data_validation_artifact = data_validation.initiate_data_validation()
        logging.info("Data validation process completed successfully")
        print(data_validation_artifact)

    except Exception as e:
        raise fraud_detection_exception(e,sys) from e