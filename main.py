from fraud_detection.exception.exception import fraud_detection_exception
from fraud_detection.logger.logging import logging
from fraud_detection.components.data_ingestion import DataIngestion
from fraud_detection.components.data_validation import DataValidation
from fraud_detection.components.data_transformation import DataTansformation
from fraud_detection.components.model_trainer import ModelTrainer
from fraud_detection.entity.config_entity import DataIngestionConfig,TrainingPipelineConfig,DataValidationConfig,DataTransformationConfig,ModelTrainerConfig
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

        logging.info("Starting data transformation process")
        data_transformation_config = DataTransformationConfig(trainingpipelineconfig)
        data_transformation = DataTansformation(data_validation_artifact=data_validation_artifact, data_transformation_config=data_transformation_config)
        data_transformation_artifact = data_transformation.initiate_data_transformation()
        logging.info("Data transformation process completed successfully")


        logging.info("starting model trainng")
        model_trainer_config = ModelTrainerConfig(trainingpipelineconfig)
        model_trainer = ModelTrainer(data_transformation_artifact,model_trainer_config)
        model_trainer_artifact = model_trainer.initiate_model_training()
        logging.info("Model Trainer process has completed")

    except Exception as e:
        raise fraud_detection_exception(e,sys) from e