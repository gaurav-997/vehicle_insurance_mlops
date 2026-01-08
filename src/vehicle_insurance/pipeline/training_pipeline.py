import sys
from src.vehicle_insurance.logger import logging
from src.vehicle_insurance.exception import MyException

from src.vehicle_insurance.components.data_ingestion import DataIngestion
from src.vehicle_insurance.components.data_validation import DataValidation
from src.vehicle_insurance.components.data_transformation import DataTransformation

from src.vehicle_insurance.entity.config_entity import DataIngestionConfig,DataValidationConfig,DataTranformationConfig

from src.vehicle_insurance.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact

class TrainingPipeline:
    def __init__(self):
        self.data_ingestion_config = DataIngestionConfig()
        self.data_validation_config = DataValidationConfig()
        self.data_tranformation_config = DataTranformationConfig()
        
    def start_data_ingestion(self) ->DataIngestionArtifact:
        try:
            logging.info("starting data ingestion pipeline")
            data_ingestion = DataIngestion(data_ingestion_config=self.data_ingestion_config) # created object
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def start_data_validation(self,data_ingestion_artifact:DataIngestionArtifact) -> DataValidationArtifact:
        try:
            logging.info("starting data validation")
            data_validation = DataValidation(data_ingestion_artifact=data_ingestion_artifact,data_validation_config= self.data_validation_config)
            data_validation_artifact = data_validation.initiate_data_validation()
            return data_validation_artifact
        except Exception as e:
            raise MyException(e,sys)
        
    def start_data_transformation(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_artifact:DataValidationArtifact) -> DataTransformationArtifact:
        try:
            logging.info("starting data transformation ")
            data_transformation = DataTransformation(data_ingestion_artifact=data_ingestion_artifact,data_transformation_config=self.data_transformation_config,
                                                     data_validation_artifact=data_validation_artifact)
            data_transformation_artifact = data_transformation.initiate_data_transformation()
            return data_transformation_artifact
        except Exception as e:
            raise MyException(e,sys)

if __name__ == "__main__":
    try:
        training_pipeline = TrainingPipeline()
        data_ingestion_artifact = training_pipeline.start_data_ingestion()
        data_validation_artifact = training_pipeline.start_data_validation(data_ingestion_artifact)
        data_transformation_artifact = training_pipeline.start_data_transformation(data_ingestion_artifact, data_validation_artifact)
    except Exception as e:
        print(f"Error: {e}")
        raise e