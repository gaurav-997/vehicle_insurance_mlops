import os
import sys
import json
import pandas as pd
from pandas import DataFrame

from src.vehicle_insurance.entity.config_entity import DataValidationConfig
from src.vehicle_insurance.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact
from src.vehicle_insurance.logger import logging
from src.vehicle_insurance.exception import MyException
from src.vehicle_insurance.utils.main_utils import read_yaml
from src.vehicle_insurance.constants import SCHEMA_FILE_PATH

class DataValidation:
    def __init__(self,data_ingestion_artifact: DataIngestionArtifact,data_validation_config: DataValidationConfig):
        try:
            logging.info("starting data validation ")
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
        
    # for reading data ingestion artifact files  
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
        
    # here we will validate number of columns , is column names correct 
    def validate_number_of_columns(self,df: DataFrame) -> bool:
        try:
            logging.info("comparing length of coluns in dataframe and schema.yaml")
            status = False
            if len(df.columns) == len(self._schema_config['columns']):
                status = True
            return status
        except Exception as e:
            raise MyException(e,sys)
        
    def is_column_names_correct(self,df:DataFrame) -> bool:
        try:
            dataframe_columns = df.columns
            missing_numerical_columns = []
            missing_categorical_columns = []
            
            for columns in self._schema_config['numerical_columns']:
                if columns not in dataframe_columns:
                    missing_numerical_columns.append(columns)
            
            for columns in self._schema_config['categorical_columns']:
                if columns not in dataframe_columns:
                    missing_categorical_columns.append(columns)
            
            return False if len(missing_numerical_columns) >0 or len(missing_categorical_columns) >0 else True
                    
        except Exception as e:
            raise MyException(e,sys)
      
    
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("starting data validation")
            validation_error_msg = ""
            train_df, test_df = (DataValidation.read_data(file_path=self.data_ingestion_artifact.trained_file_path), 
                                 DataValidation.read_data(file_path=self.data_ingestion_artifact.test_file_path))
            
            status = self.validate_number_of_columns(train_df)
            if status == False:
                validation_error_msg +="columns are missing in trianing data frame"
                
            
            status = self.validate_number_of_columns(test_df)
            if status == False:
                validation_error_msg +="columns are missing in test data"
                
                
            status = self.is_column_names_correct(df=train_df)
            if status == False:
                validation_error_msg +="column names are not correct in training data"
                
                
            status = self.is_column_names_correct(df=test_df)
            if status == False:
                validation_error_msg +="column names are not correct in test data"
                
            validation_status = len(validation_error_msg) == 0
                
                
            data_validation_artifact = DataValidationArtifact(message=validation_error_msg,status=validation_status,
                                                              validation_report_file_path= self.data_validation_config.data_validation_report_file)
            
            
            # check validation artifact file present or not 
            validation_report_file_path = os.path.dirname(self.data_validation_config.data_validation_report_file)
            os.makedirs(validation_report_file_path,exist_ok=True)
            
            # Save validation status and message to a JSON file
            validation_report = {
                "validation_status": validation_status,
                "message": validation_error_msg.strip()
            }

            with open(self.data_validation_config.data_validation_report_file, "w") as report_file:
                json.dump(validation_report, report_file, indent=4)
                
            return data_validation_artifact
            
            
        except Exception as e:
            raise MyException(e,sys)

