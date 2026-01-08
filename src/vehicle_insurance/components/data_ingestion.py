import os
import sys
from pandas import DataFrame
from sklearn.model_selection import train_test_split

from src.vehicle_insurance.entity.config_entity import DataIngestionConfig 
from src.vehicle_insurance.entity.artifact_entity import DataIngestionArtifact
from src.vehicle_insurance.logger import logging
from src.vehicle_insurance.exception import MyException
from src.vehicle_insurance.constants import *

from src.vehicle_insurance.data_access.proj1_data import Proj1Data

class DataIngestion:
    def  __init__(self,data_ingestion_config = DataIngestionConfig()):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e,sys)
        
    
    # load the data from MongoDB 
    
    def load_data_into_feature_store(self) ->DataFrame:
        try:
            # load data >> create file >> save data to file 
            logging.info("starting data ingestion")
            my_data = Proj1Data()
            df = my_data.export_collection_as_dataframe(collection_name=self.data_ingestion_config.collection_name)
            
            feature_store_file_path = self.data_ingestion_config.feature_store_file_path
            feature_store_dir = os.path.dirname(feature_store_file_path)
            os.makedirs(feature_store_dir,exist_ok=True)
            
            logging.info(f"saving data to feature store in {FILE_NAME} ")
            df.to_csv(path_or_buf=feature_store_file_path,index=False,header=True)
            logging.info("data ingestion completed")
            return df
            
        except Exception as e:
            raise MyException(e,sys)
      
    # split the loaded data and save it 
      
    def split_data_into_train_test(self,df:DataFrame) ->None:
        # split data frame >> create dirctory >> save data
        try:
            logging.info("spliting data into train test ")
            train_set,test_set = train_test_split(df,test_size=self.data_ingestion_config.train_test_split_ratio)
            
            logging.info("creating files")
            dir_path = os.path.dirname(self.data_ingestion_config.training_file_path)
            os.makedirs(dir_path,exist_ok=True)
            
            logging.info("saving test and train files into csv format")
            train_set.to_csv(self.data_ingestion_config.training_file_path,index = False,header = True)
            test_set.to_csv(self.data_ingestion_config.testing_file_path,index = False,header = True)
            
            logging.info("train test data saved successfully ")
            
        except Exception as e:
            raise MyException(e,sys)
        
    # work on artifact part , here we call above functons load data >> split it >> save artifacts
    
    def initiate_data_ingestion(self) ->DataIngestionArtifact:
        try:
            logging.info("starting data ingestion and it will generate artifacts")
            df = self.load_data_into_feature_store()
            self.split_data_into_train_test(df)
            
            data_ingestion_artifact = DataIngestionArtifact(trained_file_path=self.data_ingestion_config.training_file_path , test_file_path= self.data_ingestion_config.testing_file_path)
            return data_ingestion_artifact
        except Exception as e:
            raise MyException(e,sys)
        
        

