import os 
from src.vehicle_insurance.constants import *
from dataclasses import dataclass
from datetime import datetime

TIMESTAMP:str = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")

@dataclass
class TrainingPipelineConfig:
    pipeline_name:str = PIPELINE_NAME
    timestamp:str = TIMESTAMP
    artifact_dir:str = os.path.join(ARTIFACT_DIR,TIMESTAMP)

training_pipeline_config = TrainingPipelineConfig()

# artifacts/
# └── data_ingestion/
#     ├── feature_store/
#     │   └── vehicle_insurance.csv
#     │
#     └── ingested/
#         ├── train.csv
#         └── test.csv

@dataclass
class DataIngestionConfig:
    data_ingestion_dir:str = os.path.join(training_pipeline_config.artifact_dir,DATA_INGESTION_DIR_NAME)
    feature_store_file_path:str = os.path.join(data_ingestion_dir,DATA_INGESTION_FEATURE_STORE_DIR,FILE_NAME)
    training_file_path:str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TRAIN_FILE_NAME)
    testing_file_path:str = os.path.join(data_ingestion_dir,DATA_INGESTION_INGESTED_DIR,TEST_FILE_NAME)
    train_test_split_ratio:str = DATA_INGESTION_TRAIN_TEST_SPLIT_RATIO
    collection_name:str = DATA_INGESTION_COLLECTION_NAME
    
"""
artifacts/
├── data_ingestion/
├── data_validation/
├── data_transformation/
├── model_trainer/
└── model_evaluation/
"""

@dataclass
class DataValidationConfig:
    data_validation_dir:str = os.path.join(training_pipeline_config.artifact_dir,DATA_VALIDATION_DIR_NAME)
    data_validation_report_file:str = os.path.join(data_validation_dir,DATA_VALIDATION_REPORT_FILE_NAME)
   

@dataclass
class DataTranformationConfig:
    data_transformation_dir:str = os.path.join(training_pipeline_config.artifact_dir,DATA_TRANSFORMATION_DIR_NAME)
    transformed_train_file_path:str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TRAIN_FILE_NAME.replace('csv','npy'))
    transformed_test_file_path:str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_DATA_DIR,TEST_FILE_NAME.replace('csv','npy'))
    transformed_object_file_path:str = os.path.join(data_transformation_dir,DATA_TRANSFORMATION_TRANSFORMED_OBJECT_DIR,PREPROCSSING_OBJECT_FILE_NAME)
    # PREPROCSSING_OBJECT_FILE_NAME =  preprocessing.pkl