import os
import sys
import numpy as np  
from typing import Tuple

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,f1_score,recall_score,precision_score

from src.vehicle_insurance.logger import logging
from src.vehicle_insurance.exception import MyException
# from src.vehicle_insurance.components.data_transformation import DataTransformationArtifact
from src.vehicle_insurance.entity.artifact_entity import DataTransformationArtifact ,ModelTrainerArtifact,ClassificationMetricArtifact
from src.vehicle_insurance.entity.config_entity import ModelTrainingConfig
from src.vehicle_insurance.utils.main_utils import  load_numpy_array_data, load_object, save_object
from src.vehicle_insurance.entity.estimator import mymodel

class ModelTrainer:
    def __init__(self,data_transformation_artifact:DataTransformationArtifact,model_trainer_config:ModelTrainingConfig) -> ModelTrainerArtifact:
        self.data_transformation_artifact = data_transformation_artifact
        self.model_trainer_config = model_trainer_config
        
    def get_model_object_and_report(self,train:np.array,test:np.array) -> Tuple[object,object]:
        try:
            logging.info("Training RandomForestClassifier with specified parameters")
            
            # Splitting the train and test data into features and target variables 
            x_train,y_train,x_test,y_test = train[:,:-1],train[:,-1],test[:,:-1],test[:,-1]
            
            
            model = RandomForestClassifier(
                n_estimators= self.model_trainer_config._n_estimators,
                min_samples_split= self.model_trainer_config._min_samples_split,
                min_samples_leaf= self.model_trainer_config._min_samples_leaf,
                max_depth= self.model_trainer_config._max_depth,
                random_state= self.model_trainer_config._random_state,
                criterion= self.model_trainer_config._criterion
            )
            
            # fit the model ( i.e train the model ) >> predict on it >> calculate performance metrics
            model.fit(x_train,x_test)
            logging.info("model training completed ")
            y_pred = model.predict(y_train)
            logging.info("prediction is completed ")
            accuracy = accuracy_score(y_test,y_pred)
            f1  = f1_score(y_test,y_pred)
            precision = precision_score(y_test,y_pred)
            recall = recall_score(y_test,y_pred)
            logging.info("metrics calculation si done ")
            
            metrics_artifacts = ClassificationMetricArtifact(f1_score=f1,precision_score=precision,recall_score=recall)
            return model,metrics_artifacts
        
        
            
            
            
            
        except Exception as e:
            raise MyException(e,sys)
