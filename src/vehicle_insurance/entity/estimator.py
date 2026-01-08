import pandas as pd
import sys
from pandas import DataFrame
from src.vehicle_insurance.logger import logging
from src.vehicle_insurance.exception import MyException
from sklearn.pipeline import Pipeline

class TargetValueMaping:
    def __init__(self):
        self.yes:int = 0
        self.no:int = 1
        
    def _asdict(self):
        return self.__dict__
    
    def reverse_maping(self):
        maping_response = self._asdict
        return dict(zip(maping_response.values(),maping_response.keys()))
    
    
class mymodel:
    def __init__(self,pre_processing_object:Pipeline,trained_model_object: object):
        self.pre_processing_object = pre_processing_object
        self.trained_model_object = trained_model_object
        
    def predict(self,dataframe:pd.DataFrame) -> DataFrame:
        try:
            logging.info("Starting prediction process.")

            # Step 1: Apply scaling transformations using the pre-trained preprocessing object
            transformed_feature = self.pre_processing_object.transform(dataframe)

            # Step 2: Perform prediction using the trained model
            logging.info("Using the trained model to get predictions")
            predictions = self.trained_model_object.predict(transformed_feature)

            return predictions

        except Exception as e:
            logging.error("Error occurred in predict method", exc_info=True)
            raise MyException(e, sys) from e


    def __repr__(self):
        return f"{type(self.trained_model_object).__name__}()"

    def __str__(self):
        return f"{type(self.trained_model_object).__name__}()"
        