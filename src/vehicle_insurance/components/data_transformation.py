import sys
import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.compose import ColumnTransformer
from pandas import DataFrame

from src.vehicle_insurance.constants import TARGET_COLUMN, SCHEMA_FILE_PATH, CURRENT_YEAR
from src.vehicle_insurance.entity.artifact_entity import DataIngestionArtifact,DataValidationArtifact,DataTransformationArtifact
from src.vehicle_insurance.entity.config_entity import DataTranformationConfig
from src.vehicle_insurance.logger import logging
from src.vehicle_insurance.exception import MyException
from src.vehicle_insurance.utils.main_utils import read_yaml,save_numpy_array_data,save_object

class DataTransformation:
    def __init__(self,data_ingestion_artifact:DataIngestionArtifact,data_validation_artifact:DataValidationArtifact,data_transformation_config:DataTranformationConfig):
        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self._schema_config = read_yaml(file_path=SCHEMA_FILE_PATH)
        except Exception as e:
            raise MyException(e,sys)
     
    # for reading trian and test data files form data ingestion ( ingestion has data , validation has report only)   
    @staticmethod
    def read_data(file_path) -> DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e,sys)
     
    #  Defining custom transformers 
    #  Handle missing values , removes duplicates , NAN values , Normalize values , encode categorical values , smoteenn the imbalance dataset   # 
    #  initilize transformers >> load columns from schema.yaml >> create processor ( it will process columns with the help of transformers )
    # >> wrap pre-processor in a pipeline
    
    """ check validation status from validation artifact >> load train and test data from ingestion artifact >>
    divide data into feature coluns and target columns >> apply some basic transformation on splited data like rename column , drop column e.t.c.
    >> call the transformers and fit to input features of test and train >> now to handle imbalance data apply smootning on input feaure and target feature 
    >> we have now smoot input feature and target feature >> now concatinate both to create train and test array  >. save it and return data trasformation artifact 
    
    validation → load train & test → split input/target → custom transforms → sklearn transformer → handle imbalance → concat → save artifacts
    """
    #  defining sklean transformers 
    def get_data_transformer_object(self) -> Pipeline:
        try:
            logging.info("implementing logic for what to transform in dataframe ")
            numeric_transformer = StandardScaler()
            min_max_scaler = MinMaxScaler()
            logging.info("initilized numeric transformer and min max scaler ")
            
            # loading features from schema 
            numerical_features = self._schema_config['numerical_columns']
            mm_columns = self._schema_config['mm_columns']
            logging.info("Columns loaded from schema.")
            
            #  corect format of transformer (name, transformer, columns)
            processor = ColumnTransformer(transformers=[("StandardScaler",numeric_transformer,numerical_features),("MinMaxScaler",min_max_scaler,mm_columns)],remainder="passthrough")
            #  remainder="passthrough" means leave other columns as it is , we are only touching numeric columns and mn_columns 
            
            final_pipeline = Pipeline(steps=[("processor", processor)])
            logging.info("Final Pipeline Ready!!")
            return final_pipeline
                                                                                                            
        except Exception as e:
            raise MyException(e,sys)
     
    # """Map Gender column to 0 for Female and 1 for Male."""   
    def _map_gender(self,df):
        try:
            df['Gender'] = df['Gender'].map({'female':0,'male':1}).astype(int)
            return df
        except Exception as e:
            raise MyException(e,sys)
        
        
    def _create_dummy_columns(self,df):
        try:
            df = pd.get_dummies(df,drop_first=True)
            return df
        except Exception as e:
            raise MyException(e,sys)
    
    def _rename_columns(self, df):
        """Rename specific columns and ensure integer types for dummy columns."""
        logging.info("Renaming specific columns and casting to int")
        df = df.rename(columns={
            "Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year",
            "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"
        })
        for col in ["Vehicle_Age_lt_1_Year", "Vehicle_Age_gt_2_Years", "Vehicle_Damage_Yes"]:
            if col in df.columns:
                df[col] = df[col].astype('int')
        return df
    
    def _drop_columns(self,df):
        try:
            df = df.drop(columns=self._schema_config['drop_columns'],errors="ignore")
            return df   
        except Exception as e:
            raise MyException(e,sys)
        
    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation ")
            # 1.  first check validation status is true 
            if not self.data_validation_artifact.status:
                raise Exception(self.data_validation_artifact.message)
            
            #  2. if validation status = success then load data from ingestion aritfacts ( train and test data )
            logging.info("loading data for trasnformation ")
            train_df = self.read_data(file_path=self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(file_path=self.data_ingestion_artifact.test_file_path)
            
            # 3. now split infut feature and target features in above train and test data 
            logging.info("defining input and target columns for train and test df ")
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN]
            
            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN],axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN]
            
            # 4.  apply above custom transformation on input features of train and test df 
            logging.info("applying custom transformation on input features ")
            input_feature_train_df = self._map_gender(df=input_feature_train_df)
            input_feature_train_df = self._create_dummy_columns(df=input_feature_train_df)
            input_feature_train_df = self._drop_columns(df=input_feature_train_df)
            input_feature_train_df = self._rename_columns(df=input_feature_train_df)
            
            input_feature_test_df = self._map_gender(df=input_feature_test_df)
            input_feature_test_df = self._create_dummy_columns(df=input_feature_test_df)
            input_feature_test_df = self._drop_columns(df=input_feature_test_df)
            input_feature_test_df = self._rename_columns(df=input_feature_test_df)
            
            # 5. till data is ready so get sklearn transformer on cusotm transformerd data and start transforming
            logging.info("doing end to end transformation on input features of train and test data")
            pre_processor = self.get_data_transformer_object()
            input_feature_train_array = pre_processor.fit_transform(input_feature_train_df)
            input_feature_test_array = pre_processor.transform(input_feature_test_df)
            
            #6.  Applying SMOTEENN on transformed input feature and target feature data for handling imbalanced dataset & geting final input features and target features 
            logging.info("Applying SMOTEENN ")
            smt = SMOTEENN(sampling_strategy="minority")
            input_feature_train_final,target_feature_train_final = smt.fit_resample(input_feature_train_array,target_feature_train_df)
            # input_feature_test_final,target_feature_test_final = smt.fit_resample(input_feature_test_array,target_feature_test_df)
            # TEST DATA → NO SMOTE ( no smoothning to test data )
            input_feature_test_final = input_feature_test_array
            target_feature_test_final = target_feature_test_df
            
            #7.  concatinate both feature and target df 
            train_arr = np.c_(input_feature_train_final,np.array(target_feature_train_final))
            test_arr = np.c_(input_feature_test_final,np.array(target_feature_test_final))
            
            #  8. saving data 
            save_object(self.data_transformation_config.transformed_object_file_path,pre_processor )
            save_numpy_array_data(self.data_transformation_config.transformed_train_file_path, array=train_arr)
            save_numpy_array_data(self.data_transformation_config.transformed_test_file_path, array=test_arr)
            logging.info("Saving transformation object and transformed files. and data transformation completed successfully")
            
            return DataTransformationArtifact(transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                                              transformed_train_file_path=self.data_transformation_config.transformed_train_file_path ,
                                              transformed_test_file_path=self.data_transformation_config.transformed_test_file_path)
              
        except Exception as e:
            raise MyException(e,sys)
        
        
    