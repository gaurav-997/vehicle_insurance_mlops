import os
import numpy as np
from src.vehicle_insurance.logger import logging
from src.vehicle_insurance.exception import MyException
import sys
import yaml
from pandas import DataFrame
import dill

def read_yaml(file_path:str):
    try:
        logging.info("opening file in rb mode")
        with open(file=file_path , mode='rb') as file:
            return yaml.safe_load(file)
    except Exception as e:
        raise MyException(e,sys)
    
def write_yaml_file(file_path:str,content:object , replace:bool = False) -> None:
    try:
        logging.info("writing in a file ")
        if replace:
            if os.path.exists(file_path):
                os.removedirs(file_path)
            dir_name = os.path.dirname(file_path)
            os.makedirs(dir_name,exist_ok=True)
            
            with open(file= file_path,mode='w') as file:
                yaml.dump(content,file)
    except Exception as e:
        raise MyException(e,sys)
    
def save_numpy_array_data(file_path:str,array:np.array):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path,exist_ok=True)
        with open(file_path,'wb') as file:
            np.save(file,array)
    except Exception as e:
        raise MyException(e,sys)
    
def save_object(file_path:str,obj:object) -> None:
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(file_path,exist_ok=True)
        with open(file_path,'wb') as file:
            dill.dump(obj,file)
    except Exception as e:
        raise MyException(e,sys)
    