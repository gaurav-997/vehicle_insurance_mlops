import os
import sys 
import pymongo
import certifi
from src.vehicle_insurance.constants import  DATABASE_NAME , MONGODB_URL_KEY
from src.vehicle_insurance.logger import logging
from src.vehicle_insurance.exception import MyException

# load the certificate for connecting to mongo DB 
ca = certifi.where()

class MongoDBClient:
    client = None  # Class-level attribute for shared client
    
    def __init__(self,database_name:str = DATABASE_NAME) ->None:
        self.database_name = database_name
        try:
            # check if mongo DB client connection is already establish , if not create a new one 
            if MongoDBClient.client is None:
                mongo_db_url = os.getenv(MONGODB_URL_KEY)
                
                if mongo_db_url is None:
                    raise Exception(f"Mongo DB connection URL {MONGODB_URL_KEY} is not set ")
                
                # Establish a new MongoDB client connection
                MongoDBClient.client = pymongo.MongoClient(mongo_db_url, tlsCAFile=ca)
            
            # Use the shared MongoClient for this instance
            self.client = MongoDBClient.client
            self.database = self.client[database_name]  # Connect to the specified database
            self.database_name = database_name
            logging.info("MongoDB connection successful.")
                
        except Exception as e:
            raise MyException(e,sys)