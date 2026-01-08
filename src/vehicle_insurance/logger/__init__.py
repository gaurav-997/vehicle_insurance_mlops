import logging
import os
from datetime import datetime
from from_root import from_root
from logging.handlers import RotatingFileHandler

LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log"
BACKUP_COUNT = 3   # we will keep only 3 log files 
LOG_FILE_SIZE = 5*1024*1024  # max size of one log file 

log_dir_path = os.path.join(from_root(),LOG_DIR)
os.makedirs(log_dir_path,exist_ok=True)
log_file_path = os.path.join(log_dir_path,LOG_FILE)

def configure_logs() -> logging.Logger:
    logger = logging.getLogger('vehicle_insurance_logger')
    logger.setLevel(logging.DEBUG)
    
    formatter = logging.Formatter(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    
    file_handler = RotatingFileHandler(filename=log_file_path,maxBytes=LOG_FILE_SIZE,backupCount=BACKUP_COUNT)
    file_handler.setLevel(logging.DEBUG)
    
    
    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)
    
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
configure_logs()

