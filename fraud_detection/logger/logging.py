import os
import sys
import logging
from datetime import datetime

LOG_FILE_NAME = f"{datetime.now().strftime('%m_%d_%Y_%M_%S')}.logs"
logs_path = os.path.join(os.getcwd(),'logs',LOG_FILE_NAME)
os.makedirs(logs_path,exist_ok=True)
Log_file_path = os.path.join(logs_path,LOG_FILE_NAME)


logging.basicConfig(
    filename=Log_file_path,
    format='[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s',
    level=logging.INFO
)