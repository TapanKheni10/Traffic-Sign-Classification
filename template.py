import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

project_name = "TrafficSignRecognition"

## These is a list of the files that will be created
list_of_files = [
    "main.py",
    "requirements.txt",
    "setup.py",
    "models",
    "research",
    "research/data_link.txt",
    f"src/{project_name}",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/prediction_component",
    f"src/{project_name}/prediction_component/__init__.py",
    f"src/{project_name}/prediction_component/predict.py",
    f"src/{project_name}/model_component",
    f"src/{project_name}/model_component/__init__.py",
    f"src/{project_name}/model_component/model.py",
]

for file_path in list_of_files:
    if "." in file_path:
        Path(file_path).touch()
        logging.info(f"Created file at {file_path}")
    else:
        os.makedirs(file_path, exist_ok=True)
    logging.info(f"Created directory at {file_path}")