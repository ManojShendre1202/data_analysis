import os
from pathlib import Path
import logging

#logging string
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

project_name = 'data_analysis'

list_of_files = [
    ".github/workflows/.gitkeep",
    f"src/{project_name}/__init__.py",
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_prepration.py",
    f"src/{project_name}/components/model_training.py",
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/helper.py",
    f"src/{project_name}/utils/logger.py",
    f"src/{project_name}/utils/exception.py",
    f"src/{project_name}/pipeline/__init__.py",
    "data/raw",
    "data/processed",
    "requirements.txt",
    "setup.py",
    "research/EDA.ipynb",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir !="":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory; {filedir} for the file: {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"{filename} is already exists")