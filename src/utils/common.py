import yaml
import logging
import os
import shutil
import time

def read_yaml(filepath: str) -> dict:
    with open(filepath) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"YAML file successfully read from  {filepath}")
    return content

def create_directory(dir_paths: list) -> None:
    for dir_path in dir_paths:
        os.makedirs(dir_path, exist_ok=True)
        logging.info(f"Created directory as : {dir_path}")

def copy_files(source_download_dir: str, local_data_dir: str) -> None:
    list_of_files = os.listdir(source_download_dir)
    N = len(list_of_files)

    for file in list_of_files:
        src = os.path.join(source_download_dir, file)
        dest = os.path.join(local_data_dir, file)

        shutil.copy(src, dest)

    logging.info(f"Copy of files Succeeded")

def get_timestamp(name: str) -> str:
    timestamp = time.asctime().replace(' ','_').replace(':','.')
    unique_name = f"{name}_at_{timestamp}"
    return unique_name