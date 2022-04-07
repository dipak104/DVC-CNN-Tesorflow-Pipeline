import argparse
import os
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directory
#from src.utils.callbacks import create_and_save_tensorboard_callbacks, create_and_save_checkpoint_callbacks
import random
import time

STAGE = "PREPARE CALLBACKS" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def prepare_callbacks(config_path):
    ## read config files
    config = read_yaml(config_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    tensorboard_log_dir = os.path.join(artifacts_dir, artifacts["TENSORBOARD_ROOT_LOG_DIR"])

    checkpoint_dir = os.path.join(artifacts_dir, artifacts["CHECKPOINT_DIR"])

    callbacks_dir = os.path.join(artifacts_dir, artifacts["CALLBACKS_DIR"])

    create_directory([tensorboard_log_dir, checkpoint_dir, callbacks_dir])

    #create_and_save_tensorboard_callbacks(callbacks_dir, tensorboard_log_dir)
    #create_and_save_checkpoint_callbacks(callbacks_dir)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        prepare_callbacks(config_path=parsed_args.config)
        logging.info(f">>>>> stage {STAGE} completed!<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e