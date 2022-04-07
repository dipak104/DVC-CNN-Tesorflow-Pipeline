import argparse
import os
from tqdm import tqdm
import logging
from src.utils.common import read_yaml, create_directories
from src.utils.model import get_VGG16_model, prepare_base_model


STAGE = "PREPARE BASE MODEL" ## <<< change stage name 

logging.basicConfig(
    filename=os.path.join("logs", 'running_logs.log'), 
    level=logging.INFO, 
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
    )


def prepare_base_model(config_path: str, params_path: str) -> None:
    config = read_yaml(config_path)
    params = read_yaml(config_path)

    artifacts = config["artifacts"]
    artifacts_dir = artifacts["ARTIFACTS_DIR"]

    base_model_dir = artifacts["BASE_MODEL_DIR"]
    base_model_name = artifacts["BASE_MODEL_NAME"]

    base_model_dir_path = os.path.join(artifacts_dir, base_model_dir)
    create_directories([base_model_dir_path])

    base_model_path = o.path.join(base_model_dir_path, base_model_name)

    base_model = get_VGG16_model()

    full_model = prepare_full_model(base_model)

    updated_base_model_path  = os.path.join(base_model_dir_path, artifaccts["UPDATED_BASE_MODEL_NAME"])

    full_model.save(updated_base_model_path)



if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--config", "-c", default="configs/config.yaml")
    args.add_argument("--params", "-p", default="params.yaml")
    parsed_args = args.parse_args()

    try:
        logging.info("\n********************")
        logging.info(f">>>>> stage {STAGE} started <<<<<")
        prepare_base_model(config_path=parsed_args.config, params_path=parsed_args.params)
        logging.info(f">>>>> stage {STAGE} Completed! Base model saved<<<<<\n")
    except Exception as e:
        logging.exception(e)
        raise e