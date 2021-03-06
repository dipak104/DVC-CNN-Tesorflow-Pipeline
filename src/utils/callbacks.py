import os
import argparse
import logging
import tensorflow as tf
import joblib
from src.utils.common import get_timestamp


def create_and_save_tensorboard_callbacks(callbacks_dir: str, tensorboard_log_dir: str) -> None:
    unique_name = get_timestamp("tb_logs")
    tb_running_log_dir = os.path.join(tensorboard_log_dir, unique_name)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir = tb_running_log_dir)
    tb_callpack_path = os.path.join(callbacks_dir, "tensorboard_cb.cb")
    joblib.dump(tensorboard_callback, tb_callpack_path)
    logging.info(f"TensorBoard callback is saved at {tb_callpack_path} as binary file")


def create_and_save_checkpoint_callbacks(callbacks_dir: str, checkpoint_dir: str) -> None:
    checkpoint_file = os.path.join(checkpoint_dir, "ckpt_model.h5")
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_file,
        save_best_only=True
    )
    ckpt_callback_filepath = os.path.join(callbacks_dir, "checkpoint_cb.cb")
    joblib.dump(checkpoint_callback, ckpt_callback_filepath)

    logging.info(f"Checkpoint callback is saved at {ckpt_callback_filepath} as binary file")


def get_callbacks(callback_dir_path: str):
    callback_paths = [os.path.join(callback_dir_path, pickle_file) for pickle_file in os.listdir(callback_dir_path) if pickle_file.endswith(".cb")]
    callbacks = [
        joblib.load(path) for path in callback_paths
    ]
    logging.info(f"saved callbacks are ready to use")
    return callbacks