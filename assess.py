import argparse
import os
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from config import parse_config
from model import Model, TrainerProgressBar

import matplotlib.pyplot as plt


# Dirty hack for autocomplete
class Config:
    experiment_dir: str
    model: Model
    model_path: str

    # Dataset
    dataset: torch.utils.data.Dataset
    dataset_dir: str

    input_lang_name: str
    output_lang_name: str
    max_length: int
    training_set_ratio: float

    # Train
    max_epochs: int
    validation_interval: int
    learning_rate: float

    batch_size: int
    hidden_size: int
    dropout_prob: float
    teacher_forcing_ratio: float

    enable_checkpoint: bool
    checkpoint_path: str
    checkpoint_cnt: int


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Evaluate model's performance.")
    # fmt: off
    parser.add_argument("config", type=str, 
                        help="Experiment config path")
    # fmt: on
    config = parse_config(parser.parse_args().config)
    return config


if __name__ == "__main__":
    config = parse_args()
    if not os.path.isfile(config.model_path):
        raise Exception("Model not found")
    model = config.model.load_from_checkpoint(config.model_path)
    model.eval()

    x = list(range(1, config.max_epochs + 1))
    train_loss = model.history["train_loss"]
    val_loss = model.history["val_loss"]

    plt.plot(
        range(1, len(train_loss) + 1),
        train_loss,
        label="Train Loss",
    )
    plt.plot(
        [x * config.validation_interval for x in range(len(val_loss))],
        val_loss,
        label="Validation Loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(os.path.join(config.experiment_dir, "result.pdf"))
    # plt.show()
