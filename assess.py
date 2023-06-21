import argparse
import os
import ast
import torch

from config import parse_config
from model import Model

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
    if not os.path.isfile(os.path.join(config.experiment_dir, "metrics.log")):
        raise Exception("Model's metrics.log not found")

    train_loss = dict()
    val_loss = dict()
    with open(os.path.join(config.experiment_dir, "metrics.log"), "r") as f:
        for line in f:
            metric = ast.literal_eval(line)
            metric: dict
            if "train_loss" in metric.keys():
                train_loss[metric["epoch"]] = metric["train_loss"]
            if "val_loss" in metric.keys():
                val_loss[metric["epoch"]] = metric["val_loss"]

    plt.plot(
        sorted(train_loss.keys()),
        [train_loss[id] for id in sorted(train_loss.keys())],
        label="Train Loss",
    )
    plt.plot(
        sorted(val_loss.keys()),
        [val_loss[id] for id in sorted(val_loss.keys())],
        label="Validation Loss",
    )
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend(loc="best")
    plt.savefig(os.path.join(config.experiment_dir, "result.pdf"))
    # plt.show()
