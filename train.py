import argparse
import os
from typing import List
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
import torch

from config import parse_config
from model import Model, TrainerProgressBar, ParamLogger

import warnings

warnings.filterwarnings("ignore", ".*does not have many workers.*")
warnings.filterwarnings("ignore", ".*Checkpoint directory .* exists and is not empty.*")

torch.set_float32_matmul_precision("medium")


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

    enable_input_prefix_filter: bool
    enable_output_prefix_filter: bool

    input_prefix_filter: List[str]
    output_prefix_filter: List[str]

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
    parser = argparse.ArgumentParser(description="Training a model with config.")
    # fmt: off
    parser.add_argument("config", type=str, 
                        help="Experiment config path")
    parser.add_argument('-f', '--force', action='store_true',
                        help="Override existing model.")
    # fmt: on
    args = parser.parse_args()
    config = parse_config(args.config, override=args.force)
    return config


if __name__ == "__main__":
    config = parse_args()

    if os.path.isfile(config.model_path):
        raise Exception(
            "Model already exist. If you want to override, please use -f/--force."
        )

    dataset = config.dataset(**vars(config))

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [
            int(config.training_set_ratio * len(dataset)),
            len(dataset) - int(config.training_set_ratio * len(dataset)),
        ],
    )

    train_dataloader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=config.batch_size, shuffle=True
    )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset, batch_size=config.batch_size, shuffle=False
    )

    callbacks = [TrainerProgressBar()]
    other_kwargs = dict()
    if config.enable_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            every_n_epochs=config.max_epochs // config.checkpoint_cnt,
            save_top_k=-1,
            dirpath=config.checkpoint_path,
            save_last=True,
        )
        callbacks.append(checkpoint_callback)
        other_kwargs["ckpt_path"] = "last"

    logger = ParamLogger(config.experiment_dir)

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        logger=logger,
        enable_checkpointing=config.enable_checkpoint,
        check_val_every_n_epoch=config.validation_interval,
        callbacks=callbacks,
        default_root_dir=config.experiment_dir,
    )

    model = config.model(
        input_lang=dataset.input_lang,
        output_lang=dataset.output_lang,
        **vars(config),
    )
    trainer.fit(
        model,
        train_dataloader,
        val_dataloader,
        **other_kwargs,
    )
    if trainer.interrupted:
        print("Trainer was interrupted!")
        exit(1)

    trainer.save_checkpoint(config.model_path)
