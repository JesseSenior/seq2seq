import configparser
import os
import shutil
import random
import pathlib
import torch
from argparse import Namespace

from dataset import DatasetManythingsAnki
from model import Seq2seq

available_dataset = {
    "manythings_anki": DatasetManythingsAnki,
}

available_model = {
    "seq2seq": Seq2seq,
}


def parse_config(
    config_path: str,
    default_config_path: str = "configs/default.conf",
    override: bool = False,
) -> Namespace:
    config = configparser.ConfigParser(allow_no_value=True)
    config.read(default_config_path)
    config.read(config_path)
    if override:
        print("Warning: The existing model will be override!")
        shutil.rmtree(config["DEFAULT"]["experiment_dir"])
    save_config_path = os.path.join(config["DEFAULT"]["experiment_dir"], "config.conf")

    if os.path.isfile(save_config_path):
        old_config = configparser.ConfigParser(allow_no_value=True)
        old_config.read(save_config_path)
        config["DEFAULT"]["random_seed"] = old_config["DEFAULT"]["random_seed"]

        save_config_path_new = os.path.join(
            config["DEFAULT"]["experiment_dir"], "new_config.conf"
        )
        with open(save_config_path_new, "w") as f:
            config.write(f)

        with open(save_config_path, "r") as file1, open(
            save_config_path_new, "r"
        ) as file2:
            content1 = file1.read()
            content2 = file2.read()
            if content1 != content2:
                raise Exception("Config Not Equal")
    else:
        config.set("DEFAULT", "random_seed", str(random.randint(1, 1000)))

    random.seed(config["DEFAULT"].getint("random_seed"))
    torch.manual_seed(config["DEFAULT"].getint("random_seed"))
    torch.cuda.manual_seed(config["DEFAULT"].getint("random_seed"))

    pathlib.Path(config["DEFAULT"]["experiment_dir"]).mkdir(parents=True, exist_ok=True)
    with open(save_config_path, "w") as f:
        config.write(f)

    ret = dict()

    ret["experiment_dir"] = config["DEFAULT"]["experiment_dir"]

    ret["model"] = available_model[config["DEFAULT"]["model_type"]]
    ret["model_path"] = config["DEFAULT"]["experiment_dir"] + "model.ckpt"

    ##### Dataset #####
    ret["dataset"] = available_dataset[config["Dataset"]["dataset_type"]]
    ret["dataset_dir"] = config["Dataset"]["dataset_dir"]
    pathlib.Path(config["Dataset"]["dataset_dir"]).mkdir(parents=True, exist_ok=True)

    ret["input_lang_name"] = config["Dataset"]["input_lang_name"]
    ret["output_lang_name"] = config["Dataset"]["output_lang_name"]
    ret["max_length"] = config["Dataset"].getint("max_length")
    ret["training_set_ratio"] = config["Dataset"].getfloat("training_set_ratio")

    ret["enable_input_prefix_filter"] = config["Dataset"].getboolean(
        "enable_input_prefix_filter"
    )
    if ret["enable_input_prefix_filter"]:
        ret["input_prefix_filter"] = [
            item.strip() for item in config["Dataset"]["input_prefix_filter"].split(",")
        ]

    ret["enable_output_prefix_filter"] = config["Dataset"].getboolean(
        "enable_output_prefix_filter"
    )
    if ret["enable_output_prefix_filter"]:
        ret["output_prefix_filter"] = [
            item.strip()
            for item in config["Dataset"]["output_prefix_filter"].split(",")
        ]

    ##### Train #####
    ret["max_epochs"] = config["Train"].getint("max_epochs")
    ret["learning_rate"] = config["Train"].getfloat("learning_rate")

    ret["batch_size"] = config["Train"].getint("batch_size")
    ret["hidden_size"] = config["Train"].getint("hidden_size")
    ret["dropout_prob"] = config["Train"].getfloat("dropout_prob")
    ret["teacher_forcing_ratio"] = config["Train"].getfloat("teacher_forcing_ratio")

    ret["enable_checkpoint"] = config["Train"].getboolean("enable_checkpoint")
    if ret["enable_checkpoint"]:
        ret["checkpoint_path"] = config["DEFAULT"]["experiment_dir"] + "checkpoints/"
        ret["checkpoint_cnt"] = config["Train"].getint("checkpoint_cnt")

    return Namespace(**ret)
