import argparse
import os

from config import parse_config
from model import Model


# Dirty hack for autocomplete
class Config:
    model: Model
    model_path: str


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="Running a model in interactive mode.")
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

    print("=====")
    print("Input  Language: " + model.input_lang.name)
    print("Output Language: " + model.output_lang.name)
    print("=====")
    print()

    while True:
        try:
            s = input("<<< ")
            print(">>> " + model.infer(s))
        except (EOFError, KeyboardInterrupt):
            break
        except KeyError as e:
            print(f"!!! ERROR: Unknown key '{e.args[0]}'")
        print()