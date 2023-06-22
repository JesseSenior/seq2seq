import argparse
import os
import ast
import random
import torch
from torch.utils.data import Subset
from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from typing import Tuple

from config import parse_config
from model import Model

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import warnings

warnings.filterwarnings(
    "ignore", ".*FixedFormatter should only be used together with FixedLocator.*"
)


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

    print("Plotting model's metrics ...")
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
    plt.savefig(os.path.join(config.experiment_dir, "result.pdf"), bbox_inches="tight")
    # plt.show()
    print(f"Done. Figure saved at {os.path.join(config.experiment_dir, 'result.pdf')}")
    print()

    print("Evaluting the bleu score")
    if not os.path.isfile(config.model_path):
        raise Exception("Model not found")
    model = config.model.load_from_checkpoint(config.model_path)
    model.eval()

    dataset = config.dataset(**vars(config))

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset,
        [
            int(config.training_set_ratio * len(dataset)),
            len(dataset) - int(config.training_set_ratio * len(dataset)),
        ],
    )

    sum_score = 0
    # (score, is,ts)
    # 小根堆
    heap = []
    heap_capacity = 3

    def heap_push(x: Tuple[float, str, str]):
        heap.append(x)
        id = len(heap) - 1
        while heap[id // 2][0] > heap[id][0]:
            heap[id // 2], heap[id] = heap[id], heap[id // 2]
            id //= 2

    def heap_pop():
        id = len(heap) - 1
        heap[0], heap[id] = heap[id], heap[0]
        heap.pop()
        id = 0
        while id * 2 < len(heap):
            nid = (
                id * 2 + 1
                if id * 2 + 1 < len(heap) and heap[id * 2][0] > heap[id * 2 + 1][0]
                else id * 2
            )
            if heap[id][0] > heap[nid][0]:
                heap[id], heap[nid] = heap[nid], heap[id]
                id = nid
            else:
                break

    def heap_update(x: Tuple[float, str, str]):
        random.seed()
        if (
            len(heap) < heap_capacity
            or x[0] > heap[0][0]
            or (abs(x[0] - heap[0][0]) < 1e-3 and 0.5 < random.random())
        ):
            heap_push(x)
        while len(heap) > heap_capacity:
            heap_pop()

    progress_bar = tqdm(val_dataset, leave=False)
    for (
        input_sentence,
        target_sentence,
    ) in progress_bar:  # tqdm(Subset(val_dataset, range(100)))
        ts = target_sentence + " <EOS>"
        hs = model(input_sentence)
        bleu_score = sentence_bleu(
            [ts.split()], hs.split(), smoothing_function=SmoothingFunction().method1
        )
        progress_bar.set_postfix({"BLEU": "%.3f" % bleu_score})
        heap_update((bleu_score, input_sentence, ts))
        sum_score += bleu_score

    print(f"The average bleu score is: {sum_score/len(val_dataset):.5f}")
    print(f"The best {len(heap)} results shows as follows:")
    print()

    plt.rcParams["font.sans-serif"] = ["KaiTi"]
    plt.rcParams["axes.unicode_minus"] = False
    for id, (score, input_sentence, target_sentence) in enumerate(
        sorted(heap, key=lambda x: x[0], reverse=True), start=1
    ):
        predict_sentence, attentions = model(input_sentence, get_attention=True)

        print(f"===== No. {id} =====")
        print(f"  Input Sentence: {input_sentence}")
        print(f" Target Sentence: {target_sentence}")
        print(f"Predict Sentence: {predict_sentence}")
        print(f"      BLEU Score: {score:.3f}")

        fig, ax = plt.subplots()
        cax = ax.matshow(attentions, cmap="bone")
        mat_ratio = attentions.shape[0] / attentions.shape[1]
        fig.colorbar(cax, fraction=0.046 * mat_ratio, pad=0.04)

        # Set up axes
        ax.set_xticklabels([""] + input_sentence.split())
        ax.set_yticklabels([""] + predict_sentence.split())

        # Show label at every tick
        ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
        ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
        plt.savefig(
            os.path.join(config.experiment_dir, f"max_attn_{id}.pdf"),
            bbox_inches="tight",
        )

        print(
            f"Attention Matrix: Saved to {os.path.join(config.experiment_dir, f'max_attn_{id}.pdf')}"
        )
        print("=" * len(f"===== No. {id} ====="))
        print()
