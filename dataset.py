from io import open
import os.path
import shutil
import requests
import zipfile

import unicodedata
import re
import torch
import torch.utils.data
from typing import Tuple
from tqdm import tqdm

SOS_TOKEN = 0
EOS_TOKEN = 1

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
}


def is_download_link_available(url):
    response = requests.get(
        url,
        headers=headers,
        stream=True,
    )
    return response.status_code == 200


def download(url: str, fname: str, chunk_size=1024):
    resp = requests.get(url, headers=headers, stream=True)
    total = int(resp.headers.get("content-length", 0))
    with open(fname, "wb") as file, tqdm(
        desc=fname,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            bar.update(size)


class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {SOS_TOKEN: "<SOS>", EOS_TOKEN: "<EOS>"}
        self.n_words = 2  # Count SOS and EOS

    def addSentence(self, sentence):
        for word in sentence.split(" "):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    def toTensor(self, sentence):
        indexes = [self.word2index[word] for word in sentence.split(" ")]
        indexes.append(EOS_TOKEN)
        return torch.tensor(indexes, dtype=torch.long).view(-1, 1)

    def toSentence(self, sentence_tensor):
        word_list = [self.index2word[i.item()] for i in sentence_tensor]
        return " ".join(word_list)


def unicodeToAscii(s):
    """Turn a Unicode string to plain ASCII

    Turn a Unicode string to plain ASCII, thanks to
    https://stackoverflow.com/a/518232/2809427

    Args:
        s (str): Unicode string.

    Returns:
        str: ASCII string.
    """
    return "".join(
        c for c in unicodedata.normalize("NFD", s) if unicodedata.category(c) != "Mn"
    )


def normalizeString(s):
    """Lowercase, trim and remove non-letter characters.

    Args:
        s (str): String to be normalized.

    Returns:
        str: Normalized string.
    """
    s = s.lower().strip()
    if " " not in s:
        s = list(s)
        s = " ".join(s)
    s = unicodeToAscii(s)
    s = re.sub(r"([.!?])", r" \1", s)
    return s


def download_data(input_lang_name, output_lang_name, dataset_dir):
    if os.path.exists(dataset_dir):
        shutil.rmtree(dataset_dir)
    os.makedirs(dataset_dir)

    url = f"https://www.manythings.org/anki/{output_lang_name}-{input_lang_name}.zip"
    if not is_download_link_available(url):
        print(f"Warning: {url} not available!")
        input_lang_name, output_lang_name = output_lang_name, input_lang_name
        url = (
            f"https://www.manythings.org/anki/{output_lang_name}-{input_lang_name}.zip"
        )
        print(f"Trying {url}")
        if not is_download_link_available(url):
            raise Exception("Dataset download failed")

    dataset_filename_raw = f"{input_lang_name}-{output_lang_name}.zip"

    dataset_zip = os.path.join(dataset_dir, dataset_filename_raw)
    print(f"Downloading dataset: {url} => {dataset_zip}")
    download(url, dataset_zip)

    if not os.path.isfile(dataset_zip):
        raise Exception("Dataset download failed")

    dataset_dir_raw = os.path.join(dataset_dir, "raw")
    os.makedirs(dataset_dir_raw)

    print("Extracting example data ... ", end="")
    with zipfile.ZipFile(dataset_zip, "r") as zip_ref:
        zip_ref.extractall(dataset_dir_raw)

    dataset_file_raw = os.path.join(dataset_dir_raw, output_lang_name + ".txt")
    if not os.path.isfile(dataset_file_raw):
        raise Exception("Extract failed!")
    else:
        print("Done.")

    dataset_file = os.path.join(
        dataset_dir, f"{input_lang_name}-{output_lang_name}.txt"
    )
    print("Formatting example data ...")
    with open(dataset_file_raw, "r", encoding="utf-8") as f_raw:
        with open(dataset_file, "w", encoding="utf-8") as f_out:
            for line in f_raw:
                new_line = line.strip().split("\t")[0:2]
                f_out.write("\t".join(new_line) + "\n")

    print("Clean up temporary files ...")
    os.remove(dataset_zip)
    print("Done.")


def readLangs(input_lang_name, output_lang_name, dataset_dir):
    file_path = os.path.join(dataset_dir, f"{input_lang_name}-{output_lang_name}.txt")
    reverse = False
    if not os.path.isfile(file_path):
        print(f"Warning: File {file_path} does not exist!")
        print("Trying to read the reversed version.")
        file_path = os.path.join(
            dataset_dir, f"{output_lang_name}-{input_lang_name}.txt"
        )
        reverse = True
        if not os.path.isfile(file_path):
            print(f"Warning: File {file_path} does not exist!")
            print("Trying to download dataset from remote server.")
            download_data(input_lang_name, output_lang_name, dataset_dir)
            return readLangs(input_lang_name, output_lang_name, dataset_dir)

    # Read the file and split into lines
    print(f"Reading lines from {file_path} ...")
    lines = open(file_path, encoding="utf-8").read().strip().split("\n")

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split("\t")] for l in lines]

    # Reverse pairs if required
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]

    input_lang, output_lang = Lang(input_lang_name), Lang(output_lang_name)

    return input_lang, output_lang, pairs


def filterPair(
    p,
    max_length,
    enable_input_prefix_filter,
    enable_output_prefix_filter,
    input_prefix_filter=[],
    output_prefix_filter=[],
    **kwargs,
):
    return (
        len(p[0].split(" ")) < max_length
        and len(p[1].split(" ")) < max_length
        and (
            p[0].startswith(tuple(input_prefix_filter))
            or not enable_input_prefix_filter
        )
        and (
            p[1].startswith(tuple(output_prefix_filter))
            or not enable_output_prefix_filter
        )
    )


class DatasetManythingsAnki(torch.utils.data.Dataset):
    def __init__(
        self,
        input_lang_name,
        output_lang_name,
        dataset_dir,
        **kwargs,
    ) -> None:
        super().__init__()
        self.input_lang, self.output_lang, self.pairs = readLangs(
            input_lang_name, output_lang_name, dataset_dir
        )
        print("Read %s sentence pairs." % len(self.pairs))
        self.pairs = [pair for pair in self.pairs if filterPair(pair, **kwargs)]
        print("Trimmed to %s sentence pairs." % len(self.pairs))
        print("Counting words...", end="")
        for pair in self.pairs:
            self.input_lang.addSentence(pair[0])
            self.output_lang.addSentence(pair[1])
        print("\rCounted words:    ")
        print(
            "  %s: %s <=> %s: %s"
            % (
                self.input_lang.name,
                self.input_lang.n_words,
                self.output_lang.name,
                self.output_lang.n_words,
            )
        )

    def __getitem__(self, index) -> Tuple[str, str]:
        return self.pairs[index]

    def __len__(self) -> int:
        return len(self.pairs)
