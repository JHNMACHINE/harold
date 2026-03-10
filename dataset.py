import os

import numpy as np
import torch
from datasets import load_dataset
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from config import DistributedConfig, TrainConfig


def download_dataset(config: TrainConfig) -> None:
    """
    Scarica e tokenizza il dataset in due file .bin (train / val).
    Se i file esistono già, salta la preparazione.
    """
    if os.path.exists(config.train_bin_file) and os.path.exists(config.val_bin_file):
        print("Found existing .bin files. Skipping data preparation.")
        return

    print("Binary data files not found. Starting data download and tokenization...")

    ds  = load_dataset(config.dataset_name)
    enc = AutoTokenizer.from_pretrained(config.tokenizer_model)

    assert enc.vocab_size < 2**16, "Tokenizer vocab size too large for uint16"

    def tokenize(example):
        ids = enc.encode(
            example["text"],
            add_special_tokens=False,
            truncation=True,
            max_length=config.seq_len,
        )
        return {"ids": ids, "len": len(ids)}

    tokenized_ds = ds.map(
        tokenize,
        remove_columns=["text"],
        desc="Tokenizing splits",
        num_proc=os.cpu_count(),
    )

    for split, dset in tokenized_ds.items():
        filename     = config.val_bin_file if split == "validation" else config.train_bin_file
        total_tokens = np.sum(dset["len"], dtype=np.uint64)
        print(f"Found {total_tokens:,} tokens in the '{split}' split.")

        arr = np.memmap(filename, dtype=np.uint16, mode="w+", shape=(total_tokens,))  # type: ignore
        print(f"Writing tokens to {filename}...")

        current_idx = 0
        for batch in tqdm(dset.iter(batch_size=2048), total=len(dset) // 2048):
            chunk = np.concatenate(batch["ids"])  # type: ignore
            arr[current_idx : current_idx + len(chunk)] = chunk
            current_idx += len(chunk)

        arr.flush()

    print("Tokenization and file writing complete.")


class MaskedDataset(Dataset):
    """
    Dataset per il training del DiffusionMoE.
    Legge sequenze da un file .bin (uint16) prodotto da download_dataset()
    e restituisce dict {input_ids, attention_mask} pronti per il diffusion trainer.
    """

    def __init__(self, split: str, config: TrainConfig):
        bin_file = config.train_bin_file if split == "train" else config.val_bin_file
        assert os.path.exists(bin_file), (
            f"File {bin_file!r} non trovato. Esegui download_dataset() prima del training."
        )
        self.data      = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.seq_len   = config.seq_len
        self.n_samples = len(self.data) - self.seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        input_ids = torch.from_numpy(
            self.data[idx : idx + self.seq_len].astype(np.int64)
        )
        return {
            "input_ids":      input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }

class DistributedDataset(Dataset):
    """
    Dataset per il training del DiffusionMoE.
    Legge sequenze da un file .bin (uint16) prodotto da download_dataset()
    e restituisce dict {input_ids, attention_mask} pronti per il diffusion trainer.
    """

    def __init__(self, split: str, config: DistributedConfig):
        bin_file = config.train_bin_file if split == "train" else config.val_bin_file
        assert os.path.exists(bin_file), (
            f"File {bin_file!r} non trovato"
        )
        self.data      = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.seq_len   = config.seq_len
        self.n_samples = len(self.data) - self.seq_len

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict:
        input_ids = torch.from_numpy(
            self.data[idx : idx + self.seq_len].astype(np.int64)
        )
        return {
            "input_ids":      input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }