"""
dataset.py — loader per FineWeb-Edu (streaming) e dataset locali (.bin)
=======================================================================
Fix rispetto alla versione precedente:
  - Buffer shuffle sostituito con HF .shuffle() nativo: niente overlap
  - num_workers=0 per streaming (fix core dump con multiprocessing)
  - Separazione train/val interleaved (1 doc ogni VAL_EVERY)
  - tokenizer verbose=False per silenziare warning sul seq_len
"""

import os
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Streaming dataset per FineWeb-Edu
# ─────────────────────────────────────────────────────────────────────────────

class StreamingTextDataset(IterableDataset):
    """
    Legge FineWeb-Edu in streaming, tokenizza on-the-fly, produce
    chunk di lunghezza fissa seq_len senza overlap e senza padding.

    Fix chiave rispetto alla versione precedente:
      - usa HF .shuffle(buffer_size, seed) invece del buffer manuale
        → nessuna sequenza sovrapposta, distribuzione uniforme
      - num_workers=0 nel DataLoader per evitare il core dump con
        multiprocessing e file descriptor del dataset streaming
      - split interleaved: 1 doc ogni val_every va in val
      - stream lineare dei token: carry buffer senza indici casuali
    """

    def __init__(
        self,
        dataset_name:  str,
        split_name:    str,
        tokenizer:     PreTrainedTokenizer,
        seq_len:       int,
        buffer_size:   int = 1000,
        split:         str = "train",
        val_every:     int = 200,
        seed:          int = 42,
    ):
        super().__init__()
        self.dataset_name = dataset_name
        self.split_name   = split_name
        self.tokenizer    = tokenizer
        self.seq_len      = seq_len
        self.buffer_size  = buffer_size
        self.split        = split
        self.val_every    = val_every
        self.seed         = seed

    def _iter_documents(self) -> Iterator[list[int]]:
        from datasets import load_dataset

        ds = load_dataset(
            self.dataset_name,
            name=self.split_name,
            split="train",
            streaming=True,
        )
        ds = ds.shuffle(buffer_size=self.buffer_size, seed=self.seed)

        sep_id = (
            self.tokenizer.sep_token_id
            or self.tokenizer.eos_token_id
            or 0
        )

        doc_idx = 0
        for example in ds:
            text = (example.get("text") or "").strip()
            if not text:
                continue

            is_val_doc = (doc_idx % self.val_every == 0)
            if (self.split == "val") != is_val_doc:
                doc_idx += 1
                continue

            ids = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                verbose=False,
            )["input_ids"]

            if ids:
                ids.append(sep_id) # type: ignore
                yield ids # type: ignore

            doc_idx += 1

    def __iter__(self) -> Iterator[dict]:
        carry = []

        for ids in self._iter_documents():
            carry.extend(ids)

            while len(carry) >= self.seq_len:
                chunk = carry[:self.seq_len]
                carry = carry[self.seq_len:]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {
                    "input_ids":      input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset locale (.bin)
# ─────────────────────────────────────────────────────────────────────────────

class MaskedDataset(Dataset):
    def __init__(self, split: str, config):
        bin_file = config.train_bin_file if split == "train" else config.val_bin_file
        assert os.path.exists(bin_file), (
            f"File {bin_file!r} non trovato. Esegui download_dataset() prima."
        )
        self.data    = np.memmap(bin_file, dtype=np.uint16, mode="r")
        self.seq_len = config.seq_len
        self.n       = len(self.data) - self.seq_len

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int) -> dict:
        input_ids = torch.from_numpy(
            self.data[idx : idx + self.seq_len].astype(np.int64)
        )
        return {
            "input_ids":      input_ids,
            "attention_mask": torch.ones_like(input_ids),
        }


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(train_cfg, tokenizer: PreTrainedTokenizer):
    from torch.utils.data import DataLoader

    use_streaming = "fineweb" in train_cfg.dataset_name.lower() or \
                    not (hasattr(train_cfg, "train_bin_file") and
                         os.path.exists(train_cfg.train_bin_file))

    if use_streaming:
        print(f"Modalità streaming: {train_cfg.dataset_name} / {train_cfg.dataset_split_name}")

        train_ds = StreamingTextDataset(
            dataset_name=train_cfg.dataset_name,
            split_name=train_cfg.dataset_split_name,
            tokenizer=tokenizer,
            seq_len=train_cfg.seq_len,
            buffer_size=train_cfg.stream_buffer_size,
            split="train",
            val_every=200,
            seed=42,
        )
        val_ds = StreamingTextDataset(
            dataset_name=train_cfg.dataset_name,
            split_name=train_cfg.dataset_split_name,
            tokenizer=tokenizer,
            seq_len=train_cfg.seq_len,
            buffer_size=train_cfg.stream_buffer_size,
            split="val",
            val_every=200,
            seed=42,
        )

        # num_workers=0 OBBLIGATORIO con streaming HF
        # num_workers > 0 causa core dump per file descriptor condivisi
        train_loader = DataLoader(
            train_ds,
            batch_size=train_cfg.batch_size,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=train_cfg.batch_size,
            num_workers=0,
            pin_memory=True,
        )

    else:
        print(f"Modalità .bin locale: {train_cfg.train_bin_file}")
        train_loader = DataLoader(
            MaskedDataset("train", train_cfg),
            batch_size=train_cfg.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        val_loader = DataLoader(
            MaskedDataset("val", train_cfg),
            batch_size=train_cfg.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

    return train_loader, val_loader