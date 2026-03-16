"""
dataset.py — fase 1
====================
MixedStreamingDataset: mescola FineWeb-Edu e Wikipedia EN
in proporzioni configurabili (default 40/60) usando un
contatore cumulativo deterministico.

Vantaggi rispetto al random puro:
  - Proporzioni esatte garantite su milioni di documenti
  - Riproducibile con lo stesso seed
  - Niente squilibri accidentali su run lunghe
"""

import os
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Dataset, IterableDataset
from transformers import PreTrainedTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Dataset misto FineWeb-Edu + Wikipedia
# ─────────────────────────────────────────────────────────────────────────────

class MixedStreamingDataset(IterableDataset):
    """
    Mescola FineWeb-Edu e Wikipedia EN in streaming con proporzioni
    definite dai pesi fineweb_weight e wikipedia_weight.

    Strategia di mixing:
      Usa un contatore cumulativo deterministico — sceglie il dataset
      con il deficit maggiore rispetto alla proporzione target.
      Garantisce proporzioni esatte indipendentemente dalla lunghezza
      dei documenti o dalla fortuna del campionamento.

    Split train/val:
      1 documento ogni val_every va in validation (interleaved).
      Stessa distribuzione per train e val.
    """

    def __init__(
        self,
        tokenizer:        PreTrainedTokenizer,
        seq_len:          int,
        split:            str   = "train",
        val_every:        int   = 200,
        seed:             int   = 42,
        buffer_size:      int   = 1000,
        fineweb_weight:   float = 0.4,
        wikipedia_weight: float = 0.6,
    ):
        super().__init__()
        self.tokenizer        = tokenizer
        self.seq_len          = seq_len
        self.split            = split
        self.val_every        = val_every
        self.seed             = seed
        self.buffer_size      = buffer_size
        self.fineweb_weight   = fineweb_weight
        self.wikipedia_weight = wikipedia_weight

    # ── Iteratori per i singoli dataset ─────────────────────────────────────

    def _iter_fineweb(self) -> Iterator[list[int]]:
        from datasets import load_dataset
        ds = load_dataset(
            "HuggingFaceFW/fineweb-edu",
            name="CC-MAIN-2024-10",
            split="train",
            streaming=True,
        ).shuffle(buffer_size=self.buffer_size, seed=self.seed)

        sep_id  = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id or 0 
        doc_idx = 0

        for example in ds:
            text = (example.get("text") or "").strip()
            if not text:
                continue

            is_val = (doc_idx % self.val_every == 0)
            if (self.split == "val") != is_val:
                doc_idx += 1
                continue

            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                verbose=False,
            )
            ids = [int(x) for x in encoding.ids]

            if ids:
                yield ids + [sep_id] # type: ignore

            doc_idx += 1

    def _iter_wikipedia(self) -> Iterator[list[int]]:
        from datasets import load_dataset
        ds = load_dataset(
            "wikimedia/wikipedia",
            name="20231101.en",
            split="train",
            streaming=True,
        ).shuffle(buffer_size=self.buffer_size, seed=self.seed + 1)

        sep_id  = self.tokenizer.sep_token_id or self.tokenizer.eos_token_id or 0 
        doc_idx = 0

        for example in ds:
            text = (example.get("text") or "").strip()
            if not text:
                continue

            is_val = (doc_idx % self.val_every == 0)
            if (self.split == "val") != is_val:
                doc_idx += 1
                continue

            encoding = self.tokenizer(
                text,
                add_special_tokens=False,
                truncation=False,
                return_attention_mask=False,
                verbose=False,
            )
            ids = [int(x) for x in encoding.ids]

            if ids:
                yield ids + [sep_id] # type: ignore

            doc_idx += 1

    # ── Mixing deterministico ────────────────────────────────────────────────

    def __iter__(self) -> Iterator[dict]:
        """
        Alterna documenti tra FineWeb e Wikipedia usando un contatore
        cumulativo che garantisce le proporzioni esatte target.

        Logica:
          cum_fw   tiene conto di quanti "crediti" FineWeb ha consumato
          cum_wiki tiene conto di quanti "crediti" Wikipedia ha consumato
          Scegliamo sempre il dataset con il deficit maggiore.
        """
        gen_fw   = self._iter_fineweb()
        gen_wiki = self._iter_wikipedia()
        carry    = []

        # Contatori cumulativi: incremento = 1/peso
        # Dataset con peso 0.4 incrementa di 2.5 per doc → più raro
        # Dataset con peso 0.6 incrementa di 1.67 per doc → più frequente
        cum_fw   = 0.0
        cum_wiki = 0.0
        inc_fw   = 1.0 / self.fineweb_weight    # 2.5
        inc_wiki = 1.0 / self.wikipedia_weight  # 1.67

        while True:
            # Sceglie il dataset con il contatore minore (meno "usato")
            if cum_fw <= cum_wiki:
                try:
                    ids = next(gen_fw)
                    cum_fw += inc_fw
                except StopIteration:
                    # FineWeb esaurito — continua solo con Wikipedia
                    try:
                        ids = next(gen_wiki)
                    except StopIteration:
                        break
            else:
                try:
                    ids = next(gen_wiki)
                    cum_wiki += inc_wiki
                except StopIteration:
                    # Wikipedia esaurita — continua solo con FineWeb
                    try:
                        ids = next(gen_fw)
                    except StopIteration:
                        break

            carry.extend(ids)

            # Estrai chunk di seq_len senza overlap
            while len(carry) >= self.seq_len:
                chunk = carry[:self.seq_len]
                carry = carry[self.seq_len:]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {
                    "input_ids":      input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }


# ─────────────────────────────────────────────────────────────────────────────
# Dataset locale (.bin) — mantenuto per compatibilità
# ─────────────────────────────────────────────────────────────────────────────

class MaskedDataset(Dataset):
    def __init__(self, split: str, config):
        bin_file = config.train_bin_file if split == "train" else config.val_bin_file
        assert os.path.exists(bin_file), f"File {bin_file!r} non trovato."
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

    if train_cfg.dataset_name == "fase1":
        print(f"Modalità fase 1: FineWeb-Edu {train_cfg.fineweb_weight:.0%} "
              f"+ Wikipedia EN {train_cfg.wikipedia_weight:.0%}")

        train_ds = MixedStreamingDataset(
            tokenizer=tokenizer,
            seq_len=train_cfg.seq_len,
            split="train",
            val_every=train_cfg.val_every,
            seed=42,
            buffer_size=train_cfg.stream_buffer_size,
            fineweb_weight=train_cfg.fineweb_weight,
            wikipedia_weight=train_cfg.wikipedia_weight,
        )
        val_ds = MixedStreamingDataset(
            tokenizer=tokenizer,
            seq_len=train_cfg.seq_len,
            split="val",
            val_every=train_cfg.val_every,
            seed=42,
            buffer_size=train_cfg.stream_buffer_size,
            fineweb_weight=train_cfg.fineweb_weight,
            wikipedia_weight=train_cfg.wikipedia_weight,
        )

        # num_workers=0 obbligatorio con streaming HF
        train_loader = DataLoader(
            train_ds, batch_size=train_cfg.batch_size,
            num_workers=0, pin_memory=True,
        )
        val_loader = DataLoader(
            val_ds, batch_size=train_cfg.batch_size,
            num_workers=0, pin_memory=True,
        )

    else:
        raise ValueError(
            f"dataset_name={train_cfg.dataset_name!r} non riconosciuto. "
            f"Usa 'fase1' per il pretraining misto."
        )

    return train_loader, val_loader