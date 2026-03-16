"""
dataset.py — Harold v3
==========================
MixedStreamingDataset a 5 sorgenti:
  - FineWeb-Edu  30%  (base accademica)
  - Wikipedia EN 20%  (conoscenza fattuale)
  - Books/Gutenberg 20% (narrativa, coerenza lunga)
  - C4            15%  (testo web generico)
  - OpenWebText   15%  (Reddit/HN, registro informale)

Aggiunge:
  - compute_token_weights(): precomputa frequenze per token-weighted schedule
  - Mixing deterministico con contatore cumulativo (invariato da v2)
  - num_workers=0 obbligatorio per streaming HF (fix core dump)
"""

import os
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import IterableDataset
from transformers import PreTrainedTokenizer


# ─────────────────────────────────────────────────────────────────────────────
# Token frequency computation — per token-weighted noise schedule
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_weights(
    tokenizer:    PreTrainedTokenizer,
    save_path:    str  = "token_weights.pt",
    n_docs:       int  = 50000,
    dataset_name: str  = "HuggingFaceFW/fineweb-edu",
    subset:       str  = "CC-MAIN-2024-10",
) -> torch.Tensor:
    """
    Precomputa i pesi dei token basati sulla frequenza inversa nel dataset.

    Token rari (IDF alto) → peso alto  → mascherati prima
    Token comuni           → peso basso → mascherati dopo

    Ritorna un tensore (vocab_size,) float in [0, 1].
    Salva su disco per non ricalcolarlo ad ogni training.

    Tempo stimato: ~10 minuti su 50k documenti.
    """
    if os.path.exists(save_path):
        print(f"Token weights trovati: {save_path}")
        return torch.load(save_path, weights_only=True)

    print(f"Calcolo token weights su {n_docs} documenti...")
    from datasets import load_dataset
    from collections import Counter

    ds      = load_dataset(dataset_name, name=subset, split="train", streaming=True)
    counter = Counter()
    n_docs_seen = 0

    for example in ds:
        if n_docs_seen >= n_docs:
            break
        text = (example.get("text") or "").strip()
        if not text:
            continue
        ids = tokenizer(
            text, truncation=True, max_length=512,
            return_attention_mask=False, verbose=False,
        )["input_ids"]
        counter.update(ids) # type: ignore
        n_docs_seen += 1
        if n_docs_seen % 5000 == 0:
            print(f"  {n_docs_seen}/{n_docs} documenti processati...")

    # Costruisci tensore frequenze
    freq = torch.zeros(tokenizer.vocab_size)
    for tok_id, count in counter.items():
        if 0 <= tok_id < tokenizer.vocab_size:
            freq[tok_id] = float(count)

    # IDF: log(N / (freq + 1)) — token rari hanno IDF alto
    N   = float(freq.sum())
    idf = torch.log(N / (freq + 1.0))

    # Normalizza in [0, 1]
    w = (idf - idf.min()) / (idf.max() - idf.min() + 1e-8)

    torch.save(w, save_path)
    print(f"Token weights salvati: {save_path}")
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Iteratori per i singoli dataset
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_doc(text: str, tokenizer: PreTrainedTokenizer, sep_id: int) -> list[int]:
    """Tokenizza un documento e aggiunge il separatore finale."""
    ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
        verbose=False,
    )["input_ids"]
    return [int(x) for x in ids] + [sep_id] # type: ignore


def _iter_dataset(
    load_kwargs:  dict,
    text_field:   str,
    tokenizer:    PreTrainedTokenizer,
    sep_id:       int,
    split:        str,
    val_every:    int,
    buffer_size:  int,
    seed:         int,
) -> Iterator[list[int]]:
    """
    Iteratore generico per qualsiasi dataset HF con campo testo.
    Applica split train/val interleaved.
    """
    from datasets import load_dataset

    ds = load_dataset(**load_kwargs, streaming=True)
    ds = ds.shuffle(buffer_size=buffer_size, seed=seed)

    doc_idx = 0
    for example in ds:
        text = (example.get(text_field) or "").strip()
        if not text:
            continue

        is_val = (doc_idx % val_every == 0)
        if (split == "val") != is_val:
            doc_idx += 1
            continue

        ids = _tokenize_doc(text, tokenizer, sep_id)
        if ids:
            yield ids

        doc_idx += 1


# ─────────────────────────────────────────────────────────────────────────────
# MixedStreamingDataset v3 — 5 sorgenti
# ─────────────────────────────────────────────────────────────────────────────

class MixedStreamingDataset(IterableDataset):
    """
    Mescola 5 dataset HF in streaming con proporzioni configurabili.

    Mixing deterministico con contatore cumulativo:
      - Sceglie il dataset con il deficit maggiore rispetto alla proporzione target
      - Proporzioni esatte garantite su milioni di documenti
      - Riproducibile con lo stesso seed

    Dataset e proporzioni default:
      FineWeb-Edu  30%  — base accademica, vocabolario ricco
      Wikipedia    20%  — conoscenza fattuale strutturata
      Books        20%  — narrativa, coerenza su sequenze lunghe
      C4           15%  — testo web generico, varietà di registro
      OpenWebText  15%  — Reddit/HN, linguaggio informale/conversazionale
    """

    # Configurazione dei dataset — modifica qui per cambiare sorgenti
    DATASET_CONFIGS = {
        "fineweb": {
            "load_kwargs": {
                "path":  "HuggingFaceFW/fineweb-edu",
                "name":  "CC-MAIN-2024-10",
                "split": "train",
            },
            "text_field": "text",
        },
        "wikipedia": {
            "load_kwargs": {
                "path":  "wikimedia/wikipedia",
                "name":  "20231101.en",
                "split": "train",
            },
            "text_field": "text",
        },
        "books": {
            "load_kwargs": {
                "path":  "monology/pile-uncopyrighted",
                "split": "train",
            },
            "text_field": "text",
        },
        "c4": {
            "load_kwargs": {
                "path":  "allenai/c4",
                "name":  "en",
                "split": "train",
            },
            "text_field": "text",
        },
        "owt": {
            "load_kwargs": {
                "path":  "Skylion007/openwebtext",
                "split": "train",
            },
            "text_field": "text",
        },
    }

    def __init__(
        self,
        tokenizer:    PreTrainedTokenizer,
        seq_len:      int,
        split:        str   = "train",
        val_every:    int   = 200,
        seed:         int   = 42,
        buffer_size:  int   = 1000,
        weights:      dict  | None = None,
    ):
        """
        weights: dizionario {nome_dataset: proporzione} — default 30/20/20/15/15
        """
        super().__init__()
        self.tokenizer   = tokenizer
        self.seq_len     = seq_len
        self.split       = split
        self.val_every   = val_every
        self.seed        = seed
        self.buffer_size = buffer_size

        # Pesi default
        self.weights = weights or {
            "fineweb":   0.30,
            "wikipedia": 0.20,
            "books":     0.20,
            "c4":        0.15,
            "owt":       0.15,
        }

        # Verifica che i pesi sommino a ~1
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 0.01, f"I pesi devono sommare a 1.0, trovato {total:.3f}"

        # Verifica che tutti i dataset siano configurati
        for name in self.weights:
            assert name in self.DATASET_CONFIGS, f"Dataset sconosciuto: {name!r}"

    def _make_iterators(self) -> dict[str, Iterator[list[int]]]:
        """Crea un iteratore per ogni dataset attivo."""
        sep_id = int(self.tokenizer.sep_token_id or self.tokenizer.eos_token_id or 0) # type: ignore
        iters  = {}

        for i, (name, cfg) in enumerate(self.DATASET_CONFIGS.items()):
            if name not in self.weights:
                continue
            iters[name] = _iter_dataset(
                load_kwargs  = cfg["load_kwargs"],
                text_field   = cfg["text_field"],
                tokenizer    = self.tokenizer,
                sep_id       = sep_id,
                split        = self.split,
                val_every    = self.val_every,
                buffer_size  = self.buffer_size,
                seed         = self.seed + i * 7,   # seed diverso per ogni sorgente
            )

        return iters

    def __iter__(self) -> Iterator[dict]:
        """
        Mixing deterministico con contatore cumulativo.

        Per ogni dataset mantiene un contatore cumulativo:
          incremento = 1 / peso
        Dataset con peso 0.30 incrementa di 3.33 per doc → meno frequente
        Dataset con peso 0.15 incrementa di 6.67 per doc → ancora meno

        Scegliamo sempre il dataset con il contatore minore (il più "in debito").
        Questo garantisce le proporzioni esatte su qualsiasi numero di documenti.
        """
        iters    = self._make_iterators()
        names    = list(iters.keys())
        carry    = []

        # Contatori cumulativi: init uguale per tutti
        counters = {name: 0.0 for name in names}
        incs     = {name: 1.0 / self.weights[name] for name in names}

        # Dataset esauriti — rimuovere dalla rotazione
        exhausted = set()

        while len(exhausted) < len(names):
            # Sceglie il dataset con contatore minore tra i non esauriti
            active  = [(counters[n], n) for n in names if n not in exhausted]
            if not active:
                break
            _, chosen = min(active)

            try:
                ids = next(iters[chosen])
                counters[chosen] += incs[chosen]
            except StopIteration:
                exhausted.add(chosen)
                continue

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
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(train_cfg, tokenizer: PreTrainedTokenizer):
    """
    Costruisce train_loader e val_loader per il training v3.
    num_workers=0 obbligatorio con streaming HF (fix core dump multiprocessing).
    """
    from torch.utils.data import DataLoader

    weights = {
        "fineweb":   train_cfg.fineweb_weight,
        "wikipedia": train_cfg.wikipedia_weight,
        "books":     train_cfg.books_weight,
        "c4":        train_cfg.c4_weight,
        "owt":       train_cfg.owt_weight,
    }

    print(
        f"Dataset mix v3: "
        + ", ".join(f"{k} {v:.0%}" for k, v in weights.items())
    )

    train_ds = MixedStreamingDataset(
        tokenizer=tokenizer, seq_len=train_cfg.seq_len,
        split="train", val_every=train_cfg.val_every,
        seed=42, buffer_size=train_cfg.stream_buffer_size,
        weights=weights,
    )
    val_ds = MixedStreamingDataset(
        tokenizer=tokenizer, seq_len=train_cfg.seq_len,
        split="val", val_every=train_cfg.val_every,
        seed=42, buffer_size=train_cfg.stream_buffer_size,
        weights=weights,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.batch_size,
        num_workers=0,    # obbligatorio con streaming HF
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.batch_size,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader