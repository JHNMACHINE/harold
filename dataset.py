"""
dataset.py — Harold v0.4
==========================
MixedStreamingDataset a 5 sorgenti:
  - FineWeb-Edu  30%  (base accademica)
  - Wikipedia EN 20%  (conoscenza fattuale)
  - Books/Gutenberg 20% (narrativa, coerenza lunga)
  - C4            15%  (testo web generico)
  - OpenWebText   15%  (Reddit/HN, registro informale)

Cambiamenti v0.4 rispetto a v0.3:
  - sep_id: usa eos_token_id (GPT-2: 50256) invece di sep_token_id (BERT: 102)
  - padding SFT: usa pad_token_id del tokenizer invece di 0
    (con GPT-2, token 0 = "!" — un token valido, non padding)
  - mask SFT: usa pad_token_id per costruire response_mask correttamente
"""

import os
from typing import Iterator, Optional
import torch
from transformers import BatchEncoding, PreTrainedTokenizer
from torch.utils.data import IterableDataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Token frequency computation — per token-weighted noise schedule
# ─────────────────────────────────────────────────────────────────────────────

def compute_token_weights(
    tokenizer:    PreTrainedTokenizer,
    weights:      dict,
    save_path:    str = "token_weights.pt",
    n_docs_total: int = 50000,
) -> torch.Tensor:
    """
    Precomputa i pesi dei token campionando dal mix di dataset.
    Usato per il token-weighted noise schedule.
    """
    if os.path.exists(save_path):
        print(f"Token weights trovati: {save_path}")
        return torch.load(save_path, weights_only=True)

    from datasets import load_dataset
    from collections import Counter

    print(f"Calcolo token weights su {n_docs_total} documenti dal mix...")
    counter = Counter()

    for i, (name, cfg) in enumerate(MixedStreamingDataset.DATASET_CONFIGS.items()):
        if name not in weights:
            continue

        n_docs = max(1, int(n_docs_total * weights[name]))
        print(f"  {name}: {n_docs} documenti ({weights[name]:.0%})...")

        ds = load_dataset(**cfg["load_kwargs"], streaming=True)
        ds = ds.shuffle(buffer_size=500, seed=42 + i)

        n_seen = 0
        for example in ds:
            if n_seen >= n_docs:
                break
            text = (example.get(cfg["text_field"]) or "").strip()
            if not text:
                continue
            tokenized = tokenizer(
                text, truncation=True, max_length=512,
                return_attention_mask=False,
            )
            ids = tokenized["input_ids"]
            # Ensure ids is a list of ints
            if isinstance(ids, list):
                counter.update(ids)
            elif isinstance(ids, (int, float)):
                counter.update([int(ids)])
            n_seen += 1

    freq = torch.zeros(tokenizer.vocab_size)
    for tok_id, count in counter.items():
        if isinstance(tok_id, (int, float)) and 0 <= tok_id < tokenizer.vocab_size:
            freq[int(tok_id)] = float(count)

    N   = float(freq.sum())
    idf = torch.log(N / (freq + 1.0))
    w   = (idf - idf.min()) / (idf.max() - idf.min() + 1e-8)

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    torch.save(w, save_path)
    print(f"Token weights salvati: {save_path}")
    return w


# ─────────────────────────────────────────────────────────────────────────────
# Iteratori per i singoli dataset
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize_doc(text: str, tokenizer: PreTrainedTokenizer, sep_id: int) -> list[int]:
    """Tokenizza un documento e aggiunge il separatore finale."""
    tokenized = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_attention_mask=False,
    )
    
    # Extract input_ids safely
    ids = None
    
    # Handle dict-like objects (BatchEncoding)
    if isinstance(tokenized, dict):
        ids = tokenized.get("input_ids")
    # Handle objects with input_ids attribute (Encoding, BatchEncoding)
    elif hasattr(tokenized, "input_ids"):
        ids = tokenized.input_ids
    else:
        # Fallback: use tokenized directly
        ids = tokenized
    
    # If ids is still None, return just the separator
    if ids is None:
        return [sep_id]
    
    # Convert to list of ints
    if hasattr(ids, "tolist"):
        # PyTorch/TensorFlow tensor
        token_ids = ids.tolist()
    elif isinstance(ids, (list, tuple)):
        token_ids = list(ids)
    elif isinstance(ids, (int, float)):
        token_ids = [int(ids)]
    else:
        # Try to convert iterable to list
        try:
            token_ids = list(ids) if hasattr(ids, "__iter__") else [ids]
        except (TypeError, ValueError):
            token_ids = []
    
    # Ensure all are ints and filter out invalid values
    result = []
    for x in token_ids:
        # Skip if x is a dict or BatchEncoding or any container type
        if isinstance(x, (dict, BatchEncoding)):
            # Recursively extract from nested structures
            result.extend(_extract_token_ids_from_value(x))
        else:
            try:
                result.append(int(x))
            except (TypeError, ValueError):
                continue
    
    return result + [sep_id]

def _extract_token_ids_from_value(value) -> list[int]:
    """Helper to extract token IDs from nested values."""
    if value is None:
        return []
    
    if isinstance(value, (dict, BatchEncoding)):
        ids = value.get("input_ids") if isinstance(value, dict) else getattr(value, "input_ids", None)
        if ids is not None:
            return _extract_token_ids_from_value(ids)
        return []
    
    if isinstance(value, (list, tuple)):
        result = []
        for item in value:
            result.extend(_extract_token_ids_from_value(item))
        return result
    
    if hasattr(value, "tolist"):
        return _extract_token_ids_from_value(value.tolist())
    
    try:
        return [int(value)]
    except (TypeError, ValueError):
        return []


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
    """Iteratore generico per qualsiasi dataset HF con campo testo."""
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
# MixedStreamingDataset v0.4 — 5 sorgenti
# ─────────────────────────────────────────────────────────────────────────────

class MixedStreamingDataset(IterableDataset):
    """
    Mescola 5 dataset HF in streaming con proporzioni configurabili.
    Mixing deterministico con contatore cumulativo.
    """

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
        split:        str        = "train",
        val_every:    int        = 200,
        seed:         int        = 42,
        buffer_size:  int        = 1000,
        weights:      dict | None = None,
    ):
        super().__init__()
        self.tokenizer   = tokenizer
        self.seq_len     = seq_len
        self.split       = split
        self.val_every   = val_every
        self.seed        = seed
        self.buffer_size = buffer_size
        self.weights = weights or {
            "fineweb":   0.30,
            "wikipedia": 0.20,
            "books":     0.20,
            "c4":        0.15,
            "owt":       0.15,
        }
        total = sum(self.weights.values())
        assert abs(total - 1.0) < 0.01, f"Pesi devono sommare a 1.0, trovato {total:.3f}"
        for name in self.weights:
            assert name in self.DATASET_CONFIGS, f"Dataset sconosciuto: {name!r}"

    def _make_iterators(self) -> dict[str, Iterator[list[int]]]:
        # [v0.4] GPT-2 non ha sep_token — usa eos_token_id come separatore
        # BERT aveva sep_token_id=102 ([SEP]), GPT-2 usa eos_token_id=50256
        
        # Helper function to safely extract a single token ID
        def get_token_id(token_id):
            if token_id is None:
                return None
            if isinstance(token_id, list):
                return token_id[0] if token_id else None
            return token_id
        
        sep_token = get_token_id(self.tokenizer.sep_token_id)
        eos_token = get_token_id(self.tokenizer.eos_token_id)
        
        sep_id = int(
            sep_token
            if sep_token is not None
            else eos_token
            if eos_token is not None
            else 0
        )

        iters = {}
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
                seed         = self.seed + i * 7,
            )
        return iters

    def __iter__(self) -> Iterator[dict]:
        iters     = self._make_iterators()
        names     = list(iters.keys())
        carry: list[int] = []
        counters  = {name: 0.0 for name in names}
        incs      = {name: 1.0 / self.weights[name] for name in names}
        exhausted: set[str] = set()

        while len(exhausted) < len(names):
            active = [(counters[n], n) for n in names if n not in exhausted]
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

            while len(carry) >= self.seq_len:
                chunk = carry[:self.seq_len]
                carry = carry[self.seq_len:]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {
                    "input_ids":      input_ids,
                    "attention_mask": torch.ones_like(input_ids),
                }


# ─────────────────────────────────────────────────────────────────────────────
# Factory pretraining
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(train_cfg, tokenizer: PreTrainedTokenizer):
    """Costruisce train_loader e val_loader per il pretraining v0.4."""
    weights = {
        "fineweb":   train_cfg.fineweb_weight,
        "wikipedia": train_cfg.wikipedia_weight,
        "books":     train_cfg.books_weight,
        "c4":        train_cfg.c4_weight,
        "owt":       train_cfg.owt_weight,
    }

    print("Dataset mix v0.4: " + ", ".join(f"{k} {v:.0%}" for k, v in weights.items()))

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

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg.batch_size,
                              num_workers=0, pin_memory=True)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# SFT helpers
# ─────────────────────────────────────────────────────────────────────────────

def _format_turn(role: str, content: str) -> str:
    role_tag = "User" if role == "user" else "Assistant"
    return f"{role_tag}: {content.strip()}"


def _build_context_and_response(
    messages:      list,
    max_ctx_turns: int,
) -> tuple[str, str]:
    if not messages or messages[-1].get("role") != "assistant":
        return "", ""
    response = messages[-1]["content"].strip()
    if not response:
        return "", ""
    preceding = messages[:-1]
    ctx_turns = preceding[-max_ctx_turns:] if max_ctx_turns > 0 else []
    context   = " ".join(_format_turn(m["role"], m["content"]) for m in ctx_turns)
    return context, response


def _iter_ultrachat(
    tokenizer:      PreTrainedTokenizer,
    split:          str,
    max_ctx_len:    int,
    max_resp_len:   int,
    max_ctx_turns:  int,
    val_every:      int,
    seed:           int,
) -> Iterator[dict]:
    from datasets import load_dataset
    
    def to_list(obj):
        """Convert various types to a list."""
        if obj is None:
            return []
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return list(obj)
        if hasattr(obj, "__iter__"):
            return list(obj)
        return [obj]
    
    def extract_ids(tokenized):
        """Extract input_ids from tokenizer output."""
        if isinstance(tokenized, dict):
            ids = tokenized.get("input_ids", [])
        elif hasattr(tokenized, "input_ids"):
            ids = tokenized.input_ids
        else:
            ids = tokenized if isinstance(tokenized, list) else []
        return to_list(ids)

    ds = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft" if split == "train" else "test_sft",
        streaming=True,
    )
    ds = ds.shuffle(buffer_size=1000, seed=seed)

    doc_idx = 0
    for example in ds:
        # Safely get and convert messages
        messages = to_list(example.get("messages", []))
        
        if len(messages) < 2:
            doc_idx += 1
            continue

        is_val = (doc_idx % val_every == 0)
        if (split == "val") != is_val:
            doc_idx += 1
            continue

        context, response = _build_context_and_response(messages, max_ctx_turns)
        if not response:
            doc_idx += 1
            continue

        prompt_ids = []
        if context:
            tokenized = tokenizer(
                context, truncation=True, max_length=max_ctx_len,
                add_special_tokens=False, return_attention_mask=False,
            )
            prompt_ids = extract_ids(tokenized)

        tokenized_resp = tokenizer(
            response, truncation=True, max_length=max_resp_len,
            add_special_tokens=True, return_attention_mask=False,
        )
        resp_ids = extract_ids(tokenized_resp)

        if len(resp_ids) < 4:  
            doc_idx += 1
            continue

        yield {"prompt_ids": prompt_ids, "response_ids": resp_ids}
        doc_idx += 1


def _iter_openorca(
    tokenizer:      PreTrainedTokenizer,
    split:          str,
    max_ctx_len:    int,
    max_resp_len:   int,
    val_every:      int,
    seed:           int,
) -> Iterator[dict]:
    from datasets import load_dataset
    
    def to_list(obj):
        """Convert various types to a list."""
        if obj is None:
            return []
        if hasattr(obj, "tolist"):
            return obj.tolist()
        if isinstance(obj, (list, tuple)):
            return list(obj)
        if hasattr(obj, "__iter__"):
            return list(obj)
        return [obj]
    
    def extract_ids(tokenized):
        """Extract input_ids from tokenizer output."""
        if isinstance(tokenized, dict):
            ids = tokenized.get("input_ids", [])
        elif hasattr(tokenized, "input_ids"):
            ids = tokenized.input_ids
        else:
            ids = tokenized if isinstance(tokenized, list) else []
        return to_list(ids)

    ds = load_dataset("Open-Orca/OpenOrca", split="train", streaming=True)
    ds = ds.shuffle(buffer_size=1000, seed=seed)

    doc_idx = 0
    for example in ds:
        is_val = (doc_idx % val_every == 0)
        if (split == "val") != is_val:
            doc_idx += 1
            continue

        # Safely get fields and convert to strings
        system = example.get("system_prompt")
        system = str(system).strip() if system is not None else ""
        
        question = example.get("question")
        question = str(question).strip() if question is not None else ""
        
        response = example.get("response")
        response = str(response).strip() if response is not None else ""

        if not question or not response:
            doc_idx += 1
            continue

        context = f"{system} User: {question}".strip() if system else f"User: {question}"

        # Tokenize context
        tokenized_ctx = tokenizer(
            context, truncation=True, max_length=max_ctx_len,
            add_special_tokens=False, return_attention_mask=False,
        )
        prompt_ids = extract_ids(tokenized_ctx)

        # Tokenize response
        tokenized_resp = tokenizer(
            response, truncation=True, max_length=max_resp_len,
            add_special_tokens=True, return_attention_mask=False,
        )
        resp_ids = extract_ids(tokenized_resp)

        if len(resp_ids) < 4:
            doc_idx += 1
            continue

        yield {"prompt_ids": prompt_ids, "response_ids": resp_ids}
        doc_idx += 1


# ─────────────────────────────────────────────────────────────────────────────
# SFTDataset — mix ultrachat + openorca
# ─────────────────────────────────────────────────────────────────────────────

class SFTDataset(IterableDataset):
    """
    Dataset SFT che mixa UltraChat e OpenOrca in streaming.

    Cambiamenti v0.4:
      - padding usa pad_token_id del tokenizer invece di 0
        (GPT-2: pad_token_id = eos_token_id = 50256)
      - response_mask costruita su pad_token_id, non su 0
    """

    DATASET_WEIGHTS = {"ultrachat": 0.5, "openorca": 0.5}

    def __init__(
        self,
        tokenizer:     PreTrainedTokenizer,
        split:         str            = "train",
        max_ctx_len:   int            = 128,
        max_resp_len:  int            = 128,
        max_ctx_turns: int            = 3,
        val_every:     int            = 200,
        seed:          int            = 42,
        weights:       Optional[dict] = None,
    ):
        super().__init__()
        self.tokenizer     = tokenizer
        self.split         = split
        self.max_ctx_len   = max_ctx_len
        self.max_resp_len  = max_resp_len
        self.max_ctx_turns = max_ctx_turns
        self.val_every     = val_every
        self.seed          = seed
        self.weights       = weights or self.DATASET_WEIGHTS

        # Helper function to safely extract a single token ID
        def _first(val):
            if val is None:
                return None
            if isinstance(val, list):
                return val[0] if val else None
            return val

        # [v0.4] pad_token_id per GPT-2 = eos_token_id = 50256
        pad_token = _first(tokenizer.pad_token_id)
        eos_token = _first(tokenizer.eos_token_id)
        
        self.pad_id = int(
            pad_token
            if pad_token is not None
            else eos_token
            if eos_token is not None
            else 0
        )

    def _make_iterators(self) -> dict:
        iters = {}
        if "ultrachat" in self.weights:
            iters["ultrachat"] = _iter_ultrachat(
                self.tokenizer, self.split,
                self.max_ctx_len, self.max_resp_len,
                self.max_ctx_turns, self.val_every, seed=self.seed,
            )
        if "openorca" in self.weights:
            iters["openorca"] = _iter_openorca(
                self.tokenizer, self.split,
                self.max_ctx_len, self.max_resp_len,
                self.val_every, seed=self.seed + 7,
            )
        return iters

    def _pad_left(self, ids: list, length: int) -> torch.Tensor:
        """Padding a sinistra con pad_id."""
        t = torch.full((length,), self.pad_id, dtype=torch.long)
        n = min(len(ids), length)
        t[length - n:] = torch.tensor(ids[-n:], dtype=torch.long)
        return t

    def _pad_right(self, ids: list, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Padding a destra con pad_id. Ritorna (ids_padded, mask)."""
        t    = torch.full((length,), self.pad_id, dtype=torch.long)
        mask = torch.zeros(length, dtype=torch.bool)
        n    = min(len(ids), length)
        t[:n]    = torch.tensor(ids[:n], dtype=torch.long)
        mask[:n] = True
        return t, mask

    def __iter__(self) -> Iterator[dict]:
        iters     = self._make_iterators()
        names     = list(iters.keys())
        counters  = {n: 0.0 for n in names}
        incs      = {n: 1.0 / self.weights[n] for n in names}
        exhausted: set[str] = set()

        while len(exhausted) < len(names):
            active = [(counters[n], n) for n in names if n not in exhausted]
            if not active:
                break
            _, chosen = min(active)

            try:
                item = next(iters[chosen])
                counters[chosen] += incs[chosen]
            except StopIteration:
                exhausted.add(chosen)
                continue

            prompt_ids      = self._pad_left(item["prompt_ids"], self.max_ctx_len)
            resp_ids, mask  = self._pad_right(item["response_ids"], self.max_resp_len)

            yield {
                "prompt_ids":    prompt_ids,
                "response_ids":  resp_ids,
                "response_mask": mask,
            }


# ─────────────────────────────────────────────────────────────────────────────
# Factory SFT
# ─────────────────────────────────────────────────────────────────────────────

def build_sft_loaders(
    train_cfg,
    tokenizer:     PreTrainedTokenizer,
    max_ctx_len:   int = 128,
    max_resp_len:  int = 128,
    max_ctx_turns: int = 3,
) -> tuple[DataLoader, DataLoader]:

    train_ds = SFTDataset(
        tokenizer=tokenizer, split="train",
        max_ctx_len=max_ctx_len, max_resp_len=max_resp_len,
        max_ctx_turns=max_ctx_turns,
        val_every=train_cfg.val_every, seed=42,
    )
    val_ds = SFTDataset(
        tokenizer=tokenizer, split="val",
        max_ctx_len=max_ctx_len, max_resp_len=max_resp_len,
        max_ctx_turns=max_ctx_turns,
        val_every=train_cfg.val_every, seed=42,
    )

    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=train_cfg.batch_size,
                              num_workers=0, pin_memory=True)
    return train_loader, val_loader