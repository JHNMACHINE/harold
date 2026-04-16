"""
dataset.py — Harold v0.7
=========================
Cambiamenti rispetto a v0.6:

  [v0.7-D1] num_workers=0 FISSO e permanente.
             HuggingFace streaming datasets causano SIGABRT con qualsiasi
             valore > 0 perché i processi worker tentano di fare fork
             di oggetti non-picklable (connessioni HTTP, generatori).
             Confermato su v0.5, v0.6, v0.7. Non cambiare.

  [v0.7-D2] Prefetch threading per compensare num_workers=0.
             Un thread background legge e tokenizza i prossimi N batch
             mentre la GPU elabora quello corrente.
             Evita che il main thread blocchi il training loop.

  [v0.7-D3] Val/train split PRIMA della tokenizzazione.
             In v0.6 ogni documento veniva tokenizzato anche se doveva
             andare nel val set — spreco CPU nel train loader e viceversa.
             Ora: controlla doc_idx % val_every PRIMA di tokenizzare.

  [v0.7-D4] persistent_workers rimosso (sempre False con num_workers=0).

Invariato da v0.6:
  [OPT-D2] deque + popleft O(1) invece di list slicing
  [OPT-D5] worker_init_fn a livello modulo per Python 3.14
"""

from __future__ import annotations
from collections import deque
from pathlib import Path
from queue import Queue, Empty
from threading import Thread
from typing import Iterator, Optional
import os
import functools
import yaml
import torch
from transformers import BatchEncoding, PreTrainedTokenizer
from torch.utils.data import IterableDataset, DataLoader


# ─────────────────────────────────────────────────────────────────────────────
# Helpers generici
# ─────────────────────────────────────────────────────────────────────────────

def _to_list(obj) -> list:
    if obj is None:
        return []
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return list(obj)
    if hasattr(obj, "__iter__"):
        return list(obj)
    return [obj]


def _extract_ids(tokenized) -> list[int]:
    if isinstance(tokenized, (dict, BatchEncoding)):
        ids = tokenized.get("input_ids", [])
    elif hasattr(tokenized, "input_ids"):
        ids = tokenized.input_ids
    else:
        ids = tokenized if isinstance(tokenized, list) else []
    return [int(x) for x in _to_list(ids)]


def _first_token_id(val) -> int | None:
    if val is None:
        return None
    if isinstance(val, list):
        return int(val[0]) if val else None
    return int(val)


# ─────────────────────────────────────────────────────────────────────────────
# Caricamento configurazione YAML
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset_config(yaml_path: str = "datasets_config.yaml") -> dict:
    path = Path(yaml_path)
    if not path.is_absolute() and not path.exists():
        path = Path(__file__).parent / yaml_path
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset config non trovato: {yaml_path}\n"
            f"Cercato in: {path.resolve()}"
        )
    with open(path) as f:
        cfg = yaml.safe_load(f)

    for section in ("pretraining", "sft"):
        if section in cfg:
            total = sum(d["weight"] for d in cfg[section])
            assert abs(total - 1.0) < 0.01, \
                f"Pesi {section} devono sommare a 1.0, trovato {total:.3f}"

    return cfg


# ─────────────────────────────────────────────────────────────────────────────
# Estrazione testo
# ─────────────────────────────────────────────────────────────────────────────

def _extract_text(example: dict, text_field: str, fmt: str = "standard") -> str:
    if fmt == "codecontests":
        desc     = (example.get("description") or "").strip()
        sols     = example.get("solutions") or {}
        sol_list = sols.get("solution", []) if isinstance(sols, dict) else []
        first    = sol_list[0].strip() if sol_list else ""
        return f"{desc}\n\n{first}".strip() if first else desc
    return (example.get(text_field) or "").strip()


def _tokenize_doc(text: str, tokenizer: PreTrainedTokenizer, sep_id: int) -> list[int]:
    tokenized = tokenizer(
        text, add_special_tokens=False,
        truncation=False, return_attention_mask=False,
    )
    return _extract_ids(tokenized) + [sep_id]


# ─────────────────────────────────────────────────────────────────────────────
# [v0.7-D2] Prefetch thread — compensa num_workers=0
# ─────────────────────────────────────────────────────────────────────────────

_SENTINEL = object()


class _PrefetchIter:
    """
    Wrappa un iteratore e preleva i prossimi ``n`` elementi in un thread
    background, tenendoli in una Queue. Il main thread consuma dalla Queue
    senza bloccarsi sulla tokenizzazione.
    """

    def __init__(self, source_iter: Iterator, n: int = 32):
        self._q: Queue = Queue(maxsize=n)
        self._t = Thread(target=self._fill, args=(source_iter,), daemon=True)
        self._t.start()

    def _fill(self, it: Iterator) -> None:
        try:
            for item in it:
                self._q.put(item)
        finally:
            self._q.put(_SENTINEL)

    def __iter__(self):
        while True:
            item = self._q.get()
            if item is _SENTINEL:
                return
            yield item


# ─────────────────────────────────────────────────────────────────────────────
# Iteratore pretraining
# ─────────────────────────────────────────────────────────────────────────────

def _iter_pretraining_dataset(
    ds_cfg:      dict,
    tokenizer:   PreTrainedTokenizer,
    sep_id:      int,
    split:       str,
    val_every:   int,
    buffer_size: int,
    seed:        int,
) -> Iterator[list[int]]:
    from datasets import load_dataset

    load_kwargs: dict = {"path": ds_cfg["path"], "split": ds_cfg["split"]}
    if "config" in ds_cfg:
        load_kwargs["name"] = ds_cfg["config"]

    text_field = ds_cfg.get("text_field", "text")
    fmt        = ds_cfg.get("format", "standard")

    # [v0.7-D5] Loop infinito — evita che StopIteration risalga a MixedStreamingDataset
    # e causi la ricreazione dell'iteratore con relativa chiamata HTTP HuggingFace
    # (risoluzione file shard) che blocca la GPU per secondi.
    # Il dataset viene ricaricato con seed incrementato per variare l'ordine.
    epoch = 0
    while True:
        ds = load_dataset(**load_kwargs, streaming=True)
        ds = ds.shuffle(buffer_size=buffer_size, seed=seed + epoch * 31337)

        # [v0.7-D3] Controlla split PRIMA di tokenizzare
        doc_idx = 0
        for example in ds:
            is_val = (doc_idx % val_every == 0)
            if (split == "val") != is_val:
                doc_idx += 1
                continue
            text = _extract_text(example, text_field, fmt)
            if not text:
                doc_idx += 1
                continue
            ids = _tokenize_doc(text, tokenizer, sep_id)
            if ids:
                yield ids
            doc_idx += 1

        epoch += 1


# ─────────────────────────────────────────────────────────────────────────────
# Iteratore SFT
# ─────────────────────────────────────────────────────────────────────────────

def _format_turn(role: str, content: str) -> str:
    return f"{'User' if role == 'user' else 'Assistant'}: {content.strip()}"


def _iter_sft_dataset(
    ds_cfg:        dict,
    tokenizer:     PreTrainedTokenizer,
    split:         str,
    max_ctx_len:   int,
    max_resp_len:  int,
    val_every:     int,
    seed:          int,
    max_ctx_turns: int = 3,
) -> Iterator[dict]:
    from datasets import load_dataset

    structure = ds_cfg.get("structure", "pairs")
    fields    = ds_cfg.get("fields", {})

    split_map = ds_cfg.get("split_map", {})
    hf_split  = split_map.get(split, ds_cfg.get("split", "train"))

    load_kwargs: dict = {"path": ds_cfg["path"], "split": hf_split}
    if "config" in ds_cfg:
        load_kwargs["name"] = ds_cfg["config"]

    ds = load_dataset(**load_kwargs, streaming=True)
    ds = ds.shuffle(buffer_size=1000, seed=seed)

    def tok_ctx(text: str) -> list[int]:
        return _extract_ids(tokenizer(
            text, truncation=True, max_length=max_ctx_len,
            add_special_tokens=False, return_attention_mask=False,
        )) if text else []

    def tok_resp(text: str) -> list[int]:
        return _extract_ids(tokenizer(
            text, truncation=True, max_length=max_resp_len,
            add_special_tokens=True, return_attention_mask=False,
        ))

    def emit(doc_idx: int, context: str, response: str):
        is_val = (doc_idx % val_every == 0)
        if (split == "val") != is_val:
            return None
        resp_ids = tok_resp(response)
        if len(resp_ids) < 4:
            return None
        return {"prompt_ids": tok_ctx(context), "response_ids": resp_ids}

    if structure == "pairs":
        ctx_fields = fields.get("context", [])
        resp_field = fields.get("response", "response")
        separator  = fields.get("separator", " ")
        if isinstance(ctx_fields, str):
            ctx_fields = [ctx_fields]
        doc_idx = 0
        for example in ds:
            parts    = [str(example.get(f) or "").strip() for f in ctx_fields]
            context  = separator.join(p for p in parts if p)
            response = str(example.get(resp_field) or "").strip()
            if not response:
                doc_idx += 1
                continue
            result = emit(doc_idx, context, response)
            if result:
                yield result
            doc_idx += 1

    elif structure == "ranked_pairs":
        role_f    = fields.get("role", "role")
        text_f    = fields.get("text", "text")
        rank_f    = fields.get("rank", "rank")
        prompter  = fields.get("prompter_role", "prompter")
        assistant = fields.get("assistant_role", "assistant")
        doc_idx   = 0
        buffer: list[dict] = []
        for example in ds:
            role = str(example.get(role_f) or "")
            text = str(example.get(text_f) or "").strip()
            if not text or example.get(rank_f, 0) != 0:
                continue
            buffer.append({"role": role, "text": text})
            if (len(buffer) >= 2
                    and buffer[-2]["role"] == prompter
                    and buffer[-1]["role"] == assistant):
                result = emit(doc_idx, buffer[-2]["text"], buffer[-1]["text"])
                if result:
                    yield result
                doc_idx += 1
                buffer = []

    elif structure == "multiturn":
        msg_f   = fields.get("messages", "messages")
        role_f  = fields.get("role_field", "role")
        cont_f  = fields.get("content_field", "content")
        asst    = fields.get("assistant_role", "assistant")
        doc_idx = 0
        for example in ds:
            messages = _to_list(example.get(msg_f, []))
            if len(messages) < 2 or messages[-1].get(role_f) != asst:
                doc_idx += 1
                continue
            response  = str(messages[-1].get(cont_f) or "").strip()
            ctx_turns = messages[:-1][-max_ctx_turns:]
            context   = " ".join(
                _format_turn(m.get(role_f, ""), str(m.get(cont_f) or ""))
                for m in ctx_turns
            )
            if not response:
                doc_idx += 1
                continue
            result = emit(doc_idx, context, response)
            if result:
                yield result
            doc_idx += 1
    else:
        raise ValueError(
            f"Struttura SFT sconosciuta: {structure!r}\n"
            f"Supportate: pairs, ranked_pairs, multiturn"
        )


# ─────────────────────────────────────────────────────────────────────────────
# MixedStreamingDataset — pretraining
# ─────────────────────────────────────────────────────────────────────────────

class MixedStreamingDataset(IterableDataset):
    """Dataset di pretraining guidato da YAML."""

    def __init__(
        self,
        tokenizer:    PreTrainedTokenizer,
        seq_len:      int,
        dataset_cfg:  list[dict],
        split:        str = "train",
        val_every:    int = 200,
        seed:         int = 42,
        buffer_size:  int = 1000,
        prefetch_n:   int = 32,
    ):
        super().__init__()
        self.tokenizer   = tokenizer
        self.seq_len     = seq_len
        self.dataset_cfg = dataset_cfg
        self.split       = split
        self.val_every   = val_every
        self.seed        = seed
        self.buffer_size = buffer_size
        self.prefetch_n  = prefetch_n
        self.weights     = {d["name"]: d["weight"] for d in dataset_cfg}
        sep_token = _first_token_id(tokenizer.sep_token_id)
        eos_token = _first_token_id(tokenizer.eos_token_id)
        self.sep_id = int(sep_token if sep_token is not None
                          else eos_token if eos_token is not None
                          else 0)

    def _make_iterators(self) -> dict[str, Iterator[list[int]]]:
        return {
            d["name"]: _iter_pretraining_dataset(
                ds_cfg=d, tokenizer=self.tokenizer, sep_id=self.sep_id,
                split=self.split, val_every=self.val_every,
                buffer_size=self.buffer_size, seed=self.seed + i * 7,
            )
            for i, d in enumerate(self.dataset_cfg)
        }

    def __iter__(self) -> Iterator[dict]:
        raw_iters = self._make_iterators()

        # [v0.7-D2] Wrappa ogni iteratore con prefetch thread
        iters = {
            name: _PrefetchIter(it, n=self.prefetch_n)
            for name, it in raw_iters.items()
        }

        names    = list(iters.keys())
        carry: deque[int] = deque()
        counters = {n: 0.0 for n in names}
        incs     = {n: 1.0 / self.weights[n] for n in names}
        exhausted: set[str] = set()

        while len(exhausted) < len(names):
            active = [(counters[n], n) for n in names if n not in exhausted]
            if not active:
                break
            _, chosen = min(active)
            try:
                ids = next(iter(iters[chosen]))
                counters[chosen] += incs[chosen]
            except StopIteration:
                exhausted.add(chosen)
                continue
            carry.extend(ids)

            while len(carry) >= self.seq_len:
                chunk = [carry.popleft() for _ in range(self.seq_len)]
                input_ids = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": input_ids}


# ─────────────────────────────────────────────────────────────────────────────
# SFTDataset
# ─────────────────────────────────────────────────────────────────────────────

class SFTDataset(IterableDataset):
    """Dataset SFT guidato da YAML."""

    def __init__(
        self,
        tokenizer:     PreTrainedTokenizer,
        dataset_cfg:   list[dict],
        split:         str = "train",
        max_ctx_len:   int = 128,
        max_resp_len:  int = 128,
        max_ctx_turns: int = 3,
        val_every:     int = 200,
        seed:          int = 42,
        prefetch_n:    int = 16,
    ):
        super().__init__()
        self.tokenizer     = tokenizer
        self.dataset_cfg   = dataset_cfg
        self.split         = split
        self.max_ctx_len   = max_ctx_len
        self.max_resp_len  = max_resp_len
        self.max_ctx_turns = max_ctx_turns
        self.val_every     = val_every
        self.seed          = seed
        self.prefetch_n    = prefetch_n
        self.weights       = {d["name"]: d["weight"] for d in dataset_cfg}
        pad = _first_token_id(tokenizer.pad_token_id)
        eos = _first_token_id(tokenizer.eos_token_id)
        self.pad_id = int(pad if pad is not None else eos if eos is not None else 0)

    def _make_iterators(self) -> dict:
        return {
            ds_cfg["name"]: _iter_sft_dataset(
                ds_cfg=ds_cfg, tokenizer=self.tokenizer, split=self.split,
                max_ctx_len=self.max_ctx_len, max_resp_len=self.max_resp_len,
                val_every=self.val_every, seed=self.seed + i * 7,
                max_ctx_turns=self.max_ctx_turns,
            )
            for i, ds_cfg in enumerate(self.dataset_cfg)
        }

    def _pad_left(self, ids: list, length: int) -> torch.Tensor:
        t = torch.full((length,), self.pad_id, dtype=torch.long)
        n = min(len(ids), length)
        t[length - n:] = torch.tensor(ids[-n:], dtype=torch.long)
        return t

    def _pad_right(self, ids: list, length: int) -> tuple[torch.Tensor, torch.Tensor]:
        t    = torch.full((length,), self.pad_id, dtype=torch.long)
        mask = torch.zeros(length, dtype=torch.bool)
        n    = min(len(ids), length)
        t[:n]    = torch.tensor(ids[:n], dtype=torch.long)
        mask[:n] = True
        return t, mask

    def __iter__(self) -> Iterator[dict]:
        raw_iters = self._make_iterators()
        iters = {
            name: _PrefetchIter(it, n=self.prefetch_n)
            for name, it in raw_iters.items()
        }

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
                item = next(iter(iters[chosen]))
                counters[chosen] += incs[chosen]
            except StopIteration:
                exhausted.add(chosen)
                continue
            prompt_ids     = self._pad_left(item["prompt_ids"], self.max_ctx_len)
            resp_ids, mask = self._pad_right(item["response_ids"], self.max_resp_len)
            yield {"prompt_ids": prompt_ids, "response_ids": resp_ids, "response_mask": mask}


# ─────────────────────────────────────────────────────────────────────────────
# Worker init functions — a livello di modulo per serializzabilità Python 3.14
# ─────────────────────────────────────────────────────────────────────────────

def worker_init_fn(worker_id: int) -> None:
    """Partiziona lo stream tra i worker per evitare duplicati."""
    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        ds = worker_info.dataset
        current_seed = getattr(ds, "seed", 42)
        setattr(ds, "seed", current_seed + worker_info.id * 31)


def _worker_init_fn_ddp(worker_id: int, rank_seed: int = 42) -> None:
    """[OPT-D5] worker_init_fn serializzabile per DDP."""
    wi = torch.utils.data.get_worker_info()
    if wi is not None:
        ds = wi.dataset
        setattr(ds, "seed", getattr(ds, "seed", rank_seed) + wi.id * 31)


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────

def build_loaders(
    train_cfg,
    tokenizer:  PreTrainedTokenizer,
    yaml_path:  str = "datasets_config.yaml",
) -> tuple[DataLoader, DataLoader]:
    cfg         = load_dataset_config(yaml_path)
    dataset_cfg = cfg["pretraining"]
    print("Dataset mix pretraining: " +
          ", ".join(f"{d['name']} {d['weight']:.0%}" for d in dataset_cfg))

    train_ds = MixedStreamingDataset(
        tokenizer=tokenizer, seq_len=train_cfg.seq_len, dataset_cfg=dataset_cfg,
        split="train", val_every=train_cfg.val_every,
        seed=42, buffer_size=train_cfg.stream_buffer_size,
        prefetch_n=32,
    )
    val_ds = MixedStreamingDataset(
        tokenizer=tokenizer, seq_len=train_cfg.seq_len, dataset_cfg=dataset_cfg,
        split="val", val_every=train_cfg.val_every,
        seed=42, buffer_size=train_cfg.stream_buffer_size,
        prefetch_n=8,
    )

    # [v0.7-D1] num_workers=0 FISSO — HF streaming causa SIGABRT con fork
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
    return train_loader, val_loader


def build_sft_loaders(
    train_cfg,
    tokenizer:     PreTrainedTokenizer,
    max_ctx_len:   int = 128,
    max_resp_len:  int = 128,
    max_ctx_turns: int = 3,
    yaml_path:     str = "datasets_config.yaml",
) -> tuple[DataLoader, DataLoader]:
    cfg         = load_dataset_config(yaml_path)
    dataset_cfg = cfg["sft"]
    print("Dataset mix SFT: " +
          ", ".join(f"{d['name']} {d['weight']:.0%}" for d in dataset_cfg))

    train_ds = SFTDataset(
        tokenizer=tokenizer, dataset_cfg=dataset_cfg, split="train",
        max_ctx_len=max_ctx_len, max_resp_len=max_resp_len,
        max_ctx_turns=max_ctx_turns, val_every=train_cfg.val_every,
        seed=42, prefetch_n=16,
    )
    val_ds = SFTDataset(
        tokenizer=tokenizer, dataset_cfg=dataset_cfg, split="val",
        max_ctx_len=max_ctx_len, max_resp_len=max_resp_len,
        max_ctx_turns=max_ctx_turns, val_every=train_cfg.val_every,
        seed=42, prefetch_n=4,
    )

    # [v0.7-D1] num_workers=0 FISSO
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size,
        num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader


def build_loaders_ddp(
    train_cfg,
    tokenizer:  PreTrainedTokenizer,
    rank:       int,
    yaml_path:  str = "datasets_config.yaml",
) -> tuple[DataLoader, DataLoader]:
    """
    Costruisce i DataLoader partizionati per rank DDP/FSDP.

    Con IterableDataset streaming il partizionamento avviene tramite seed
    offset per rank — non esiste DistributedSampler per dataset infiniti.
      rank 0: seed=42
      rank 1: seed=1042
      ...
    """
    cfg         = load_dataset_config(yaml_path)
    dataset_cfg = cfg["pretraining"]

    rank_seed = 42 + rank * 1000

    train_ds = MixedStreamingDataset(
        tokenizer=tokenizer, seq_len=train_cfg.seq_len, dataset_cfg=dataset_cfg,
        split="train", val_every=train_cfg.val_every,
        seed=rank_seed, buffer_size=train_cfg.stream_buffer_size,
        prefetch_n=32,
    )
    val_ds = MixedStreamingDataset(
        tokenizer=tokenizer, seq_len=train_cfg.seq_len, dataset_cfg=dataset_cfg,
        split="val", val_every=train_cfg.val_every,
        seed=rank_seed, buffer_size=train_cfg.stream_buffer_size,
        prefetch_n=8,
    )

    # [v0.7-D1] num_workers=0 FISSO anche in DDP/FSDP
    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size,
        num_workers=0, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size,
        num_workers=0, pin_memory=True,
    )
    return train_loader, val_loader