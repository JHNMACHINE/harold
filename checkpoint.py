"""
Harold v0.5 — checkpoint.py
============================
Gestione centralizzata dei checkpoint per pretraining e SFT.

API unificata:
  save_checkpoint(...)  — salva checkpoint pretraining o SFT
  load_checkpoint(...)  — carica checkpoint pretraining o SFT
  cleanup_old_checkpoints(...)  — rimuove checkpoint periodici vecchi

Pretraining:
  save_checkpoint(..., stage=None, push_hf=True, wait_hf=True)
  iter_num, best_val, train_losses, val_losses = load_checkpoint(...)

SFT:
  save_checkpoint(..., stage=1)
  stage, iter_num, best_val, train_losses, val_losses = load_checkpoint(..., load_stage=True)
"""

import glob
import os
import threading
from collections import deque
from typing import Union

import torch
import torch.nn as nn


def cleanup_old_checkpoints(
    checkpoint_dir:    str,
    checkpoint_prefix: str,
    stage:             "int | None" = None,
    keep_last:         int = 2,
) -> None:
    """
    Mantiene solo gli ultimi `keep_last` checkpoint periodici.

    stage=None -> pattern: {prefix}_[0-9]*.pt           (pretraining)
    stage=int  -> pattern: {prefix}_s{stage}_[0-9]*.pt  (SFT)
    """
    if stage is None:
        pattern = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_[0-9]*.pt")
    else:
        pattern = os.path.join(checkpoint_dir, f"{checkpoint_prefix}_s{stage}_[0-9]*.pt")

    checkpoints = sorted(glob.glob(pattern))
    for old in checkpoints[:-keep_last]:
        os.remove(old)
        print(f"  Rimosso checkpoint vecchio: {os.path.basename(old)}")


def _push_to_hf(
    path:       str,
    model:      nn.Module,
    wait:       bool,
) -> None:
    """
    Upload su HuggingFace in thread separato.
    Carica sia il checkpoint .pt completo che i pesi in .safetensors.
    """
    def _push() -> None:
        try:
            from huggingface_hub import HfApi
            from safetensors.torch import save_file
            from config import HF_REPO_ID, HF_FILENAME
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                print("  WARNING HF_TOKEN non trovato -- skip push HuggingFace.")
                return
            api = HfApi()
            api.create_repo(repo_id=HF_REPO_ID, repo_type="model",
                            exist_ok=True, token=hf_token)

            # Upload checkpoint .pt completo (con optimizer, config, losses)
            api.upload_file(
                path_or_fileobj=path,
                path_in_repo=HF_FILENAME,
                repo_id=HF_REPO_ID, repo_type="model", token=hf_token,
            )
            print(f"  OK HuggingFace -> {HF_REPO_ID}/{HF_FILENAME}")

            # Upload pesi in .safetensors (solo model state, memory-mappable)
            sf_filename = HF_FILENAME.replace(".pt", ".safetensors")
            sf_path     = path.replace(".pt", ".safetensors")
            save_file(model.state_dict(), sf_path)
            api.upload_file(
                path_or_fileobj=sf_path,
                path_in_repo=sf_filename,
                repo_id=HF_REPO_ID, repo_type="model", token=hf_token,
            )
            os.remove(sf_path)
            print(f"  OK HuggingFace -> {HF_REPO_ID}/{sf_filename}")
        except BaseException as e:
            print(f"  WARNING Errore push HuggingFace: {e}")

    t = threading.Thread(target=_push, daemon=True)
    t.start()
    if wait:
        print("  Attendo upload HuggingFace...")
        t.join()


def save_checkpoint(
    path:         str,
    model:        nn.Module,
    optimizer:    torch.optim.Optimizer,
    scaler:       torch.GradScaler,
    iter_num:     int,
    val_loss:     float,
    model_cfg:    object,
    cfg:          object,
    train_losses: Union[deque, list],
    val_losses:   list,
    full:         bool = True,
    stage:        "int | None" = None,
    push_hf:      bool = False,
    wait_hf:      bool = False,
) -> None:
    """
    Salva un checkpoint (pretraining o SFT).

    full=True  -> include optimizer + scaler state (best/final)
    full=False -> solo model state (periodici) + cleanup automatico

    stage=None -> pretraining (cfg = TrainConfig, chiave "train_cfg")
    stage=int  -> SFT (cfg = SFTConfig, chiave "sft_cfg", aggiunge "stage")

    push_hf / wait_hf -> upload HuggingFace opzionale
    """
    ckpt: dict = {
        "iter_num":     iter_num,
        "model_state":  model.state_dict(),
        "val_loss":     val_loss,
        "model_cfg":    model_cfg,
        "train_losses": list(train_losses),
        "val_losses":   val_losses,
    }

    if stage is None:
        ckpt["train_cfg"] = cfg
    else:
        ckpt["sft_cfg"] = cfg
        ckpt["stage"]   = stage

    if full:
        ckpt["optimizer_state"] = optimizer.state_dict()
        ckpt["scaler_state"]    = scaler.state_dict()

    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)

    if not full:
        prefix = os.path.basename(path).rsplit("_", 1)[0]
        cleanup_old_checkpoints(
            os.path.dirname(path), prefix,
            stage=stage, keep_last=2,
        )

    if push_hf:
        _push_to_hf(path, model, wait=wait_hf)


def load_checkpoint(
    path:       str,
    model:      nn.Module,
    optimizer:  torch.optim.Optimizer,
    scaler:     torch.GradScaler,
    device:     str,
    load_stage: bool = False,
) -> tuple:
    """
    Carica un checkpoint (pretraining o SFT).

    load_stage=False -> (iter_num, best_val, train_losses, val_losses)
    load_stage=True  -> (stage, iter_num, best_val, train_losses, val_losses)
    """
    print(f"Carico checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])

    if "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
        print("  optimizer state caricato")
    else:
        print("  optimizer state assente (checkpoint periodico) -- riparte da zero")

    if "scaler_state" in state:
        scaler.load_state_dict(state["scaler_state"])

    iter_num     = state.get("iter_num", 0) + 1
    best_val     = state.get("val_loss", float("inf"))
    train_losses = state.get("train_losses", [])
    val_losses   = state.get("val_losses", [])
    stage        = state.get("stage", 1)
    del state

    if load_stage:
        return stage, iter_num, best_val, train_losses, val_losses
    return iter_num, best_val, train_losses, val_losses