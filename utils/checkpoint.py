"""
Harold v0.6 — checkpoint.py
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

Upload HuggingFace:
  .pt e .safetensors vengono caricati in parallelo su thread separati.
"""

import glob
import os
import threading
from collections import deque
from typing import Union

import torch
import torch.nn as nn
from safetensors.torch import save_model

from core.config import HF_FILENAME, HF_REPO_ID

def ensure_disk_space(
    required_gb:    float = 20.0,
    checkpoint_dir: str   = ".",
    stage:          "int | None" = None,
    prefix:         str   = "",
) -> None:
    """
    Verifica spazio disco prima di salvare. Se insufficiente,
    cancella checkpoint periodici vecchi e file .tmp corrotti.
    """
    import shutil
 
    def free_gb() -> float:
        return shutil.disk_usage(checkpoint_dir).free / (1024 ** 3)
 
    if free_gb() >= required_gb:
        return
 
    print(f"  WARNING: spazio disco insufficiente ({free_gb():.1f}GB liberi, richiesti {required_gb:.1f}GB)")
 
    pat = (os.path.join(checkpoint_dir, f"{prefix}_s{stage}_[0-9]*.pt")
           if stage is not None else
           os.path.join(checkpoint_dir, f"{prefix}_[0-9]*.pt"))
 
    for ckpt in sorted(glob.glob(pat)):
        if free_gb() >= required_gb:
            break
        sz = os.path.getsize(ckpt) / (1024 ** 3)
        os.remove(ckpt)
        print(f"  Liberato {sz:.1f}GB — rimosso {os.path.basename(ckpt)}")
 
    for tmp in glob.glob(os.path.join(checkpoint_dir, "*.tmp")):
        if free_gb() >= required_gb:
            break
        sz = os.path.getsize(tmp) / (1024 ** 3)
        os.remove(tmp)
        print(f"  Liberato {sz:.1f}GB — rimosso {os.path.basename(tmp)}")
 
    print(f"  Spazio disponibile dopo cleanup: {free_gb():.1f}GB")

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


def _upload_pt(path: str, api: object, hf_repo_id: str, hf_token: str) -> None:
    """Upload del checkpoint .pt completo su HuggingFace."""
    try:
        api.upload_file(                                     # type: ignore[attr-defined]
            path_or_fileobj=path,
            path_in_repo=HF_FILENAME,
            repo_id=hf_repo_id, repo_type="model", token=hf_token,
        )
        print(f"  OK HuggingFace -> {hf_repo_id}/{HF_FILENAME}")
    except BaseException as e:
        print(f"  WARNING Errore upload .pt: {e}")


def _upload_safetensors(
    path: str, model: nn.Module, api: object, hf_repo_id: str, hf_token: str,
) -> None:
    """Salva e carica i pesi in .safetensors su HuggingFace, poi rimuove il file locale."""
    try:
        sf_filename = HF_FILENAME.replace(".pt", ".safetensors")
        sf_path     = path.replace(".pt", ".safetensors")
        save_model(model, sf_path)
        api.upload_file(                                     # type: ignore[attr-defined]
            path_or_fileobj=sf_path,
            path_in_repo=sf_filename,
            repo_id=hf_repo_id, repo_type="model", token=hf_token,
        )
        os.remove(sf_path)
        print(f"  OK HuggingFace -> {hf_repo_id}/{sf_filename}")
    except BaseException as e:
        print(f"  WARNING Errore upload .safetensors: {e}")


def _push_to_hf(path: str, model: nn.Module, wait: bool) -> None:
    """
    Upload su HuggingFace in thread separati (uno per .pt, uno per .safetensors).
    I due upload avvengono in parallelo per sfruttare i core CPU disponibili.
    """
    def _push() -> None:
        try:
            from huggingface_hub import HfApi
            hf_token = os.environ.get("HF_TOKEN")
            if not hf_token:
                print("  WARNING HF_TOKEN non trovato -- skip push HuggingFace.")
                return

            api = HfApi()
            api.create_repo(
                repo_id=HF_REPO_ID, repo_type="model",
                exist_ok=True, token=hf_token,
            )

            # Lancia i due upload in parallelo
            t_pt = threading.Thread(
                target=_upload_pt,
                args=(path, api, HF_REPO_ID, hf_token),
                daemon=True,
            )
            t_sf = threading.Thread(
                target=_upload_safetensors,
                args=(path, model, api, HF_REPO_ID, hf_token),
                daemon=True,
            )
            t_pt.start()
            t_sf.start()
            t_pt.join()
            t_sf.join()

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

    push_hf / wait_hf -> upload HuggingFace opzionale (.pt e .safetensors in parallelo)
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

    _dir = os.path.dirname(path) or "."
    _pfx = (os.path.basename(path).split(f"_s{stage}_")[0] if stage is not None
            else "_".join(os.path.basename(path).replace(".pt", "").split("_")[:-1]))
    ensure_disk_space(required_gb=20.0, checkpoint_dir=_dir, stage=stage, prefix=_pfx)

    tmp = path + ".tmp"
    torch.save(ckpt, tmp)
    os.replace(tmp, path)

    if not full:
        # Usa il prefix senza il numero di iterazione
        # es. "harold_v06_sft_s1_0001000.pt" -> prefix="harold_v06_sft", stage=1
        # es. "harold_v06_0001000.pt"         -> prefix="harold_v06",     stage=None
        basename = os.path.basename(path)
        if stage is not None:
            # Rimuovi "_s{stage}_{iter}.pt" dalla fine
            prefix = basename.split(f"_s{stage}_")[0]
        else:
            # Rimuovi "_{iter}.pt" dalla fine
            prefix = "_".join(basename.replace(".pt", "").split("_")[:-1])
        cleanup_old_checkpoints(
            os.path.dirname(path), prefix,
            stage=stage, keep_last=2,
        )

    if push_hf:
        _push_to_hf(path, model, wait=wait_hf)


def _default_result(load_stage: bool) -> tuple:
    """Risultato di default per training da zero."""
    if load_stage:
        return 1, 0, float("inf"), [], []
    return 0, float("inf"), [], []


def _load_from_hf(path: str) -> bool:
    """
    Scarica il checkpoint da HuggingFace e lo salva in path.
    Ritorna True se il download è riuscito.
    """
    try:
        from huggingface_hub import hf_hub_download
        hf_token = os.environ.get("HF_TOKEN")
        print(f"  Checkpoint locale non trovato — scarico da {HF_REPO_ID}/{HF_FILENAME}...")
        downloaded = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            token=hf_token,
            local_dir=os.path.dirname(path) or ".",
        )
        if downloaded != path:
            import shutil
            shutil.move(downloaded, path)
        print(f"  Scaricato -> {path}")
        return True
    except Exception as e:
        print(f"  Download HuggingFace fallito: {e}")
        return False


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

    Ordine di ricerca:
      1. File locale in path
      2. Download da HuggingFace (HF_REPO_ID/HF_FILENAME)
      3. Riparte da zero

    load_stage=False -> (iter_num, best_val, train_losses, val_losses)
    load_stage=True  -> (stage, iter_num, best_val, train_losses, val_losses)
    """
    if not os.path.isfile(path):
        if not _load_from_hf(path):
            print("  Nessun checkpoint trovato — parto da zero.")
            return _default_result(load_stage)

    print(f"Carico checkpoint: {path}")
    state = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])

    if "optimizer_state" in state:
        optimizer.load_state_dict(state["optimizer_state"])
        print("  optimizer state caricato")
    else:
        print("  optimizer state assente (checkpoint periodico) — riparte da zero")

    if "scaler_state" in state:
        scaler.load_state_dict(state["scaler_state"])

    iter_num     = state.get("iter_num", 0)
    best_val     = state.get("val_loss", float("inf"))
    train_losses = state.get("train_losses", [])
    val_losses   = state.get("val_losses", [])
    stage        = state.get("stage", 1)
    del state

    if load_stage:
        return stage, iter_num, best_val, train_losses, val_losses
    return iter_num, best_val, train_losses, val_losses