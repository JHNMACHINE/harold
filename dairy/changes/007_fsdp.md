# [007] FSDP per training multi-GPU

**Data**: Aprile 2026  
**File modificati**: `utils/fsdp.py` (nuovo), `training/setup.py`, `core/config.py`, `core/context.py`  
**Tag**: `[v0.7-S3]`

## Perché

DDP replica i parametri su ogni GPU — con 3.2B parametri e 8 GPU,
ogni GPU tiene 3.2B in VRAM. FSDP sharda parametri, gradienti e stato
ottimizzatore: ogni GPU tiene solo ~1/N. Per il full run da 100k iter
su 8x H200 è necessario per avere batch size ragionevole.

## Prima

```python
# solo DDP disponibile
active_model = DDP(model, device_ids=[local_rank])
```

## Dopo

```python
# FSDP opzionale via TrainConfig.use_fsdp (default False)
if use_fsdp and is_ddp():
    fsdp_ctx     = FSDPContext().setup()
    active_model = wrap_model_fsdp(model, local_rank, mixed_precision=True)
    # wrap policy: JambaBlock separato, Mamba3Block no (kernel Triton)
```

## Pro

- ~1/N VRAM per GPU — permette batch size molto più grandi
- Compatible con `torch.compile` via `use_orig_params=True`
- Checkpoint bidirezionale: carica sia ckpt FSDP che DDP standard
- Default `False` — retrocompatibile al 100%

## Contro / Trade-off

- Wrap policy specifica: `JambaBlock` wrappato, `Mamba3Block` no
  (i kernel Triton/CUDA richiedono che i parametri SSM siano co-locati)
- `cpu_offload=True` incompatibile con `torch.compile`
- Checkpoint FSDP richiede tutti i rank per il salvataggio
- Non testato in produzione — da validare sul full run multi-GPU

## Risultati attesi

- Su 8x H200: ~17GB VRAM per GPU invece di 140GB
- Batch size effettivo 8x più grande rispetto a single-GPU
- Throughput ~6-7x (overhead comunicazione ~15%)

## Risultati reali

> *Da testare sul full run multi-GPU — single-GPU attuale non usa FSDP*
