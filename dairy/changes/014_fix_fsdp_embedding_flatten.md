# [010] Fix FSDP embedding flatten crash

**Data**: Aprile 2026  
**File modificati**: `utils/fsdp.py`  
**Tag**: `[v0.7-F1]`

## Perché

Il training crashava con:
```
RuntimeError: 'weight' must be 2-D
```
su `self.token_emb(x0)` in `compute_loss`. La wrap policy FSDP wrappava
solo `JambaBlock` — l'embedding `token_emb` finiva nel FSDP root module,
i cui parametri vengono flattenati in un unico `FlatParameter` 1D.
`torch.nn.functional.embedding` richiede un peso 2D `(vocab_size, d_model)`.

## Prima

```python
def _get_Harold_wrap_policy():
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from core.model.blocks import JambaBlock
    return ModuleWrapPolicy({JambaBlock})
```

## Dopo

```python
def _get_Harold_wrap_policy():
    from torch.distributed.fsdp.wrap import ModuleWrapPolicy
    from core.model.blocks import JambaBlock
    return ModuleWrapPolicy({JambaBlock, nn.Embedding})
```

## Pro

- L'embedding mantiene la sua shape 2D durante il forward
- `nn.Embedding` è un modulo standard PyTorch — nessun rischio di incompatibilità
- Shardato separatamente: l'embedding (vocab_size × d_model = 32k × 1792 ≈ 115M params) viene distribuito tra le GPU

## Contro / Trade-off

- Un FSDP unit in più (41 unità wrappate invece di 40) — overhead trascurabile
- Se Harold avesse altri `nn.Embedding` (non ne ha) verrebbero wrappati anche quelli

## Risultati attesi

- `compute_loss` esegue `token_emb(x0)` senza crash
- Training loop FSDP parte e produce loss valide

## Risultati reali

> *Da aggiornare dopo il training*
