# [003] Scaling architettura da 1.5B a 3.2B

**Data**: Aprile 2026  
**File modificati**: `core/config.py`, `core/model.py`  
**Tag**: architettura

## Perché

v0.6 era a 1.5B parametri — troppo piccolo per un'emergenza linguistica
robusta nei diffusion LM. Target: 3B parametri mantenendo la struttura
Jamba e il ratio MoE invariati.

## Prima

```python
d_model=1280, n_layers=36, n_heads=20, n_kv_heads=5, d_ff=3584
ds_moe_n_shared_experts=1, moe_n_routed_experts=8
# divisori hardcoded: shared=d_ff//2, routed=d_ff//4
```

## Dopo

```python
d_model=1792, n_layers=40, n_heads=28, n_kv_heads=7, d_ff=4864
ds_moe_n_shared_experts=2, moe_n_routed_experts=16
moe_routed_hidden=608   # d_ff // 8 — calibrato per ~3.2B totali
moe_shared_hidden=1216  # d_ff // 4
```

## Pro

- Maggiore capacità di rappresentazione (2x parametri)
- 16 expert routed: specializzazione più fine (12.5% pool attivo vs 25%)
- `head_dim=64` invariato — coerenza con Mamba3 headdim
- Divisori MoE espliciti in config — no magic numbers nel codice

## Contro / Trade-off

- VRAM: da ~3GB a ~6GB per i soli pesi
- Overhead ottimizzatore AdamW (momenti fp32): ~25GB aggiuntivi
- Richiede H200 140GB per training confortevole

## Problema scoperto durante l'implementazione

I divisori originali `d_ff//2` e `d_ff//4` producevano **6B invece di 3B**
con d_ff=4864. Il MoE occupava 3.8B su 5.1B totali.

Diagnosi:
```python
moe_params = sum(p.numel() for n,p in m.named_parameters()
                 if any(x in n for x in ['routed','shared','router'])) / 1e6
# → 3837.5M — quasi tutto il budget
```

Fix: calibrare i divisori esplicitamente e leggerli da config.

## Risultati attesi

- Parametri totali: ~3.20B
- Emergenza linguistica più precoce rispetto a 1.5B

## Risultati reali

- Parametri verificati: **3.20B** ✓
- OOM su A100 40GB ✓ (atteso — migrato a H200 NVL)
