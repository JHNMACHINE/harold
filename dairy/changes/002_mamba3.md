# [002] Mamba2Block → Mamba3Block

**Data**: Aprile 2026  
**File modificati**: `core/model.py`, `core/config.py`  
**Tag**: `[v0.7-M3]`

## Perché

Mamba3 (Lahoti et al., ICLR 2026) introduce tre miglioramenti su Mamba2:
discretizzazione exponential-trapezoidal più espressiva, state update
complex-valued che risolve il parity problem, e MIMO per più espressività
senza aumentare la latenza di decoding.

## Prima

```python
from mamba_ssm import Mamba2
self.mamba = Mamba2(
    d_model = config.d_model,
    d_state = config.mamba_d_state,
    d_conv  = config.mamba_d_conv,   # rimosso in v0.7
    expand  = config.mamba_expand,   # rimosso in v0.7
    headdim = config.d_model // config.n_heads,
)
```

## Dopo

```python
from mamba_ssm import Mamba3
self.mamba = Mamba3(
    d_model         = config.d_model,
    d_state         = config.mamba_d_state,
    headdim         = config.d_model // config.n_heads,
    is_mimo         = False,   # SISO — MIMO richiede TileLang non ancora stabile
    is_outproj_norm = False,   # AdaLN nel JambaBlock genitore già normalizza
    dtype           = torch.bfloat16,
)
```

## Pro

- Ricorrenza exponential-trapezoidal: più espressiva di Euler
- State update complex-valued: risolve parity problem irrisolvibile da Mamba2
- Rimuove `d_conv` e `expand` da config — meno iperparametri

## Contro / Trade-off

- Mamba3 non su PyPI al momento (aprile 2026): richiede build da GitHub
- MIMO disabilitato: TileLang non ancora stabile in produzione
- Kernel CUDA non portabili: richiede CUDA, non gira su CPU o Apple Silicon

## Risultati attesi

- Migliore coerenza su sequenze lunghe
- State tracking superiore su task che richiedono memoria strutturata

## Risultati reali

> *Da aggiornare dopo il training*
