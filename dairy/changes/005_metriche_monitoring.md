# [005] Metriche di monitoring per rilevare failure mode

**Data**: Aprile 2026  
**File modificati**: `training/validation.py`, `training/train.py`, `training/trainer.py`  
**Tag**: `[v0.7-M2a..c]`, `[v0.7-T4..5]`

## Perché

La training loss (MSE del Flow Matching) è fuorviante — può scendere
anche se il modello sta collassando linguisticamente. Con x0-prediction
e Mamba3 entrambi nuovi in v0.7, servono metriche specifiche per rilevare
i failure mode prima che diventino irrecuperabili.

## Prima

Solo `loss`, `score`, `ce`, `grad_norm` aggregato nel log.

## Dopo

**`validation.py`**:
```python
t_values = [0.1, 0.3, 0.5, 0.7, 0.9]  # era [0.3, 0.5, 0.7]

def compute_moe_load_stats(model):
    # max_load / mean_load per layer — ratio > 4x = WARNING
    ...
```

**`train.py`** (ogni 50 step, primi 10k):
```python
mamba_norm = get_total_norm([p for n,p in model.named_parameters() if 'mamba' in n])
attn_norm  = get_total_norm([p for n,p in model.named_parameters() if 'q_proj' in n ...])
if mamba_norm / attn_norm > 10.0:
    print("WARNING: grad Mamba/Attn ratio > 10x")
```

**`model.py`** (in `compute_loss`):
```python
x0_norm_mean  = x0_pred[mask].norm(dim=-1).mean().item()
x0_norm_std   = x0_pred[mask].norm(dim=-1).std().item()
x0_var_tokens = x0_pred.var(dim=1).mean().item()
```

**`trainer.py`**: `run_grad_accum` ritorna ora 6 valori (aggiunto `extra_metrics`).

## Pro

- Rileva x0 collapse prima che diventi irrecuperabile
- Rileva expert dominanti nel MoE (imbalance > 4x)
- Rileva dipendenza squilibrata Mamba/Attention (ratio > 10x)
- Overhead trascurabile — calcoli su tensori già in memoria

## Contro / Trade-off

- `run_grad_accum` a 6 valori: breaking change — tutti i chiamanti aggiornati
- Grad norm Mamba/Attn solo primi 10k step e ogni 50 — non continuo

## Risultati attesi

- `x0_norm_mean` stabile 0.5–2.0: training sano
- MoE imbalance < 3x: bilanciamento sano
- Grad ratio Mamba/Attn < 5x: uso equilibrato dei due mixer

## Risultati reali

> *Da aggiornare dopo il training*
