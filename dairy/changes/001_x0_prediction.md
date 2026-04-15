# [001] Migrazione a x0-prediction

**Data**: Aprile 2026  
**File modificati**: `core/model.py`  
**Tag**: `[v0.7-X1..X5]`

## Perché

In v0.6 Harold prediceva la velocità del flusso `v = noise - x0`.
Questo target varia molto tra timestep diversi, rendendo il training
meno stabile. Soprattutto, x0-prediction è il prerequisito necessario
per l'iterative decoding — serve la predizione esplicita di x0 per
sapere quali token congelare ad alta confidenza.

## Prima

```python
self.vel_pred = nn.Linear(config.d_model, config.d_model, bias=False)
# forward:
vel_pred  = self.vel_pred(x_out)
ce_logits = self.ce_head(x_out)        # CE da x_out
# loss: MSE(vel_pred, noise - x0_emb)
```

## Dopo

```python
self.x0_pred = nn.Linear(config.d_model, config.d_model, bias=False)
# forward:
x0_pred   = self.x0_pred(x_out)
ce_logits = self.ce_head(x0_pred)      # CE da x0_pred — consistente
# loss: MSE(x0_pred, x0_emb)
```

## Pro

- Target più stabile: x0 è costante lungo la traiettoria, v varia
- CE loss consistente con la predizione principale
- Prerequisito per iterative decoding
- Self-conditioning semanticamente coerente: condiziona su `x0_prev.mean(dim=1)`

## Contro / Trade-off

- Breaking change: checkpoint v0.6 incompatibili (rinomina `vel_pred` → `x0_pred`)
- Il sampler deve convertire: `v = (x_t - x0_pred) / t.clamp(min=1e-4)`

## Risultati attesi

- Convergenza più stabile, specialmente ai timestep estremi
- Val loss più uniforme tra t=[0.1..0.9]

## Risultati reali

> *Da aggiornare dopo il training*
