# [004] Logit-Normal timestep sampling

**Data**: Aprile 2026  
**File modificati**: `core/model.py`, `core/config.py`  
**Tag**: `[v0.7-T1]`

## Perché

Con `t ~ U[0,1]` il modello campionava uniformemente tutti i timestep.
I timestep estremi (t≈0 e t≈1) sono meno informativi — a t≈0 il modello
fa denoising fine su poco rumore, a t≈1 deve ricostruire tutto dal rumore
puro. Entrambi hanno gradienti più deboli rispetto a t≈0.5.
Rischio concreto: il modello impara a predire x0 costante per minimizzare
la loss sugli estremi senza imparare la struttura linguistica.

## Prima

```python
t = torch.rand(B, device=device)  # uniforme [0, 1]
```

## Dopo

```python
def sample_t(self, B, device):
    if self.t_sampling == "logit_normal":
        u = torch.randn(B, device=device) * self.t_logit_normal_std
        return torch.sigmoid(u)  # concentrato intorno a t=0.5
    elif self.t_sampling == "cosine":
        u = torch.rand(B, device=device)
        return 0.5 * (1.0 - torch.cos(math.pi * u))
    else:  # uniform — baseline v0.6
        return torch.rand(B, device=device)
```

## Pro

- Campiona più frequentemente t≈0.5 dove il gradiente è più informativo
- Previene velocity/x0 collapse agli estremi
- Validato in SD3 e altri modelli FM recenti
- Configurabile via `t_sampling` in ModelConfig senza toccare il codice

## Contro / Trade-off

- `t_logit_normal_std=0.5` è un iperparametro aggiuntivo
- Timestep estremi visti meno frequentemente — possibile penalità sulla qualità agli estremi

## Risultati attesi

- Score loss a t=0.5 scende più velocemente di t=0.1 e t=0.9
- `x0_norm_mean` stabile > 0.1 nelle prime 10k iter (no collapse)

## Risultati reali

> *Da aggiornare dopo il training*
