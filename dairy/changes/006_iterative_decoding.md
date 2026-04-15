# [006] Iterative decoding nel sampler

**Data**: Aprile 2026  
**File modificati**: `sampler.py`  
**Tag**: `[v0.7-S2]`

## Perché

Il denoising uniforme su tutti i token è inefficiente e produce testo
meno coerente. I token ad alta confidenza (già "decisi" dal modello)
continuano a ricevere rumore inutilmente ad ogni step. Con x0-prediction
abbiamo la confidenza per token esplicitamente disponibile tramite i CE logits.

## Prima

```python
# tutti i token denoised uniformemente per N step
x_t = x_t + dt * vel
```

## Dopo

```python
# calcola confidenza per token
conf = F.softmax(ce_logits / temperature, dim=-1).max(dim=-1).values
candidate = (conf >= freeze_threshold) & ~frozen

# congela i token sicuri
if candidate.any():
    best_tokens = ce_logits.argmax(dim=-1)
    x_frozen[candidate] = model.token_emb(best_tokens)[candidate].detach()
    frozen |= candidate

# solo i token non congelati continuano il denoising
x_t[frozen] = x_frozen[frozen]
x_t = x_t + dt * vel
```

## Pro

- Token sicuri non perturbati inutilmente — testo più coerente
- Riduce il compute effettivo negli step finali (meno token da denoising)
- Prerequisito per chunk generation e assistente vocale futuro
- `freeze_threshold` configurabile — facile da disabilitare con `--no_iterative`

## Contro / Trade-off

- Con modello poco trained (< 20k iter) quasi nessun token supera la soglia
- `freeze_threshold=0.9` conservativo — possibile che sia troppo alto inizialmente
- Aggiunge complessità al sampler rispetto al semplice Euler step

## Risultati attesi

- Con modello convergente: >50% token congelati entro i primi 20 step su 50
- Testo più coerente rispetto al denoising uniforme a parità di step

## Risultati reali

> *Da aggiornare dopo il training — testare con modello a 10k e 50k iter*
