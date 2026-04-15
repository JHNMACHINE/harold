# Harold v0.7 — Training Log

*Aggiornato manualmente dopo ogni checkpoint significativo*

---

## Run di test — 10k iter su H200 NVL

**Hardware**: 1x H200 NVL (140GB VRAM)
**Batch effettivo**: 64 seq/step (batch_size=4, grad_accum=16)
**Token/step**: ~262k (64 × 4096)
**Token totali run**: ~2.6B

---

## Iter 0 — Baseline pre-training

> *Da aggiornare dopo il primo step di validation*

**Val loss attesa**: ~10.4 (CE di un modello random su vocab 32000: log(32000) ≈ 10.37)

---

## Iter 1 — Primo step

**Loss**: 1.0471  
**Score loss**: 0.0094  
**CE implicita**: ~10.4 (= (1.0471 - 0.0094) / 0.1)  
**Grad norm**: 0.08  
**Throughput**: 18.11s/it (torch.compile in corso — normale)

**Note**: la loss totale è dominata dal CE loss nei primissimi step,
esattamente come atteso. Score loss bassa perché x0-prediction con
modello random produce previsioni casuali ma il MSE è comunque piccolo
(gli embedding sono tutti vicini all'origine con init std=0.02).

---

## Iter ~500 — Post-warmup

> *Da aggiornare*

---

## Iter 2500 — Primo checkpoint

> *Da aggiornare*

**Metriche da registrare**:
- val loss totale e per t=[0.1, 0.3, 0.5, 0.7, 0.9]
- x0_norm_mean (atteso: ~0.5-1.0 se il modello sta imparando)
- MoE imbalance (atteso: <3x se il router bias funziona)
- grad ratio Mamba/Attn (atteso: <5x)

---

## Checklist convergenza sana

Segni positivi nei primi 10k iter:

- [ ] Val loss scende monotonicamente
- [ ] Score loss a t=0.5 scende più velocemente degli estremi
- [ ] x0_norm_mean > 0.1 (no collapse verso zero)
- [ ] MoE imbalance < 4x (nessun expert dominante)
- [ ] Grad ratio Mamba/Attn < 10x
- [ ] CE loss scende da ~10.4 verso ~8-9

---

## Note tecniche

**torch.compile**: i primi 1-3 step sono molto lenti (compilazione kernel).
Dal 4° step in poi il throughput si stabilizza.

**Self-conditioning**: abilitato con probabilità 0.5. Raddoppia il compute
per il 50% degli step ma migliora significativamente la qualità della generazione.

**Logit-Normal sampling**: `t_sampling="logit_normal"`, std=0.5.
La distribuzione campiona più frequentemente t≈0.5 — si vede dai log
come la score loss a t=0.5 scende più velocemente rispetto agli estremi.