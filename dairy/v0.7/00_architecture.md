# Harold v0.7 — Decisioni Architetturali

*Aprile 2026*

---

## Perché un diffusion LM invece di un autoregressivo?

La scelta di usare Flow Matching invece di next-token prediction non è casuale.
Un diffusion LM genera tutti i token in parallelo — ogni token vede l'intera sequenza
durante il denoising, non solo il passato. Questo permette correzioni globali che
un modello autoregressivo non può fare: se il token 50 è sbagliato, i token 1-49
non possono correggerlo retroattivamente.

Il trade-off è la latenza: invece di un token alla volta, servono N step di denoising.
Lo risolviamo con iterative decoding — i token ad alta confidenza vengono congelati
presto, riducendo il compute necessario per i passi finali.

---

## v0.6 → v0.7: le tre grandi scelte

### 1. x0-prediction invece di v-prediction

In v0.6 Harold prediceva la *velocità del flusso* `v = noise - x0`.
In v0.7 predice direttamente `x0` — l'embedding del token originale.

**Perché**: il target x0 è più stabile numericamente. La velocità varia molto
tra timestep diversi; x0 è costante lungo la traiettoria. Soprattutto,
x0-prediction è il prerequisito per l'iterative decoding — serve la predizione
esplicita di x0 per sapere quali token congelare.

**Formula di conversione per il sampler**:
```
v = (x_t - x0_pred) / t
```

### 2. Mamba3 invece di Mamba2

Mamba3 (Lahoti et al., ICLR 2026) introduce tre miglioramenti su Mamba2:
- Discretizzazione exponential-trapezoidal (più espressiva di Euler)
- State update complex-valued (risolve il parity problem)
- MIMO — più espressività senza aumentare la latenza di decoding

**Problema pratico**: Mamba3 non è su PyPI al momento della scrittura (aprile 2026).
Il wheel precompilato `mamba_ssm-2.3.1` non include `Mamba3`. Soluzione:
```bash
pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation --no-deps --force-reinstall
```
Poi salvare il pacchetto compilato per le istanze successive:
```bash
cp -r /venv/main/lib/python3.12/site-packages/mamba_ssm /workspace/wheels/mamba_ssm_pkg_full
```

### 3. Scaling da 1.5B a 3.2B

Il salto di scala porta con sé una lezione importante sul MoE.

Con `d_ff=4864` e `ds_moe_n_shared_experts=2`, `moe_n_routed_experts=16`,
i divisori originali `d_ff//2` (shared) e `d_ff//4` (routed) producevano
**6B parametri invece di 3B** — il MoE era sovradimensionato.

**Fix**: calibrare i divisori per il target di parametri desiderato.
```python
# Stima parametri MoE per layer:
# routed: n_routed * 2 * d_model * routed_hidden
# shared: n_shared * 3 * d_model * shared_hidden
```
Con `d_ff//8=608` per i routed e `d_ff//4=1216` per gli shared:
→ ~1920M MoE totali → ~3.2B totali. ✓

**Lesson learned**: con MoE, il conto dei parametri va fatto esplicitamente
prima di lanciare il training. I divisori "standard" non scalano linearmente.

---

## Architettura finale v0.7

| Componente | Valore |
|---|---|
| Parametri totali | ~3.2B |
| d_model | 1792 |
| n_layers | 40 |
| n_heads | 28 (head_dim=64) |
| n_kv_heads | 7 (GQA ratio 4x) |
| d_ff | 4864 |
| Pattern layer | [Mamba3, Mamba3, Mamba3, Attention] × 10 |
| MoE per layer | 2 shared + 16 routed top-2 |
| moe_routed_hidden | 608 (d_ff // 8) |
| moe_shared_hidden | 1216 (d_ff // 4) |
| Compute attivo/token | 4 expert fwd (2+2) |
| seq_len | 4096 |
| Flow Matching | CFM lineare, x0-prediction |
| t_sampling | logit_normal (std=0.5) |

---

## Logit-Normal timestep sampling

In v0.6 campionavamo `t ~ U[0,1]`. In v0.7 usiamo logit-normal:
```python
u = torch.randn(B) * 0.5
t = torch.sigmoid(u)  # concentrato intorno a t=0.5
```

**Perché**: i timestep intermedi (t≈0.3-0.7) sono quelli dove la struttura
linguistica emerge e il gradiente è più informativo. A t≈0 (poco rumore)
il modello fa denoising fine; a t≈1 (molto rumore) quasi tutto è da
ricostruire. Campionare uniformemente spreca budget di training sugli
estremi meno informativi.

Questa tecnica è usata in SD3 e altri modelli FM recenti. Previene anche
il *velocity collapse* — il rischio che il modello impari a predire x0
costante per minimizzare la loss sugli estremi.

---

## Monitoring durante il training

Tre metriche critiche oltre alla loss totale:

**1. x0_pred_norm** — rileva collapse
- `x0_norm_mean < 0.1`: il modello collassa verso vettore zero
- `x0_var_tokens ≈ 0`: mode collapse (tutti i token → stesso embedding)

**2. MoE load imbalance** — rileva expert dominanti
- Ratio `max_load / mean_load > 4x`: un singolo expert riceve troppi token
- Il router bias update ogni 10 iter dovrebbe prevenire questo

**3. Grad norm Mamba vs Attention** — rileva dipendenza squilibrata
- Ratio `norm_mamba / norm_attn > 10x` nei primi 10k step: il modello
  usa solo la memoria a breve termine (Mamba) ignorando il contesto globale