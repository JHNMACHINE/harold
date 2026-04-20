# [NNN] Refactor: split di `core/model.py` in package `core/model/`

**Data**: Aprile 2026
**File modificati**: `core/model.py` → `core/model/` (package), aggiornamenti import in `training/*`, `eval/*`
**Tag**: refactor (nessun tag `[v0.7-XN]` nuovo)

> *Sostituisci `NNN` con il numero progressivo corretto. Conta con:*
> `ls diary/v0.7/changes/*.md | wc -l` e usa quel numero + 1.

## Perché

`core/model.py` era cresciuto a ~800 righe con 13 classi, dalle primitive FP8
fino al top-level `Harold`. Due problemi concreti:

1. **Navigazione**: cercare `DeepSeekMoELayer.forward` significava scorrere 500
   righe di codice irrilevante. Con un editor moderno si gestisce, ma rallenta
   le modifiche mirate e gli agenti che lavorano sul file (Cursor, Claude Code)
   producono diff peggiori quando il context è diluito.
2. **Boundary delle modifiche**: toccare la MoE non dovrebbe richiedere di
   tenere in mente l'attention. File separati rendono espliciti i confini di
   responsabilità.

I commenti cronologici `[v0.7-X1]`…`[v0.7-FP8]` erano ormai rumore: i tag si
riferiscono a patch già documentate nel diary (001–007). Il codice deve
descrivere cosa fa *adesso*, non la storia di come ci è arrivato — quella vive
nel diary.

## Prima

```
core/
├── config.py
├── context.py
├── model.py               ← ~800 righe, 13 classi
├── dataset.py
└── datasets_config.yaml
```

Commenti tipici sparsi nel file:

```python
# [v0.7-X1] x0-prediction invece di v-prediction
# [v0.7-M3] Mamba2Block → Mamba3Block
# [v0.7-OPT7] Router FiLM: separa condizionamento x e t_emb
# [v0.7-OPT-GC] t_normalized e kv_offset salvati come stato temporaneo
# [v0.7-FP8] FP8Linear con _scaled_mm
```

## Dopo

```
core/
├── config.py
├── context.py
├── model/
│   ├── __init__.py        ← re-export pubblico: Harold, build_model, FlowMatchingSchedule
│   ├── quantization.py    ← FP8Linear, maybe_fp8
│   ├── norm.py            ← AdaLN
│   ├── schedule.py        ← FlowMatchingSchedule
│   ├── moe.py             ← Expert, SharedExpert, DeepSeekMoELayer, HashMoELayer
│   ├── attention.py       ← RotaryEmbedding, BlockCausalAttention
│   ├── ssm.py             ← Mamba3Block (con fallback Mamba2)
│   ├── blocks.py          ← JambaBlock (orchestratore)
│   └── harold.py          ← Harold, build_model
├── dataset.py
└── datasets_config.yaml
```

### Grafo delle dipendenze interne

```
quantization.py              (foglia, nessuna dipendenza interna)
norm.py                      (foglia)
schedule.py                  (foglia)
moe.py            → quantization
attention.py      → config
ssm.py            → config
blocks.py         → attention, ssm, moe, norm
harold.py         → blocks, quantization, schedule
__init__.py       → harold, schedule
```

Grafo aciclico, import sempre relativi al package (`from .quantization import …`).
Nessun rischio di circular import.

### Import da aggiornare nei file consumer

```python
# Prima
from core.model import Harold, build_model, FlowMatchingSchedule

# Dopo (identico — `core.model` ora è un package invece di un modulo)
from core.model import Harold, build_model, FlowMatchingSchedule
```

In realtà **gli import pubblici non cambiano**. La nuova `__init__.py` re-esporta
esattamente le stesse tre entità. Chi importava da `core.model` continua a
funzionare senza modifiche.

Gli unici import da aggiornare sono quelli che puntavano a classi interne:

| Prima                                       | Dopo                                   |
|---------------------------------------------|----------------------------------------|
| `from core.model import JambaBlock`         | `from core.model.blocks import JambaBlock` |
| `from core.model import DeepSeekMoELayer`   | `from core.model.moe import DeepSeekMoELayer` |
| `from core.model import BlockCausalAttention` | `from core.model.attention import BlockCausalAttention` |
| `from core.model import FP8Linear`          | `from core.model.quantization import FP8Linear` |

Controlla `training/` ed `eval/` con:

```bash
grep -rn "from core.model import" training/ eval/ sampler.py
```

## Commenti rimossi e dove sono stati consolidati

I commenti `[v0.7-XN]` inline sono stati rimossi dal codice. I riferimenti
cronologici restano nel diary, nei file patch appropriati:

| Tag                | File patch diary           | Cosa faceva                                     |
|--------------------|----------------------------|-------------------------------------------------|
| `[v0.7-X1]`…`[v0.7-X5]` | `001_x0_prediction.md`   | Migrazione da v-prediction a x0-prediction      |
| `[v0.7-M3]`        | `002_mamba3.md`            | Mamba2 → Mamba3 con fallback                    |
| `[v0.7-T1]`        | `004_logit_normal.md`      | Logit-Normal timestep sampling                  |
| `[v0.7-M4]`, `[v0.7-P4]` | `005_metriche_monitoring.md` | x0_norm_mean, x0_norm_std, x0_var_tokens |
| `[v0.7-OPT1…OPT9]` | Da creare se non già presenti | Ottimizzazioni: searchsorted, FiLM, in-place AdaLN, stack+view per timestep emb, routing vettorializzato, buffer pre-campionato per t |
| `[v0.7-OPT-GC]`    | Da creare                  | Gradient checkpointing con metodi di istanza (no closure) |
| `[v0.7-FP8]`       | Da creare                  | FP8Linear con straight-through estimator        |

Le docstring tecniche di valore permanente (firma, semantica, shape,
riferimenti a paper) sono state mantenute e riformattate in stile Sphinx/reST
standard. I commenti procedurali brevi (es. "sort once, O(log M) boundaries")
restano nel codice perché spiegano il *comportamento corrente*, non la storia.

## Pro

- File media ~200 righe, massimo ~450 (`moe.py` per via dei due layer MoE
  tenuti insieme).
- Confini semantici netti: modificare la MoE tocca un solo file; cambiare il
  backbone di attention ne tocca un altro; nessuna sovrapposizione.
- Il package `core.model/` ha una API pubblica esplicita in `__init__.py` —
  i consumer esterni non devono sapere nulla della struttura interna.
- `quantization.py` isolato prepara il terreno per future varianti (INT4,
  NVFP4, MXFP8) senza toccare il resto del modello.
- Leggibilità dei diff migliorata: una PR che tocca solo `moe.py` non richiede
  review del file gigante.
- Gli agenti (Cursor, Claude Code) hanno context più focalizzato e producono
  modifiche più precise.

## Contro / Trade-off

- **Breaking change negli import interni**: chi importava `JambaBlock`,
  `FP8Linear`, `BlockCausalAttention` e simili direttamente da `core.model`
  deve aggiornare il path al submodulo corretto. Gli import pubblici
  (`Harold`, `build_model`, `FlowMatchingSchedule`) sono invariati.
- **Perdita di storia inline**: i commenti `[v0.7-XN]` erano un TOC implicito
  delle modifiche recenti. Chi era abituato a scorrere il file per vedere
  "cosa è cambiato da v0.6" ora deve aprire il diary. Mitigato da questa
  stessa patch file.
- **Più file = più navigazione tra tab** durante modifiche che toccano l'intero
  stack (rari, per la maggior parte del lavoro si tocca un singolo livello).
- **Test suite**: se ci sono test che fanno `from core.model import <classe_interna>`
  vanno aggiornati insieme agli altri consumer.

## Risultati attesi

- `grep -rn "from core.model import" training/ eval/ sampler.py` elenca i
  consumer da aggiornare — il numero atteso è piccolo (4–10 file).
- `python -c "from core.model import Harold, build_model, FlowMatchingSchedule"`
  funziona senza errori dopo il refactor.
- `pytest tests/` (se presente) passa senza modifiche alla logica del modello.
- Nessun cambiamento misurabile in performance di training o inference —
  il refactor è puramente organizzativo.
- Il primo diff su `moe.py` dopo il refactor è <100 righe di context vs >500
  del file monolitico — segnale concreto del beneficio.

## Risultati reali

> *Da aggiornare dopo il merge del refactor e l'aggiornamento dei file consumer.*

## Checklist di esecuzione

- [ ] Copiare i 9 file nuovi in `core/model/` (package).
- [ ] Eliminare il vecchio `core/model.py` monolitico.
- [ ] Aggiornare gli import in `training/` ed `eval/` per le classi interne
  (vedi tabella sopra). Import pubblici invariati.
- [ ] Controllare con `grep -rn "from core.model" .` che non ci siano residui
  che puntano a path errati.
- [ ] Verificare con `python -c "from core.model import Harold; Harold(cfg)"`
  che il modello si costruisca.
- [ ] Fare un dry-run di training (1 step) per verificare runtime equivalence.
- [ ] Commit atomico con messaggio: `refactor(model): split core/model.py into package`.