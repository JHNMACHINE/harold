# [008] Fix dataset config + supporto data_dir

**Data**: Aprile 2026  
**File modificati**: `core/dataset.py`, `core/datasets_config.yaml`  
**Tag**: `[v0.7-D6]`

## Perché

Il training crashava al caricamento dei dataset di codice:

```
ValueError: BuilderConfig 'C' not found. Available: ['default']
```

Tre problemi concreti:
1. `macrocosm-os/code-parrot-github-code` non supporta config per linguaggio
   (ha solo `default`), il campo testo è `code` non `content`, e il filtraggio
   per linguaggio richiede `languages=[]` come parametro di `load_dataset`
2. `bigcode/the-stack-dedup` usa `data_dir="data/c"` (lowercase) per selezionare
   il linguaggio, non `config: C` — e richiede accettazione licenza gated
3. `dataset.py` non passava `data_dir` a `load_dataset()`, solo `config` come `name`

## Prima

```yaml
# datasets_config.yaml — non funzionava
- name: github-code
  path: macrocosm-os/code-parrot-github-code
  split: train
  text_field: content    # campo sbagliato, è "code"
  weight: 0.25

- name: the-stack-systems
  path: bigcode/the-stack-dedup
  config: C              # BuilderConfig 'C' non esiste
  split: train
  text_field: content
  weight: 0.05
```

```python
# dataset.py — _iter_pretraining_dataset
load_kwargs: dict = {"path": ds_cfg["path"], "split": ds_cfg["split"]}
if "config" in ds_cfg:
    load_kwargs["name"] = ds_cfg["config"]
# data_dir non gestito
```

## Dopo

```yaml
# datasets_config.yaml — bigcode/starcoderdata con data_dir
- name: starcoderdata-python
  path: bigcode/starcoderdata
  data_dir: python
  split: train
  text_field: content
  weight: 0.12

- name: starcoderdata-c
  path: bigcode/starcoderdata
  data_dir: c
  split: train
  text_field: content
  weight: 0.05
```

```python
# dataset.py — helper centralizzato _build_load_kwargs
def _build_load_kwargs(ds_cfg: dict) -> dict:
    load_kwargs: dict = {"path": ds_cfg["path"], "split": ds_cfg["split"]}
    if "config" in ds_cfg:
        load_kwargs["name"] = ds_cfg["config"]
    if "data_dir" in ds_cfg:
        load_kwargs["data_dir"] = ds_cfg["data_dir"]
    return load_kwargs
```

## Pro

- `bigcode/starcoderdata` è pulito (dedup + PII removal), non gated, 86 linguaggi
- Granularità per linguaggio: 7 stream separati (Python, JS, C, C++, Rust, Go, Shell) con pesi indipendenti
- Helper `_build_load_kwargs` centralizza la logica — evita duplicazione tra pretraining e SFT
- Nessun breaking change: dataset senza `data_dir` funzionano come prima

## Contro / Trade-off

- 14 stream di pretraining (vs 11 prima) → più connessioni HTTP simultanee a HF Hub
- `starcoderdata` è basato su The Stack v1 (2022), non v2 — codice meno recente
- Pesi codice ridistribuiti manualmente: Python 12% è dominante, potrebbe skeware verso Python

## Risultati attesi

- Il training parte senza `ValueError: BuilderConfig not found`
- Mix codice 35% del totale preservato (0.12+0.05+0.05+0.04+0.03+0.03+0.03 = 0.35)
- Nessun impatto su convergenza rispetto al mix originale (stesso volume totale di codice)

## Risultati reali

> *Da aggiornare dopo il training*