# [009] Fix OOM — eval qualitativa inline invece di subprocess

**Data**: Aprile 2026  
**File modificati**: `training/train.py`, `eval/eval_generation.py`  
**Tag**: bug fix

## Perché

L'eval qualitativa dopo ogni checkpoint periodico usava `subprocess.Popen`
per lanciare `eval_generation.py` in background mentre il training era ancora
in esecuzione. `eval_generation.py` chiama `load_model()` che carica Harold
3B in VRAM — ma il training occupava già ~100GB su H200.

```
Training:  ~100GB VRAM (pesi + optimizer + attivazioni)
+ eval:    ~6GB  (pesi Harold 3B bf16)
= OOM garantito
```

## Prima

```python
# In train.py — checkpoint periodico
subprocess.Popen([
    "python", eval_script,
    "--checkpoint", p,  # load_model() → +6GB VRAM → OOM
    ...
])
```

## Dopo

```python
# eval_generation.py — nuovo parametro
def evaluate_checkpoint(checkpoint, tokenizer, pad_token_id, args,
                        step=0, model=None):  # model=None bypassa load
    _model_provided = model is not None
    if model is None:
        model = load_model(checkpoint, ...)  # solo se model non fornito

# eval_generation.py — nuova funzione pubblica
def run_eval_on_model(model, tokenizer, pad_token_id, step, out_dir, ...):
    was_training = model.training
    model.eval()
    try:
        result = evaluate_checkpoint(..., model=model)  # no reload
    finally:
        if was_training:
            model.train()

# train.py — chiamata inline
from eval.eval_generation import run_eval_on_model
run_eval_on_model(model=ctx.model, tokenizer=ctx.tokenizer, ...)
```

## Pro

- Zero VRAM aggiuntiva — usa il modello già in memoria
- Il modello viene correttamente messo in `eval()` e riportato in `train()`
- Eval disponibile a ogni checkpoint periodico, non solo alla fine

## Contro / Trade-off

- L'eval blocca il training loop per la durata dell'inferenza (~2-3 min su H200)
- Se l'eval crasha, il training continua comunque (try/except)

## Risultati attesi

- Nessun OOM durante l'eval qualitativa
- Risultati eval disponibili a ogni checkpoint (10k, 20k, ...)

## Risultati reali

> *Da verificare al primo checkpoint a 10k iter*
