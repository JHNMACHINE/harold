# [009] Fix FSDP setup: tokenizer barrier + bf16 cast

**Data**: Aprile 2026  
**File modificati**: `training/setup.py`  
**Tag**: `[v0.7-S4]`, `[v0.7-S5]`

## Perché

Due crash distinti durante il setup FSDP su 8×RTX 5090:

1. **Tokenizer race condition**: tutti gli 8 rank chiamavano
   `AutoTokenizer.from_pretrained()` simultaneamente, causando errori
   intermittenti di connessione HTTP a HuggingFace Hub. L'errore si
   manifestava su rank casuali (nell'ultimo caso rank 5).

2. **FSDP dtype mismatch**: `wrap_model_fsdp()` falliva con:
   ```
   ValueError: Must flatten tensors with uniform dtype but got
   torch.float32 and torch.bfloat16
   ```
   Alcuni parametri (Mamba2 `A_log`, `D`, router bias, eventuali bias
   lineari) restavano in fp32 dopo `build_model()` mentre il resto era
   già in bfloat16 via mixed precision.

## Prima

```python
# setup.py — tutti i rank scaricano contemporaneamente
tokenizer = AutoTokenizer.from_pretrained(
    train_cfg.tokenizer_model,
    token=os.environ.get("HF_TOKEN"),
)

# setup.py — nessun cast prima di FSDP
from utils.fsdp import wrap_model_fsdp
active_model = wrap_model_fsdp(model, local_rank, mixed_precision=True)
```

## Dopo

```python
# [v0.7-S4] Solo rank 0 scarica, gli altri aspettano
if dist.is_initialized() and dist.get_rank() != 0:
    dist.barrier()

tokenizer = AutoTokenizer.from_pretrained(
    train_cfg.tokenizer_model,
    token=os.environ.get("HF_TOKEN"),
)

if dist.is_initialized() and dist.get_rank() == 0:
    dist.barrier()

# [v0.7-S5] Cast forzato a bfloat16 prima di FSDP
for name, param in model.named_parameters():
    if param.dtype != torch.bfloat16:
        param.data = param.data.to(torch.bfloat16)

from utils.fsdp import wrap_model_fsdp
active_model = wrap_model_fsdp(model, local_rank, mixed_precision=True)
```

## Pro

- Tokenizer download deterministico: rank 0 scarica, gli altri usano la cache
- FSDP wrapping non crasha più su dtype misti
- Print diagnostico mostra esattamente quali parametri erano in fp32
- Nessun impatto su single-GPU o DDP (barrier solo se dist.is_initialized())

## Contro / Trade-off

- Barrier aggiunge ~1-2 secondi al setup (tempo di download tokenizer)
- Cast a bf16 di parametri come `A_log` e `D` è safe (bf16 ha stesso range di fp32)
  ma in teoria potrebbe ridurre precisione per parametri molto piccoli — in pratica
  irrilevante dato che il training è già in bf16

## Risultati attesi

- Setup FSDP completa senza errori su 8 GPU
- Tutti i parametri in bfloat16 uniforme al momento del wrapping FSDP
- Training loop parte normalmente dopo il setup

## Risultati reali

> *Da aggiornare dopo il training*
