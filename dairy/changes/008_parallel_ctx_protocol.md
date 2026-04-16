# [008] Fix type error — _ParallelCtx Protocol per ddp_ctx

**Data**: Aprile 2026  
**File modificati**: `core/context.py`  
**Tag**: bug fix

## Perché

Il type checker (Pylance) segnalava errore su `ctx.ddp_ctx.teardown()`:

```
Cannot access attribute "teardown" for class "object"
Attribute "teardown" is unknown
```

`ddp_ctx` era tipizzato come `Optional[object]` per evitare una dipendenza
circolare (`context.py` → `utils.fsdp` → `core.model`). Ma `object` non
espone `teardown()`, quindi il type checker non poteva verificare la chiamata.

## Prima

```python
# ddp_ctx: Optional[object] — type checker cieco su teardown()
ddp_ctx: Optional[object] = field(default=None)
```

## Dopo

```python
class _ParallelCtx(Protocol):
    """Protocol minimo condiviso da DDPContext e FSDPContext."""
    def teardown(self) -> None: ...

# ddp_ctx: Optional[_ParallelCtx] — type checker verifica teardown()
ddp_ctx: Optional[_ParallelCtx] = field(default=None)
```

## Pro

- Il type checker ora verifica `ctx.ddp_ctx.teardown()` correttamente
- Nessuna dipendenza circolare — Protocol è definito localmente
- Sia `DDPContext` che `FSDPContext` lo soddisfano strutturalmente
  senza ereditarietà esplicita

## Contro / Trade-off

- `_ParallelCtx` espone solo `teardown()` — se si aggiungono altri metodi
  comuni in futuro, il Protocol va aggiornato

## Risultati attesi

- Nessun errore Pylance su `ctx.ddp_ctx.teardown()`

## Risultati reali

- Errore risolto ✓
