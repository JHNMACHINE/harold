# Harold v0.7 — Setup e Problemi Risolti

*Aprile 2026*

---

## Ambiente

- **Cloud**: Vast.ai
- **GPU**: H200 NVL (140GB VRAM)
- **Storage**: /workspace (overlay, ~500GB liberi)
- **Python**: 3.12
- **PyTorch**: 2.x + CUDA 12.7

---

## Problema 1: Mamba3 non disponibile su PyPI

**Sintomo**:
```
ImportError: cannot import name 'Mamba3' from 'mamba_ssm'
```

**Causa**: `mamba_ssm==2.3.1` su PyPI non include ancora `Mamba3`.
Il modulo esiste nel repo GitHub ma non è stato rilasciato.

**Soluzione**:
```bash
pip install git+https://github.com/state-spaces/mamba.git \
    --no-build-isolation --no-deps --force-reinstall
pip install einops
```

**Persistenza tra istanze** — salvare il pacchetto compilato:
```bash
mkdir -p /workspace/wheels
cp -r /venv/main/lib/python3.12/site-packages/mamba_ssm \
    /workspace/wheels/mamba_ssm_pkg_full
cp /root/.cache/pip/wheels/.../causal_conv1d-*.whl /workspace/wheels/
```

**Script di setup per nuove istanze** (`/workspace/setup_env.sh`):
```bash
pip install /workspace/wheels/causal_conv1d-*.whl --no-build-isolation --no-deps
cp -r /workspace/wheels/mamba_ssm_pkg_full \
    /venv/main/lib/python3.12/site-packages/mamba_ssm
pip install einops
python -c "from mamba_ssm import Mamba3; print('OK')"
```

---

## Problema 2: Mamba3 MIMO richiede TileLang

**Sintomo**:
```
AssertionError: Fails to import Mamba-3 MIMO kernels.
Please ensure you installed the necessary dependencies, such as TileLang.
```

**Causa**: `is_mimo=True` richiede TileLang, non installato di default.

**Soluzione**: usare SISO (`is_mimo=False`) per ora.
```python
self.mamba = Mamba3(
    d_model         = config.d_model,
    d_state         = config.mamba_d_state,
    headdim         = config.d_model // config.n_heads,
    is_mimo         = False,   # SISO: kernel Triton standard
    is_outproj_norm = False,
    dtype           = torch.bfloat16,
)
```

**Nota**: MIMO darà un ulteriore boost qualitativo quando TileLang
sarà più stabile. Da rivalutare in v0.8.

---

## Problema 3: MoE oversized (6B invece di 3B)

**Sintomo**: il modello aveva 5.1B parametri invece dei 3.2B attesi.

**Causa**: i divisori `d_ff//2` (shared) e `d_ff//4` (routed) erano
calibrati per d_model=1280. Con d_model=1792 e d_ff=4864 producevano
expert troppo grandi.

**Diagnosi**:
```python
moe = sum(p.numel() for n,p in m.named_parameters()
          if any(x in n for x in ['routed','shared','router'])) / 1e6
# → 3837.5M — quasi tutto il budget di parametri
```

**Soluzione**: calibrare i divisori per il target di parametri:
```python
# Stima: routed_hidden=608, shared_hidden=1216 → ~3.2B totali
moe_routed_hidden: int = 608   # d_ff // 8
moe_shared_hidden: int = 1216  # d_ff // 4
```
I valori ora vengono letti da `ModelConfig` invece di essere hardcoded
in `DeepSeekMoELayer`.

---

## Problema 4: OOM su A100 40GB

**Sintomo**: `torch.OutOfMemoryError` su istanza A100 con 40GB VRAM.

**Causa**: avevamo noleggiato per errore una A100 40GB invece della SXM4 80GB.
Con 3.2B parametri e optimizer AdamW, il solo overhead fisso è ~38GB.

**Soluzione**: cambiare macchina. Usare H200 NVL (140GB) per avere
margine sufficiente per attivazioni con batch_size=4, seq_len=4096.

---

## Problema 5: venv separato dal pip di sistema

**Sintomo**: `pip install` installava i pacchetti ma il training non li trovava.

**Causa**: il training usa `/venv/main/bin/python`, mentre `pip` di default
installava in un ambiente diverso.

**Soluzione**: usare sempre `/venv/main/bin/pip` o il `pip` attivato dal venv.
Verifica:
```bash
which python  # deve essere /venv/main/bin/python
python -c "import mamba_ssm; print(mamba_ssm.__version__)"
```

---

## Configurazione finale funzionante

```bash
# Setup una tantum su nuova istanza H200
bash /workspace/setup_env.sh

# Verifica
python -c "from mamba_ssm import Mamba3; print('Mamba3 OK')"
python -c "from core.config import get_model_config; from core.model import Harold; \
    m=Harold(get_model_config()); \
    print(f'{sum(p.numel() for p in m.parameters())/1e9:.2f}B params')"
# → 3.20B params

# Lancio
mkdir -p /workspace/checkpoints/v0.7
torchrun --nproc_per_node=1 main.py --mode pretrain
```