# Harold v0.7 — Jamba Flow Matching Mixture-of-Experts Diffusion Language Model

[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2026.xxxxx)
[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Training](https://img.shields.io/badge/Status-Training%20in%20progress-orange.svg)]()

Harold v0.7 is a **3.2B-parameter continuous diffusion language model** that combines:

- **Jamba architecture** — hybrid SSM-Transformer backbone alternating Mamba3 and Attention layers (3:1 ratio)
- **Flow Matching** — linear trajectory with **x0-prediction** and logit-normal timestep sampling
- **DeepSeek MoE** — timestep-conditioned routing via `[token_repr; timestep_emb]`, on every block
- **Iterative decoding** — high-confidence tokens are frozen progressively during denoising
- **MLA** (Multi-head Latent Attention) with compressed KV caching
- **DSA** (Diagonal Sparse Attention) with local window + global tokens
- **YaRN RoPE** scaling for context extension
- **Flash Attention 2** with automatic fallback to SDPA
- **AdaLN** for timestep conditioning on every block
- **Self-conditioning** and **CFG** for conditional generation

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Jamba (Mamba3 + Attention) + DeepSeek MoE + Flow Matching |
| Parameters | ~3.20B total |
| d_model | 1792 |
| Layers | 40 (30 Mamba3 + 10 Attention, ratio 3:1) |
| Attention heads | 28 (GQA 4:1, 7 KV heads) |
| MLA latent dim | 224 |
| Mamba3 d_state | 128 |
| MoE experts | 2 shared SwiGLU + 16 routed GELU (top-2) |
| MoE routed hidden | 608 (d_ff // 8) |
| MoE shared hidden | 1216 (d_ff // 4) |
| DSA window size | 256 |
| DSA global every | 64 |
| Max seq len | 4096 |
| Tokenizer | LLaMA-2 BPE (32,000 vocab) |
| Flow sigma_min | 1e-4 |
| t_sampling | logit_normal (std=0.5) |

## Changes from v0.6

| Component | v0.6 | v0.7 |
|-----------|------|------|
| Parameters | ~1.24B | ~3.20B |
| SSM | Mamba2 | Mamba3 (complex-valued, exponential-trapezoidal) |
| Layers | 36 (27M2 + 9A) | 40 (30M3 + 10A) |
| d_model | 1280 | 1792 |
| MoE experts | 1 shared + 8 routed | 2 shared + 16 routed |
| Diffusion target | v-prediction | **x0-prediction** |
| Timestep sampling | Uniform U[0,1] | **Logit-Normal (std=0.5)** |
| Decoding | Euler ODE | **Iterative decoding** (freeze high-confidence tokens) |
| Parallelism | DDP | DDP + **FSDP** (multi-GPU) |
| License | MIT | **Apache 2.0** |

## Architecture

### Jamba with Mamba3

Harold v0.7 uses **Mamba3** (Lahoti et al., ICLR 2026) as the SSM mixer. Each of the 40 blocks follows:

```
[Mamba3, Mamba3, Mamba3, Attention] × 10
```

Mamba3 improves on Mamba2 with three changes: exponential-trapezoidal discretization (more expressive than Euler), complex-valued state updates (enables richer state tracking), and a MIMO formulation for better expressivity at equal inference latency.

### x0-Prediction

Harold v0.7 switches from velocity prediction to direct x0-prediction:

```
v0.6: model predicts v = noise - x0  (varies along trajectory)
v0.7: model predicts x0              (constant along trajectory, more stable)
```

The velocity needed by the ODE solver is recovered as `v = (x_t - x0_pred) / t`.

### Iterative Decoding

At each denoising step, tokens with CE confidence above a threshold are **frozen** — replaced with their discrete embedding and excluded from further denoising. Only uncertain tokens continue to receive noise and are refined. This is inspired by MDLM and PLAID.

### Logit-Normal Timestep Sampling

Instead of uniform `t ~ U[0,1]`, training uses:

```python
u = torch.randn(B) * 0.5
t = torch.sigmoid(u)  # concentrated around t=0.5
```

Timesteps near 0.5 carry the most gradient signal for learning linguistic structure. This prevents x0-collapse at the extremes and is used in Stable Diffusion 3.

## Training

**Pretraining** — 100k steps on a multi-source corpus:

| Dataset | Weight |
|---------|--------|
| FineWeb-Edu | 25% |
| Wikipedia EN | 15% |
| SlimPajama | 20% |
| C4 | 13% |
| arXiv | 9% |
| PG-19 | 12% |
| GitHub Code (Python) | 3% |
| Open-Web-Math | 3% |

**Hardware:** 1× H200 NVL (140GB) via Vast.ai  
**Precision:** bfloat16  
**Optimizer:** MuonAdamW (Muon for 2D matrices, AdamW for embeddings/norms)  
**lr:** 8e-5 → 8e-6 cosine decay, 1000 warmup steps  
**Batch:** 4 × 16 grad accum = 64 effective (4096 tokens/seq)  
**Self-cond prob:** 0.5  
**CE loss weight:** 0.1  

## Usage

```python
import torch
from transformers import AutoTokenizer
from core.config import get_model_config
from core.model import Harold, build_model
from sampler import build_sampler

# Load model
state     = torch.load("harold-v0.7-3B.pt", map_location="cpu", weights_only=False)
model_cfg = state.get("model_cfg", get_model_config())
model     = build_model(model_cfg).cuda().bfloat16()
model.load_state_dict(state["model_state"])
model.eval()

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Generate with iterative decoding
sampler = build_sampler(model, n_steps=32, freeze_threshold=0.9, cfg_scale=3.0)
tokens  = sampler.generate(batch_size=1, seq_len=256)
text    = tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True)
print(text)
```

Or via the CLI:

```bash
python sampler.py --prompt "Explain how neural networks work" --max_steps 32
python sampler.py --prompt "Write a Python function to sort a list" --cfg_scale 2.0
python sampler.py --prompt "..." --no_iterative  # uniform denoising (v0.6 style)
```

## Installation

```bash
git clone https://github.com/JHN-MACHINE/harold
cd harold
pip install -r requirements.txt

# Mamba3 requires build from source (not yet on PyPI as of April 2026)
pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation --no-deps
pip install einops
```

## Limitations

- **Training in progress:** v0.7 pretraining is currently running. Generation quality metrics will be reported once the full 100k-step run completes.
- **Mamba3 not on PyPI:** Requires building from source. MIMO mode disabled (requires TileLang).
- **Diffusion latency:** Unlike autoregressive models, generation requires N forward passes. With iterative decoding and 32 steps, latency is ~1s on H200 for 256 tokens.
- **No SFT yet:** Supervised fine-tuning planned after pretraining completes.
- **CE comparability:** The auxiliary CE loss is computed at various noise levels and is not directly comparable to autoregressive perplexity.

## Training Diary

Development decisions, bugs encountered, and lessons learned are documented in [`diary/`](diary/):

- [`diary/v0.7/00_architecture.md`](diary/v0.7/00_architecture.md) — architectural decisions
- [`diary/v0.7/01_setup.md`](diary/v0.7/01_setup.md) — setup problems and solutions
- [`diary/v0.7/02_training_log.md`](diary/v0.7/02_training_log.md) — training metrics
- [`diary/v0.7/changes/`](diary/v0.7/changes/) — per-patch change log

## Citation

```bibtex
@article{vecchione2026haroldv07,
  title   = {Harold v0.7: Jamba Flow Matching with Mamba3, x0-Prediction and Iterative Decoding},
  author  = {Vecchione, Jonathan},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://huggingface.co/JHN-MACHINE/harold-v0.7}
}
```

## License

Apache License 2.0. See [LICENSE](LICENSE) for details.