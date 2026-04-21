- jamba
pipeline_tag: text-generation
library_name: harold
---

# Harold v0.7

**Harold** is a 3.2B-parameter continuous diffusion language model built for edge and IoT deployment. Unlike standard small LLMs — which are dense autoregressive Transformers shrunk down — Harold is designed from first principles for constrained hardware: subquadratic SSM layers, sparse MoE activation, and parallel diffusion decoding.

Developed by [Minya AI](https://minya.ai) · [GitHub](https://github.com/JHNMACHINE/harold) · Apache 2.0

> ⚠️ **Training in progress.** Harold v0.7 is currently completing its 100k-iteration pretraining run. Weights will be released upon completion. This card documents architecture and design decisions.

---

## Architecture

Harold v0.7 combines three ideas:

### 1. Jamba — Hybrid SSM-Transformer Backbone

3 out of every 4 layers use **Mamba3** State Space Models instead of attention. Only 1 in 4 is standard attention. This gives subquadratic complexity on long sequences — critical for edge devices where memory bandwidth is the bottleneck.

```
[Mamba3, Mamba3, Mamba3, Attention] × 10  →  40 layers total
```

Mamba3 (Lahoti et al., ICLR 2026) improves on Mamba2 with exponential-trapezoidal discretization, complex-valued state updates, and MIMO formulation.

### 2. Sparse Mixture of Experts

DeepSeek-style MoE with **2 shared + 16 routed experts, top-2 selection**. Harold has 3.2B total parameters but activates ~800M per forward pass — the compute cost of a much smaller model with the capacity of a larger one.

Routing is timestep-conditioned: the router receives `[token_repr; timestep_emb]`, allowing different experts to specialize at different noise levels.

### 3. Continuous Flow Matching Diffusion

Harold does not predict the next token. Instead, it refines an entire sequence from Gaussian noise toward coherent text using **x0-prediction Flow Matching** with logit-normal timestep sampling. This enables:

- **Parallel decoding** — all tokens refined simultaneously
- **Native infill** — fill-in-the-middle without tricks
- **Iterative decoding** — high-confidence tokens are frozen progressively, reducing unnecessary computation

---

## Model Details

| Property | Value |
|---|---|
| Architecture | Jamba (Mamba3 + Attention) + DeepSeek MoE + Flow Matching |
| Parameters | ~3.20B total / ~800M active |
| d_model | 1792 |
| Layers | 40 (30 Mamba3 + 10 Attention) |
| Attention | GQA 4:1 (28 heads, 7 KV) + MLA (latent dim 224) |
| Attention mask | DSA — local window 256 + global every 64 |
| Mamba3 d_state | 128 |
| MoE | 2 shared SwiGLU + 16 routed GELU (top-2) |
| Max seq len | 4096 (YaRN RoPE scale=4.0) |
| Tokenizer | LLaMA-2 BPE (32,000 vocab) |
| Diffusion | x0-prediction CFM, logit-normal t ~ σ(N(0, 0.5)) |
| Self-conditioning | enabled (p=0.5) |
| CFG | enabled at inference |

---

## Pretraining Dataset Mix

Edge/IoT-oriented mix — prioritizes code and systems content over long-form narrative:

| Dataset | Weight | Purpose |
|---|---|---|
| FineWeb-Edu | 20% | High-quality web text |
| SlimPajama | 15% | Thematic diversity |
| GitHub Code | 25% | General code (30+ languages) |
| The Stack (C/C++/Rust) | 10% | Systems & embedded languages |
| C4 | 10% | Web text |
| Wikipedia EN | 10% | Factual grounding |
| Open-Web-Math | 5% | Mathematical reasoning |
| arXiv | 3% | Technical writing |
| PG-19 | 2% | Long-form coherence |

**Hardware:** Vast.ai 8×B200  
**Precision:** bfloat16  
**Optimizer:** MuonAdamW  
**Learning rate:** 1e-4 → 1e-5 cosine, 1000 warmup steps  
**Effective batch:** 4 × 32 grad accum × 4096 tokens  

---

## Benchmarks (Harold v0.6, 1.5B)

Throughput vs equivalent dense Transformer baseline (single GPU, bfloat16):

| Seq Len | Harold tok/s | Transformer tok/s | Speedup |
|---|---|---|---|
| 256 | 1,250 | 1,374 | 0.91× |
| 512 | 2,450 | 2,726 | 0.90× |
| 1024 | 4,826 | 5,426 | 0.89× |
| 2048 | 9,171 | 9,924 | 0.92× |
| **4096** | **14,940** | **13,982** | **1.07×** |

The Mamba3 advantage compounds beyond 4096 tokens. Harold v0.7 with 3.2B params is expected to show a larger advantage due to the increased MoE sparsity.

---

## Usage

> Weights not yet released. The following assumes v0.7 checkpoint is available.

```python
import torch
from transformers import AutoTokenizer
from core.config import get_model_config
from core.model import build_model
from sampler import build_sampler

state     = torch.load("Harold-v0.7-3B-Base.pt", map_location="cpu", weights_only=False)
model_cfg = state.get("model_cfg", get_model_config())
model     = build_model(model_cfg).cuda().bfloat16()
model.load_state_dict(state["model_state"])
model.eval()

tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-2-7b-hf")
tokenizer.pad_token = tokenizer.eos_token

sampler = build_sampler(model, n_steps=32, freeze_threshold=0.9, cfg_scale=3.0)
tokens  = sampler.generate(batch_size=1, seq_len=256)
print(tokenizer.decode(tokens[0].tolist(), skip_special_tokens=True))
```

```bash
python sampler.py --prompt "Write a C function to read a sensor value" --max_steps 32
python sampler.py --prompt "..." --no_iterative  # uniform denoising
```

## Installation

```bash
git clone https://github.com/JHNMACHINE/harold
cd harold
pip install -r requirements.txt

# Mamba3 requires build from source
pip install git+https://github.com/state-spaces/mamba.git --no-build-isolation --no-deps
pip install einops
```

---

## Changelog

| Version | Params | Key changes |
|---|---|---|
| v0.4 | 733M | VP-SDE, GPT-2 tokenizer, pure Transformer |
| v0.5 | 1.25B | Flow Matching, LLaMA-2 tokenizer |
| v0.6 | 1.51B | Jamba (Mamba2), MoE, SFT |
| **v0.7** | **3.2B** | Mamba3, x0-prediction, iterative decoding, FSDP, edge/IoT focus |

---

## Limitations

- **Training in progress** — generation quality metrics pending full 100k-step run
- **Diffusion latency** — requires N forward passes; ~1s on H200 for 256 tokens at 32 steps
- **No SFT yet** — instruction following planned post-pretraining
- **Mamba3** — requires building from source, MIMO mode disabled

---

## Citation

```bibtex
@article{Harold,
  title   = {Harold v0.7: Edge-Optimized Diffusion Language Model with Jamba, Flow Matching and Sparse MoE},
  author  = {Vecchione, Jonathan},
  year    = {2026},
  url     = {https://huggingface.co/JHN-MACHINE/harold}
}
```

---

## License

Apache 2.0 — see [LICENSE](https://github.com/JHNMACHINE/harold/blob/main/LICENSE)

Built in Naples, Italy · [minya.ai](https://minya.ai)