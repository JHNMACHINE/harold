# Harold v0.5 — Flow Matching Mixture-of-Experts Diffusion Language Model

[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2026.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Harold v0.5 is a **1B-parameter continuous diffusion language model** that combines:

- **Flow Matching** — linear trajectory interpolation with velocity prediction, replacing VP-SDE
- **DeepSeek MoE** — timestep-conditioned routing via `[token_repr; timestep_emb]`
- **MLA** (Multi-head Latent Attention) for compressed KV caching
- **DSA** (Diagonal Sparse Attention) with local window + global tokens
- **YaRN RoPE** scaling for context extension without fine-tuning
- **Flash Attention 2** with automatic fallback to SDPA
- **AdaLN** for timestep conditioning
- **Self-conditioning** for iterative refinement
- **CFG** (Classifier-Free Guidance) for conditional generation

## Model Details

| Property | Value |
|----------|-------|
| Architecture | Flow Matching Diffusion Transformer + DeepSeek MoE |
| Parameters | ~1B total |
| d_model | 1280 |
| Layers | 36 |
| Attention heads | 20 (GQA 4:1, 5 KV heads) |
| MLA latent dim | 160 |
| MoE experts | 2 shared SwiGLU + 4 routed GELU (top-2) |
| DSA window size | 256 |
| DSA global every | 64 |
| Max seq len | 1024 |
| Tokenizer | LLaMA-3 BPE (32,000 vocab) |
| Flow sigma_min | 1e-4 |

## Changes from v0.4

| Component | v0.4 | v0.5 |
|-----------|------|------|
| Parameters | 733M | ~1B |
| d_model | 1024 | 1280 |
| Layers | 32 | 36 |
| Diffusion | VP-SDE (ε-prediction) | Flow Matching (v-prediction) |
| Tokenizer | GPT-2 BPE (50,257) | LLaMA-3 BPE (32,000) |
| Min-SNR weighting | Yes | No (uniform target) |
| Sampler | DPM-Solver++ | Euler ODE |

## How Flow Matching Works

Harold v0.5 generates text through **ODE integration** along a linear trajectory:

1. Start from pure Gaussian noise `x_1 ~ N(0, I)` in embedding space
2. For each step `t` from 1 to 0 (20 steps default):
   - Predict velocity `v = model(x_t, t)` — the direction from noise to data
   - Integrate: `x_{t-dt} = x_t - dt * v`
3. Decode final embeddings via the auxiliary CE head

The training target is `v = noise - x0` — constant along the trajectory and uniform across all timesteps, making training significantly more stable than VP-SDE without Min-SNR weighting.

The DeepSeek MoE router conditions on both the token representation and the timestep embedding, enabling experts to specialize implicitly across the diffusion trajectory.

## Training

**Pretraining** — target: 100k steps on a multi-source corpus:

| Dataset | Weight |
|---------|--------|
| FineWeb-Edu | 25% |
| SlimPajama | 25% |
| Wikipedia EN | 20% |
| C4 | 15% |
| OpenMathInstruct | 10% |
| CodeContests | 5% |

**Hardware:** NVIDIA RTX PRO 6000 (96GB) via Vast.ai  
**Precision:** bfloat16  
**Optimizer:** AdamW, lr=1e-4 → 1e-5 cosine decay  
**Batch:** 8 × 16 grad accum = 128 effective  
**Self-cond prob:** 0.5  
**CE loss weight:** 0.1  

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from model import build_model
from sampler import HaroldSampler

# Download model from Hugging Face
ckpt_path = hf_hub_download(
    repo_id="JHN-MACHINE/harold-v0.5",
    filename="harold-v0.5-1B.pt",
)

# Load model
state     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model_cfg = state["model_cfg"]
model     = build_model(model_cfg).cuda()
model.load_state_dict(state["model_state"])
model.eval()

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-1B")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

sampler = HaroldSampler(model, tokenizer, device="cuda")

# Unconditional generation
out = sampler.generate(prompt="Once upon a time", gen_len=128, steps=20)
print(out)

# Conditional generation with CFG
out = sampler.generate_conditioned(
    context="What is the capital of France?",
    gen_len=128, steps=20, cfg_scale=3.0,
)
print(out)
```

## Limitations

- **Training in progress:** The current release covers pretraining only. Generation quality metrics will be reported once the full 100k step run completes.
- **No SFT yet:** Supervised fine-tuning with CFG for conditional generation is planned after pretraining.
- **CE comparability:** The auxiliary CE loss is computed at various noise levels, not at t=0, and is not directly comparable to autoregressive perplexity.

## Citation

```bibtex
@article{vecchione2026haroldv05,
  title   = {Harold v0.5: Flow Matching with Mixture-of-Experts for Continuous Diffusion Language Modeling},
  author  = {Vecchione, Jonathan},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://huggingface.co/JHN-MACHINE/harold-v0.5}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.