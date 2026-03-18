# Harold v0.3 — Mixture-of-Experts Diffusion Language Model

[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2026.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Harold v0.3 is a 168M-parameter **continuous diffusion language model** that combines:

- **VP-SDE** (Variance-Preserving SDE) with epsilon prediction in embedding space
- **Timestep-conditioned MoE** — the router receives `[token_repr; timestep_emb]`, enabling expert specialization across noise levels
- **Min-SNR weighting** for stable training across all timesteps
- **MLA** (Multi-head Latent Attention) for compressed KV caching
- **DSA** (Diagonal Sparse Attention) for efficient long-range modeling
- **AdaLN** for timestep conditioning
- **Self-conditioning** for iterative refinement
- **CFG** (Classifier-Free Guidance) for conditional generation

## Model Details

| Property | Value |
|----------|-------|
| Architecture | VP-SDE Diffusion Transformer + MoE |
| Parameters | 168M total / ~80M active per token |
| d_model | 768 |
| Layers | 12 |
| Attention heads | 12 (GQA 3:1, 4 KV heads) |
| MLA latent dim | 96 |
| MoE experts | 2 shared SwiGLU + 4 routed GELU (top-2) |
| Tokenizer | bert-base-uncased (30,522 vocab) |
| VP-SDE β | [0.1, 20.0] |
| SNR clip | 5.0 |

## Training

**Pretraining** — 20,000 steps on a 5-source English corpus:

| Dataset | Weight |
|---------|--------|
| FineWeb-Edu | 30% |
| Wikipedia EN | 20% |
| Books/Gutenberg | 20% |
| C4 | 15% |
| OpenWebText | 15% |

**SFT** — Two-stage fine-tuning with Classifier-Free Guidance:
- Stage 1: UltraChat-200k (10k steps, lr=2e-5)
- Stage 2: OpenOrca (5k steps, lr=1e-5)

## Benchmark Results (WikiText-103)

| Metric | Value |
|--------|-------|
| Perplexity (via CE aux head, t=0.1) | 407.22 |
| BERTScore F1 | 0.3181 |
| MAUVE | 0.0041 |

> **Note:** Perplexity is computed via the auxiliary CE head at t=0.1 and is not directly comparable to autoregressive or masked-diffusion perplexity. See the paper for details.

## Usage

```python
import torch
from transformers import AutoTokenizer
from model import build_model
from sampler_v3 import HaroldSampler

# Load model
state     = torch.load("harold_sft_final.pt", map_location="cpu", weights_only=False)
model_cfg = state["model_cfg"]
model     = build_model(model_cfg)
model.load_state_dict(state["model_state"], strict=False)
model.eval()

tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
sampler   = HaroldSampler(model, tokenizer, device="cuda")

# Unconditional generation
out = sampler.generate(
    prompt="Once upon a time",
    gen_len=64,
    steps=100,
    mode="argmax",
)
print(out)

# Conditional generation with CFG (requires SFT checkpoint)
out = sampler.generate_conditioned(
    context="What is the capital of France?",
    gen_len=64,
    steps=50,
    cfg_scale=3.0,
)
print(out)
```

## How It Works

Harold generates text through **reverse SDE integration** (Euler-Maruyama):

1. Start from pure Gaussian noise `x_1 ~ N(0, I)`
2. For each step `t` from 1 to 0:
   - Predict noise `ε` with the model (two forward passes for CFG)
   - Compute score: `∇ log p_t(x) ≈ -ε / σ(t)`
   - Integrate reverse SDE: `dx = [-0.5β(t)x - β(t)·score] dt + √β(t) dW`
3. Decode final embeddings via auxiliary CE head (argmax)

The MoE router conditions on both the token representation and the timestep embedding, allowing experts to implicitly specialize on different noise levels (high-noise steps may focus on global semantics; low-noise steps on local syntax).

## Limitations

- **Scale**: 168M parameters is below the scale where diffusion LMs become competitive with autoregressive models. Harold v1.0 targets 1B parameters.
- **Tokenizer**: BERT-uncased is suboptimal for generation (lowercase normalization, unused vocab slots). Future versions will use a BPE tokenizer trained on the pretraining corpus.
- **Token bias**: The SFT model exhibits token bias from narrative fiction in UltraChat. This affects MAUVE scores and will be addressed with dataset filtering in future versions.
- **Training duration**: 20k pretraining steps (~5B tokens) is significantly less than competitive models.

## Citation

```bibtex
@article{vecchione2026harold,
  title   = {Harold v0.3: A Mixture-of-Experts Diffusion Language Model with Variance-Preserving SDE},
  author  = {Vecchione, Jonathan},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://huggingface.co/JHN-MACHINE/harold-v0.3}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.