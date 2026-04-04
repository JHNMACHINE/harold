# Harold v0.4 — Mixture-of-Experts Continuous Diffusion Language Model

[![arXiv](https://img.shields.io/badge/arXiv-2026.xxxxx-b31b1b.svg)](https://arxiv.org/abs/2026.xxxxx)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Harold v0.4 is a **733M-parameter continuous diffusion language model** that combines:

- **VP-SDE** (Variance-Preserving SDE) with epsilon prediction in embedding space
- **DeepSeek MoE** — timestep-conditioned routing via `[token_repr; timestep_emb]`, enabling expert specialization across noise levels
- **MLA** (Multi-head Latent Attention) for compressed KV caching
- **DSA** (Diagonal Sparse Attention) with local window + global tokens for efficient long-range modeling
- **YaRN RoPE** scaling for context extension without fine-tuning
- **Flash Attention 2** with automatic fallback to SDPA
- **AdaLN** for timestep conditioning
- **Self-conditioning** for iterative refinement
- **CFG** (Classifier-Free Guidance) for conditional generation
- **Min-SNR weighting** for stable training across all timesteps

## Model Details

| Property | Value |
|----------|-------|
| Architecture | VP-SDE Diffusion Transformer + DeepSeek MoE |
| Parameters | 733M total |
| d_model | 1024 |
| Layers | 32 |
| Attention heads | 16 (GQA 4:1, 4 KV heads) |
| MLA latent dim | 128 |
| MoE experts | 2 shared SwiGLU + 4 routed GELU (top-2) |
| DSA window size | 256 |
| DSA global every | 64 |
| Max seq len | 1024 |
| Tokenizer | GPT-2 BPE (50,257 vocab, byte-level, case-sensitive) |
| VP-SDE β | [0.1, 20.0] |
| SNR clip | 5.0 |

## Changes from v0.3

| Component | v0.3 | v0.4 |
|-----------|------|------|
| Parameters | 168M | 733M |
| d_model | 768 | 1024 |
| Layers | 12 | 32 |
| Tokenizer | BERT-uncased (30,522) | GPT-2 BPE (50,257) |
| RoPE | Standard | YaRN scaled |
| Attention | Standard SDPA | Flash Attention 2 + SDPA fallback |
| Context | 512 | 1024 |

## Training

**Pretraining** — ongoing run on a multi-source English corpus (target: 100k steps):

| Dataset | Weight |
|---------|--------|
| FineWeb-Edu | 30% |
| Wikipedia EN | 20% |
| Books/Gutenberg | 20% |
| C4 | 15% |
| OpenWebText | 15% |

**Hardware:** NVIDIA RTX A6000 (48GB) via Vast.ai  
**Precision:** float16 with GradScaler  
**Optimizer:** AdamW (fused), lr=1e-4 → 1e-5 cosine decay  
**Batch:** 8 × 16 grad accum = 128 effective  
**Self-cond prob:** 0.5  
**CE loss weight:** 0.1  

## Validation Results (20k step checkpoint)

Val loss per timestep (lower = better):

| t | total | score | CE |
|---|-------|-------|----|
| 0.1 | 0.695 | 0.053 | 6.42 |
| 0.3 | 0.779 | 0.024 | 7.55 |
| 0.5 | 0.785 | 0.016 | 7.68 |
| 0.7 | 0.769 | 0.005 | 7.64 |
| 0.9 | 0.764 | 0.000 | 7.64 |

> Score ≈ 0 at t≥0.5 indicates the denoising task is largely solved; remaining loss is driven by the CE auxiliary head.

## Usage

```python
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer
from model import build_model

# Download model from Hugging Face
ckpt_path = hf_hub_download(
    repo_id="JHN-MACHINE/harold-v0.4",
    filename="harold_v04_final.pt",
)

# Load model
state     = torch.load(ckpt_path, map_location="cpu", weights_only=False)
model_cfg = state["model_cfg"]
model     = build_model(model_cfg).cuda()
model.load_state_dict(state["model_state"])
model.eval()

tokenizer = AutoTokenizer.from_pretrained("gpt2")
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

## How It Works

Harold generates text through **reverse SDE integration** (Euler-Maruyama):

1. Start from pure Gaussian noise `x_1 ~ N(0, I)` in embedding space
2. For each step `t` from 1 to 0:
   - Predict noise `ε` with the model (optionally two passes for CFG)
   - Compute score: `∇ log p_t(x) ≈ -ε / σ(t)`
   - Integrate reverse SDE: `dx = [-0.5β(t)x - β(t)·score] dt + √β(t) dW`
3. Decode final embeddings via nearest-neighbor lookup in token embedding matrix

The DeepSeek MoE router conditions on both the token representation and the timestep embedding, enabling experts to specialize implicitly across noise levels — high-noise steps focus on global semantics, low-noise steps on local syntax.

YaRN RoPE allows context extension beyond the training length without fine-tuning, while DSA balances local attention (window=256) with global token access (every 64 positions) for efficient long-range modeling.

## Limitations
- **CE still high:** The auxiliary CE head shows perplexity well above competitive autoregressive models at this training stage. Expected to improve substantially over the full run.
- **Single GPU:** Training runs on a single A6000; multi-GPU DDP support is planned for future scaling.
- **No sampler yet:** The v0.4 sampler is under development. Generation quality metrics will be reported once sampling is validated.

## Citation

```bibtex
@article{vecchione2026harold,
  title   = {Harold v0.4: A 733M-Parameter Mixture-of-Experts Continuous Diffusion Language Model},
  author  = {Vecchione, Jonathan},
  journal = {arXiv preprint},
  year    = {2026},
  url     = {https://huggingface.co/JHN-MACHINE/harold-v0.4}
}
```

## License

MIT License. See [LICENSE](LICENSE) for details.