"""
test_inference.py
=================
Sanity check: carica il checkpoint trasferito e fa un forward pass.
"""
import torch
from transformers import AutoTokenizer
from config import ModelConfig
from model import build_model

# ── Config (deve matchare esattamente transfer.py) ──────────────────
cfg = ModelConfig()
cfg.vocab_size    = 126464
cfg.mask_token_id = 126336  # token MASK di LLaDA2
cfg.d_model       = 4096
cfg.n_layers      = 32
cfg.n_heads       = 32
cfg.n_kv_heads    = 8
cfg.d_ff          = 12288
cfg.rope_theta    = 500000.0

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CKPT   = "checkpoints/diffusion_moe_init.pt"

# ── Carica tokenizer ────────────────────────────────────────────────
print("Caricamento tokenizer...")
tok = AutoTokenizer.from_pretrained(
    "GSAI-ML/LLaDA-8B-Instruct",
    trust_remote_code=True,
)

# ── Carica modello ──────────────────────────────────────────────────
print("Caricamento checkpoint...")
torch.set_default_dtype(torch.bfloat16)
with torch.device(DEVICE):
    model = build_model(cfg)

ckpt = torch.load(CKPT, map_location=DEVICE)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()
print(f"Modello caricato su {DEVICE}")

# ── Forward pass ────────────────────────────────────────────────────
print("\nTest forward pass...")
text = "The quick brown fox"
tokens = tok(text, return_tensors="pt").input_ids.to(DEVICE)

# Maschera alcuni token
xt = tokens.clone()
xt[0, -2:] = cfg.mask_token_id  # maschera gli ultimi 2 token

t = torch.tensor([64], device=DEVICE)  # timestep a metà

with torch.no_grad():
    logits, _ = model(xt, t)

print(f"Input shape:  {xt.shape}")
print(f"Output shape: {logits.shape}")
print(f"Logits range: [{logits.min():.3f}, {logits.max():.3f}]")

# ── Decodifica predizione sui token mascherati ───────────────────────
pred_ids = logits[0, -2:].argmax(dim=-1)
pred_tokens = tok.decode(pred_ids)
print(f"\nToken mascherati predetti: '{pred_tokens}'")
print("\n✓ Forward pass completato senza errori!")