"""
DiffusionMoE v2 — main.py
==========================
Esempio di utilizzo completo: build, training step, e generazione.
"""

import torch
from config import ModelConfig
from model import DiffusionMoE
from decoding import ThresholdDecoding
from ebpo import EBPOTrainer
from train import DiffusionMoETrainer


def build_model(config: ModelConfig) -> DiffusionMoE:
    model = DiffusionMoE(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"DiffusionMoE v2 — parametri: {n_params / 1e6:.1f}M")
    return model


def demo_training_step(config: ModelConfig, model: DiffusionMoE):
    """Training con dati che coprono TUTTO il vocabolario."""
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    trainer = DiffusionMoETrainer(model, config, optimizer)

    print("Training il modello per 500 step...")
    
    for step in range(500):
        # IMPORTANTE: usa TUTTO il vocabolario, non solo numeri piccoli!
        batch = torch.randint(
            1, config.vocab_size,  # da 1 a 999
            (2, 64)
        )
        
        model.train()
        loss, losses = trainer.train_step(batch, use_mtf=True, num_turns=3)
        trainer.optimizer_step(loss)
        
        if step % 100 == 0:
            print(f"Step {step}: loss = {loss:.4f}, m2t_loss = {losses.get('m2t_loss', 0):.4f}")

    print("Training completato!")
    return model


def demo_generation(config: ModelConfig, model: DiffusionMoE):
    """Mostra generazione con S Mode e Q Mode + MBE."""
    decoder = ThresholdDecoding(config)

    # Prompt di 8 token, resto mascherato
    B, L       = 1, 64
    prompt_len = 8
    initial    = torch.full((B, L), config.mask_token_id)
    initial[:, :prompt_len] = torch.randint(1, config.vocab_size, (B, prompt_len))

    model.eval()

    print("Generazione S Mode...")
    out_s = decoder.decode(model, initial, mode="S", use_mbe=False)
    print("mask_token_id:", config.mask_token_id)
    print("token unici in out_s:", out_s.unique())
    print(f"  Output shape: {out_s.shape}, mask residui: {(out_s == config.mask_token_id).sum().item()}")

    print("Generazione Q Mode + MBE...")
    out_q = decoder.decode(model, initial, mode="Q", use_mbe=True, num_blocks=4)
    print(f"  Output shape: {out_q.shape}, mask residui: {(out_q == config.mask_token_id).sum().item()}")
    print("mask_token_id:", config.mask_token_id)
    print("token unici in out_q:", out_q.unique())


def demo_ebpo_step(config: ModelConfig, model: DiffusionMoE):
    """Mostra un singolo step EBPO."""
    import copy
    old_model = copy.deepcopy(model)

    ebpo = EBPOTrainer(model, config, learning_rate=1e-6)

    B, L      = 2, 32
    prompts   = torch.randint(1, config.vocab_size, (B, L))
    responses = torch.randint(1, config.vocab_size, (B, L))
    rewards   = torch.tensor([1.0, -0.5])

    model.train()
    metrics = ebpo.train_step(prompts, responses, rewards, old_model)
    print("EBPO step metrics:", {k: f"{v:.4f}" for k, v in metrics.items()})


if __name__ == "__main__":
    config = ModelConfig(
        vocab_size  = 1000,
        d_model     = 256,      # <-- AUMENTA da 128
        n_layers    = 4,         # <-- AUMENTA da 2
        n_heads     = 8,         # <-- AUMENTA da 4
        n_kv_heads  = 4,         # <-- AUMENTA da 2
        d_ff        = 1024,      # <-- AUMENTA da 512
        moe_n_routed_experts   = 8,   # <-- AUMENTA
        moe_top_k              = 2,
        ds_moe_n_shared_experts= 2,   # <-- AUMENTA
        max_seq_len = 256,
        block_size  = 64,
        diffusion_T = 16,
        mask_token_id = 1000
    )

    model = build_model(config)

    print("\n--- Training step ---")
    model = demo_training_step(config, model)

    print("\n--- Generazione ---")
    demo_generation(config, model)

    print("\n--- EBPO step ---")
    demo_ebpo_step(config, model)