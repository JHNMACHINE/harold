import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from config import ModelConfig


class Expert(nn.Module):
    """MLP expert per i routed experts. Usa attivazione GELU."""
    def __init__(self, n_embd: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, hidden_dim, bias=False),
            nn.GELU(),
            nn.Linear(hidden_dim, n_embd, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SharedExpert(nn.Module):
    """Shared expert con SwiGLU — sempre attivo su tutti i token."""
    def __init__(self, n_embd: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1      = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w3      = nn.Linear(n_embd, hidden_dim, bias=False)
        self.w2      = nn.Linear(hidden_dim, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.dropout(F.silu(self.w1(x)) * self.w3(x)))


class DeepSeekMoELayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_routed_experts  = config.moe_n_routed_experts
        self.top_k             = config.moe_top_k
        self.bias_update_gamma = 1e-3

        self.shared_experts = nn.ModuleList([
            SharedExpert(config.d_model, config.d_ff // 2, dropout=config.dropout)
            for _ in range(config.ds_moe_n_shared_experts)
        ])
        self.routed_experts = nn.ModuleList([
            Expert(config.d_model, config.d_ff // 4, dropout=config.dropout)
            for _ in range(self.n_routed_experts)
        ])
        self.router = nn.Linear(config.d_model, self.n_routed_experts, bias=False)
        self.register_buffer("router_bias", torch.zeros(self.n_routed_experts))

        # Salvato durante forward, usato da update_bias()
        self.router_indices: torch.Tensor | None = None

    def _affinity(self, x_flat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.router(x_flat).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        x_flat  = x.view(-1, C)   # (B*T, C)

        # ── Shared experts (sempre attivi su tutti i token) ───────────────
        shared_out = torch.zeros_like(x_flat)
        for expert in self.shared_experts:
            shared_out += expert(x_flat)

        # ── Router ────────────────────────────────────────────────────────
        s            = self._affinity(x_flat)                        # (N, E)
        sel_scores   = s + self.router_bias.to(s.device)             # type: ignore
        topk_indices = torch.topk(sel_scores, self.top_k, dim=-1).indices  # (N, top_k)

        # Salva per update_bias() — detach per non influenzare il gradiente
        self.router_indices = topk_indices.detach()

        # ── Gating (normalizzato sulle affinità reali, non sui sel_scores) ─
        s_sel = s.gather(dim=1, index=topk_indices)                  # (N, top_k)
        denom = s_sel.sum(dim=1, keepdim=True)
        gates = torch.where(
            denom > 1e-9,
            s_sel / (denom + 1e-9),
            torch.full_like(s_sel, 1.0 / self.top_k),
        ).to(x.dtype)                                                # (N, top_k)

        # ── Routed experts ────────────────────────────────────────────────
        routed_out = torch.zeros_like(x_flat)
        for i in range(self.n_routed_experts):
            mask             = (topk_indices == i)                   # (N, top_k)
            row_idx, which_k = mask.nonzero(as_tuple=True)
            if row_idx.numel() == 0:
                continue
            expert_out  = self.routed_experts[i](x_flat.index_select(0, row_idx))
            gate_values = gates[row_idx, which_k].unsqueeze(1)
            routed_out.index_add_(0, row_idx, expert_out * gate_values)

        return (shared_out + routed_out).view(B, T, C)

    @torch.no_grad()
    def update_bias(self):
        """
        Aggiorna il router_bias dopo ogni step:
          - expert sovraccarichi → bias scende → meno favoriti
          - expert scarichi      → bias sale   → più favoriti
        Va chiamato nel training loop DOPO optimizer.step().
        """
        if self.router_indices is None:
            return
        counts = torch.bincount(
            self.router_indices.view(-1),
            minlength=self.n_routed_experts,
        ).float().to(self.router_bias.device)   # type: ignore
        self.router_bias += self.bias_update_gamma * (counts.mean() - counts).sign()
        self.router_indices = None


class DiffusionMultiHeadAttention(nn.Module):
    """
    MHA bidirezionale per Diffusion LM.
    Niente KV cache, niente causal mask — vede l'intera sequenza.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_head   = config.n_heads
        self.n_embd   = config.d_model
        self.dropout  = config.dropout
        self.head_dim = config.d_model // config.n_heads
        self.c_attn        = nn.Linear(config.d_model, 3 * config.d_model, bias=config.bias)
        self.c_proj        = nn.Linear(config.d_model, config.d_model,     bias=config.bias)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape

        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)

        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # is_causal=False → bidirezionale, nessuna maschera causale
        y = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class AdaLN(nn.Module):
    """
    Adaptive Layer Norm.
    Condiziona gli hidden states sul timestep embedding:
      out = norm(x) * (1 + scale) + shift
    dove scale e shift sono proiettati da t_emb.
    """
    def __init__(self, d_model: int):
        super().__init__()
        self.norm = nn.LayerNorm(d_model, elementwise_affine=False)
        # Init a zero → all'inizio si comporta come LayerNorm standard
        self.proj = nn.Linear(d_model, 2 * d_model, bias=True)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, L, D)
        t_emb : (B, D)
        """
        scale, shift = self.proj(t_emb).chunk(2, dim=-1)   # (B, D) ciascuno
        scale = scale.unsqueeze(1)                          # (B, 1, D)
        shift = shift.unsqueeze(1)
        return self.norm(x) * (1.0 + scale) + shift


class Block(nn.Module):
    """
    Transformer block per Diffusion LM:
      - Attention bidirezionale
      - DeepSeekMoE come FFN
      - AdaLN per conditioning sul timestep
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.ada_ln_1  = AdaLN(config.d_model)
        self.ada_ln_2  = AdaLN(config.d_model)
        self.attn      = DiffusionMultiHeadAttention(config)
        self.moe       = DeepSeekMoELayer(config)
        self.attn_gate = nn.Parameter(torch.zeros(1))
        self.moe_gate  = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        """
        x     : (B, L, D)
        t_emb : (B, D)
        """
        x = x + self.attn_gate.tanh() * self.attn(self.ada_ln_1(x, t_emb))
        x = x + self.moe_gate.tanh()  * self.moe(self.ada_ln_2(x, t_emb))
        return x

class MaskDiffusionSchedule:
    """
    Cosine noise schedule per masked diffusion.
    alpha_t = probabilità che un token NON sia mascherato al tempo t.
    alpha va da 1 (t=0, nessun mask) a 0 (t=T, tutto mascherato).
    """

    def __init__(self, config: ModelConfig):
        self.T             = config.diffusion_T
        self.mask_token_id = config.mask_token_id

        # Cosine schedule precomputato — buffer CPU, spostato su device in q_sample
        t            = torch.linspace(0, self.T, self.T + 1)
        alphas       = torch.cos((t / self.T) * math.pi / 2) ** 2
        self.alphas  = alphas / alphas[0]   # normalizza a 1 in t=0

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Forward process: maschera x0 al timestep t.
        Ogni token viene mascherato con probabilità (1 - alpha_t).

        x0 : (B, L)  — sequenza originale
        t  : (B,)    — timestep per ogni sample

        Returns:
          xt   : (B, L)  — sequenza corrotta
          mask : (B, L)  — True dove il token è stato mascherato
        """
        alpha_t  = self.alphas[t].to(x0.device)            # (B,)
        mask_prob = 1.0 - alpha_t.unsqueeze(1)              # (B, 1) → broadcast su L

        mask = torch.bernoulli(mask_prob.expand_as(x0.float())).bool()

        xt      = x0.clone()
        xt[mask] = self.mask_token_id
        return xt, mask

    def get_alpha(self, t: torch.Tensor) -> torch.Tensor:
        """Restituisce alpha_t per il timestep t."""
        return self.alphas[t]


class DiffusionMoE(nn.Module):
    """
    Predice x0 (testo originale) dato xt (testo corrotto) e il timestep t.
    Approccio: x0-prediction con AdaLN per timestep conditioning.
    """
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config        = config
        self.d_model       = config.d_model
        self.mask_token_id = config.mask_token_id

        # ── Embeddings ────────────────────────────────────────────────────
        self.token_emb = nn.Embedding(config.vocab_size, config.d_model, padding_idx=0)
        self.pos_emb   = nn.Embedding(config.max_seq_len, config.d_model)

        # ── Timestep MLP ──────────────────────────────────────────────────
        self.time_emb = nn.Sequential(
            nn.Linear(config.d_model, config.d_model * 4),
            nn.SiLU(),
            nn.Linear(config.d_model * 4, config.d_model),
        )

        # ── Frequenze sinusoidali precompute ──────────────────────────────
        half  = config.d_model // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, dtype=torch.float32) / half
        )
        self.register_buffer("t_freqs", freqs)   # (D/2,)

        # ── Transformer blocks ────────────────────────────────────────────
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])

        # ── Output ────────────────────────────────────────────────────────
        self.norm_out = nn.LayerNorm(config.d_model)
        self.lm_head  = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # Init prima, weight tying dopo (altrimenti _init_weights sovrascrive il tying)
        self._init_weights()
        self.lm_head.weight = self.token_emb.weight

    def _init_weights(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                std = 0.02 / math.sqrt(2 * self.config.n_layers) if name.endswith("c_proj") else 0.02
                nn.init.normal_(module.weight, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)

    def get_timestep_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """
        t      : (B,)   int
        return : (B, D) float — sinusoidal embedding
        """
        args = t[:, None].float() * self.t_freqs[None]  # type: ignore  # (B, D/2)
        emb  = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)    # (B, D)
        if self.d_model % 2:
            emb = F.pad(emb, (0, 1))
        return emb

    def forward(self, xt: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        xt : (B, L)  — sequenza corrotta al timestep t
        t  : (B,)    — timestep per ogni sample

        Returns: logits (B, L, V)
        """
        B, L   = xt.shape
        device = xt.device

        positions = torch.arange(L, device=device).unsqueeze(0).expand(B, -1)
        x = self.token_emb(xt) + self.pos_emb(positions)

        t_emb = self.time_emb(self.get_timestep_embedding(t))

        for block in self.blocks:
            x = block(x, t_emb)

        return self.lm_head(self.norm_out(x))
    
def build_model(config: ModelConfig) -> DiffusionMoE:   
    # Create the transformer
    model = DiffusionMoE(config)
    
    # Initialize the parameters
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    
    return model