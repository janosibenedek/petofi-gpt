"""Character-level GPT (decoder-only Transformer) for the petofi-gpt project.

Classes:
  GPTConfig            all hyperparameters of the architecture (stored in every checkpoint)
  CausalSelfAttention  multi-head masked self-attention
  DiffCausalSelfAttention  differential attention (Ye et al., 2024): difference of two softmax maps
  MLP                  position-wise feed-forward network
  Block                pre-norm Transformer block:  x + attn(ln1(x)),  then  x + mlp(ln2(x))
  GPT                  embeddings -> blocks -> final LayerNorm -> language-model head

Shapes used in the comments:
  B = batch size, T = sequence length (<= block_size), C = n_embd,
  V = vocab_size, nh = n_head, hs = head size = C // nh

Run `python model.py` for a quick self-test (shapes, initial loss, causality, ...).
"""
import math
from dataclasses import dataclass, replace

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int               # number of distinct characters (data/meta.json -> "vocab_size")
    block_size: int = 128         # context length: how many previous characters the model can see
    n_embd: int = 128             # embedding size C (width of every token vector)
    n_head: int = 4               # number of attention heads (n_embd must be divisible by it)
    n_layer: int = 4              # number of Transformer blocks
    dropout: float = 0.1          # dropout probability (regularization, active only in train mode)
    activation: str = "gelu"      # MLP nonlinearity: "gelu" or "relu"           (ablation)
    residual: bool = True         # False removes the skip connections           (ablation)
    attention: str = "standard"   # "standard" or "diff" (Differential Transformer)

    def __post_init__(self):
        """Runs automatically after the generated __init__: reject invalid settings early."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.activation not in ("gelu", "relu"):
            raise ValueError(f"unknown activation: {self.activation!r}")
        if self.attention not in ("standard", "diff"):
            raise ValueError(f"unknown attention type: {self.attention!r}")
        if self.attention == "diff" and self.n_head % 2 != 0:
            raise ValueError("differential attention needs an even n_head (uses n_head // 2 heads)")


# Architecture presets; use them as  GPTConfig(vocab_size=V, **PRESETS["cpu"])
PRESETS = {
    "cpu": dict(block_size=128, n_embd=128, n_head=4, n_layer=4, dropout=0.1),    # ~0.8M params
    "colab": dict(block_size=256, n_embd=384, n_head=6, n_layer=6, dropout=0.2),  # ~10.8M params
}


class CausalSelfAttention(nn.Module):
    """Multi-head self-attention where position t may only attend to positions <= t."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_head = config.n_head
        self.head_size = config.n_embd // config.n_head
        # W_Q, W_K and W_V of ALL heads in a single matrix: one matmul instead of 3 * n_head
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.proj = nn.Linear(config.n_embd, config.n_embd)   # W_O: mixes the concatenated heads
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        # lower-triangular boolean mask, True = "may attend"; a buffer moves with .to(device)
        # but is not a trainable parameter; persistent=False keeps it out of the state_dict
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size),
                             persistent=False)
        self.store_attention = False   # set True (via GPT.set_store_attention) to keep the maps
        self.last_att = None           # (B, nh, T, T) attention weights of the last forward pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)                           # 3 x (B, T, C)
        # split C into nh heads of size hs, then move the head axis forward:
        # (B, T, C) -> (B, T, nh, hs) -> (B, nh, T, hs), so every head is an independent "batch"
        q = q.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_size).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_size).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_size)   # (B, nh, T, T) scores
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))  # hide the future
        att = F.softmax(att, dim=-1)                                    # each row sums to 1
        if self.store_attention:
            self.last_att = att.detach()      # copy without gradient history, for analysis
        att = self.attn_drop(att)

        y = att @ v                                                     # (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C)                # concatenate the heads
        return self.resid_drop(self.proj(y))                            # (B, T, C)


def lambda_init_fn(layer_idx: int) -> float:
    """Initial lambda of the Differential Transformer paper: 0.8 - 0.6 * exp(-0.3 * (l - 1)),
    with l = 1, 2, ... (here layer_idx = l - 1 starts at 0): 0.2 in the first layer -> ~0.8 deep."""
    return 0.8 - 0.6 * math.exp(-0.3 * layer_idx)


class DiffCausalSelfAttention(nn.Module):
    """Differential attention (Ye et al., 2024, "Differential Transformer").

    Every head computes TWO softmax maps from two query/key pairs and subtracts them:
        A = softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d))
    Attention that both maps give to irrelevant positions ("noise") cancels out,
    like the common-mode noise in a differential amplifier.
    It uses n_head // 2 heads, each twice as wide, so the parameter count stays (almost) equal
    to the standard attention: only lambda's 4 vectors and the per-head RMSNorm are new.
    """

    def __init__(self, config: GPTConfig, layer_idx: int):
        super().__init__()
        self.n_head = config.n_head // 2                  # half as many heads ...
        self.head_size = config.n_embd // config.n_head   # d: size of Q1, Q2, K1, K2 (V has 2d)
        d = self.head_size
        self.qkv = nn.Linear(config.n_embd, 3 * config.n_embd)   # same matrices as standard
        self.proj = nn.Linear(config.n_embd, config.n_embd)
        self.attn_drop = nn.Dropout(config.dropout)
        self.resid_drop = nn.Dropout(config.dropout)
        # lambda = exp(lq1 . lk1) - exp(lq2 . lk2) + lambda_init : a learnable scalar per layer
        self.lambda_init = lambda_init_fn(layer_idx)
        self.lambda_q1 = nn.Parameter(torch.randn(d) * 0.1)
        self.lambda_k1 = nn.Parameter(torch.randn(d) * 0.1)
        self.lambda_q2 = nn.Parameter(torch.randn(d) * 0.1)
        self.lambda_k2 = nn.Parameter(torch.randn(d) * 0.1)
        self.subln = nn.RMSNorm(2 * d, eps=1e-5)          # per-head normalization of the output
        mask = torch.tril(torch.ones(config.block_size, config.block_size, dtype=torch.bool))
        self.register_buffer("mask", mask.view(1, 1, config.block_size, config.block_size),
                             persistent=False)
        self.store_attention = False
        self.last_att = None            # (B, h, T, T) DIFFERENTIAL map: may contain negative values
        self.last_lambda = None

    def lambda_value(self) -> torch.Tensor:
        return (torch.exp((self.lambda_q1 * self.lambda_k1).sum())
                - torch.exp((self.lambda_q2 * self.lambda_k2).sum()) + self.lambda_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        h, d = self.n_head, self.head_size
        q, k, v = self.qkv(x).split(C, dim=2)                           # 3 x (B, T, C)
        # every head gets two queries and two keys of size d:
        # (B, T, C) -> (B, T, h, 2, d) -> (B, h, 2, T, d)
        q = q.view(B, T, h, 2, d).permute(0, 2, 3, 1, 4)
        k = k.view(B, T, h, 2, d).permute(0, 2, 3, 1, 4)
        v = v.view(B, T, h, 2 * d).transpose(1, 2)                      # (B, h, T, 2d)

        att = (q @ k.transpose(-2, -1)) / math.sqrt(d)                  # (B, h, 2, T, T)
        att = att.masked_fill(~self.mask[:, :, :T, :T], float("-inf"))  # same causal mask
        att = F.softmax(att, dim=-1)                                    # two maps per head
        lam = self.lambda_value()
        att = att[:, :, 0] - lam * att[:, :, 1]                         # (B, h, T, T) difference
        if self.store_attention:
            self.last_att = att.detach()
            self.last_lambda = lam.item()
        att = self.attn_drop(att)

        y = att @ v                                                     # (B, h, T, 2d)
        y = self.subln(y) * (1 - self.lambda_init)                      # per-head norm, rescale
        y = y.transpose(1, 2).contiguous().view(B, T, C)                # concatenate the heads
        return self.resid_drop(self.proj(y))


class MLP(nn.Module):
    """Feed-forward network applied to every position separately: C -> 4C -> C."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.act = nn.GELU() if config.activation == "gelu" else nn.ReLU()
        self.proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.drop = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    """Pre-norm Transformer block: attention collects context, the MLP processes it."""

    def __init__(self, config: GPTConfig, layer_idx: int = 0):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = (DiffCausalSelfAttention(config, layer_idx) if config.attention == "diff"
                     else CausalSelfAttention(config))
        self.ln2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)
        self.residual = config.residual

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.residual:
            x = x + self.attn(self.ln1(x))    # skip connection: add the change to the input
            x = x + self.mlp(self.ln2(x))
        else:                                 # ablation: each sublayer REPLACES its input
            x = self.attn(self.ln1(x))
            x = self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    """Decoder-only language model: predicts the next character at every position."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.n_embd)   # which character
        self.pos_emb = nn.Embedding(config.block_size, config.n_embd)   # where in the context
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config, i) for i in range(config.n_layer)])
        self.ln_f = nn.LayerNorm(config.n_embd)
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)  # C -> V logits
        self.apply(self._init_weights)        # calls _init_weights on every submodule

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        """GPT-2 style init: small random weights, so every character starts ~equally likely."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)
        # nn.LayerNorm keeps its default init: weight = 1, bias = 0

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx, targets: (B, T) character indices; targets = idx shifted left by one.
        Returns (logits of shape (B, T, V), mean cross-entropy loss or None)."""
        B, T = idx.shape
        if T > self.config.block_size:
            raise ValueError(f"sequence length {T} > block_size {self.config.block_size}")
        pos = torch.arange(T, device=idx.device)                 # 0, 1, ..., T-1
        x = self.tok_emb(idx) + self.pos_emb(pos)                # (B, T, C) + (T, C) broadcast
        x = self.drop(x)
        for block in self.blocks:
            x = block(x)
        logits = self.lm_head(self.ln_f(x))                      # (B, T, V)

        loss = None
        if targets is not None:
            # cross_entropy expects (N, V) scores and (N,) class indices: flatten B and T
            loss = F.cross_entropy(logits.view(B * T, -1), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int,
                 temperature: float = 1.0, top_k: int | None = None) -> torch.Tensor:
        """Append max_new_tokens sampled characters to idx (B, T). Call model.eval() first.
        temperature < 1: safer, more repetitive text; > 1: more surprising, more errors.
        top_k: sample only from the k most likely characters."""
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size:]          # crop to the context window
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature              # only the last position: (B, V)
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)   # draw one index per row: (B, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    def set_store_attention(self, flag: bool) -> None:
        """Switch on/off keeping the attention maps of the next forward pass (for analysis)."""
        for block in self.blocks:
            block.attn.store_attention = flag
            block.attn.last_att = None

    def attention_maps(self) -> list[torch.Tensor]:
        """Attention weights of the last forward pass: one (B, nh, T, T) tensor per layer."""
        return [block.attn.last_att for block in self.blocks]


# --------------------------------------------------------------------------------------------
# Self-test: python model.py
# --------------------------------------------------------------------------------------------
def _self_test() -> None:
    torch.manual_seed(0)
    V, B, T = 97, 4, 16
    cfg = GPTConfig(vocab_size=V, block_size=T, n_embd=32, n_head=4, n_layer=2, dropout=0.0)
    model = GPT(cfg)
    idx = torch.randint(0, V, (B, T))
    targets = torch.randint(0, V, (B, T))

    # 1) shapes and initial loss: with tiny random weights every character is ~equally likely
    logits, loss = model(idx, targets)
    assert logits.shape == (B, T, V), logits.shape
    print(f"1) logits {tuple(logits.shape)}, initial loss {loss.item():.3f} "
          f"(expected ~ ln {V} = {math.log(V):.3f})")
    assert abs(loss.item() - math.log(V)) < 0.1

    for attention in ("standard", "diff"):
        m = model if attention == "standard" else GPT(replace(cfg, attention="diff"))
        m.eval()

        # 2) causality: changing the LAST character must not change the outputs before it
        changed = idx.clone()
        changed[:, -1] = (changed[:, -1] + 1) % V
        out_a, _ = m(idx)
        out_b, _ = m(changed)
        assert torch.allclose(out_a[:, :-1], out_b[:, :-1]), "future token leaked into the past!"
        assert not torch.allclose(out_a[:, -1], out_b[:, -1])

        # 3) attention maps: one per layer, zero above the diagonal; rows sum to 1 (standard)
        #    or to 1 - lambda (differential: difference of two maps that both sum to 1)
        m.set_store_attention(True)
        m(idx)
        maps = m.attention_maps()
        nh = cfg.n_head if attention == "standard" else cfg.n_head // 2
        assert len(maps) == cfg.n_layer and maps[0].shape == (B, nh, T, T)
        upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        assert torch.all(maps[0][..., upper] == 0)
        row_sum = 1.0 if attention == "standard" else 1.0 - m.blocks[0].attn.last_lambda
        assert torch.allclose(maps[0].sum(-1), torch.full((B, nh, T), row_sum), atol=1e-5)
        m.set_store_attention(False)
        print(f"2-3) {attention:8}: causality OK, attention maps {len(maps)} x "
              f"{tuple(maps[0].shape)}, rows sum to {row_sum:.3f}")

    # 4) generation, also beyond block_size (the context window is cropped)
    out = model.generate(idx[:1, :3], max_new_tokens=2 * T, temperature=0.8, top_k=10)
    assert out.shape == (1, 3 + 2 * T) and int(out.max()) < V
    print(f"4) generate OK: {tuple(out.shape)}")

    # 5) the ablation variants build and run
    for kwargs in (dict(activation="relu"), dict(residual=False), dict(attention="diff")):
        GPT(replace(cfg, **kwargs))(idx, targets)        # replace: copy of cfg with changes
    print("5) variants OK: activation='relu', residual=False, attention='diff'")

    # 6) the model can memorize one fixed batch (if not, backprop is broken somewhere)
    for attention in ("standard", "diff"):
        torch.manual_seed(0)
        m = GPT(replace(cfg, attention=attention))
        opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
        start = m(idx, targets)[1].item()
        for _ in range(150):
            _, loss = m(idx, targets)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        print(f"6) {attention:8}: overfitting one batch: loss {start:.2f} -> {loss.item():.2f}")
        assert loss.item() < 1.0

    # 7) size of the presets
    for name, preset in PRESETS.items():
        for attention in ("standard", "diff"):
            n = GPT(GPTConfig(vocab_size=V, attention=attention, **preset)).num_params()
            print(f"7) preset {name!r:8} {attention:8}: {n:,} parameters")
    print("All tests passed.")


if __name__ == "__main__":
    _self_test()