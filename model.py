"""Character-level GPT (decoder-only Transformer) for the petofi-gpt project.

Classes:
  GPTConfig            all hyperparameters of the architecture (stored in every checkpoint)
  CausalSelfAttention  multi-head masked self-attention
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
    attention: str = "standard"   # "standard"; "diff" will be added in a later commit

    def __post_init__(self):
        """Runs automatically after the generated __init__: reject invalid settings early."""
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})")
        if self.activation not in ("gelu", "relu"):
            raise ValueError(f"unknown activation: {self.activation!r}")
        if self.attention != "standard":
            raise ValueError(f"unknown attention type: {self.attention!r}")


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

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
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
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
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

    # 2) causality: changing the LAST character must not change the outputs before it
    model.eval()
    changed = idx.clone()
    changed[:, -1] = (changed[:, -1] + 1) % V
    out_a, _ = model(idx)
    out_b, _ = model(changed)
    assert torch.allclose(out_a[:, :-1], out_b[:, :-1]), "future token leaked into the past!"
    assert not torch.allclose(out_a[:, -1], out_b[:, -1])
    print("2) causality OK: positions 0..T-2 do not see the last character")

    # 3) attention maps: one per layer, zero above the diagonal, rows sum to 1
    model.set_store_attention(True)
    model(idx)
    maps = model.attention_maps()
    assert len(maps) == cfg.n_layer and maps[0].shape == (B, cfg.n_head, T, T)
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.all(maps[0][..., upper] == 0)
    assert torch.allclose(maps[0].sum(-1), torch.ones(B, cfg.n_head, T))
    model.set_store_attention(False)
    print(f"3) attention maps OK: {len(maps)} x {tuple(maps[0].shape)}")

    # 4) generation, also beyond block_size (the context window is cropped)
    out = model.generate(idx[:1, :3], max_new_tokens=2 * T, temperature=0.8, top_k=10)
    assert out.shape == (1, 3 + 2 * T) and int(out.max()) < V
    print(f"4) generate OK: {tuple(out.shape)}")

    # 5) the ablation variants build and run
    for kwargs in (dict(activation="relu"), dict(residual=False)):
        GPT(replace(cfg, **kwargs))(idx, targets)        # replace: copy of cfg with changes
    print("5) ablations OK: activation='relu', residual=False")

    # 6) the model can memorize one fixed batch (if not, backprop is broken somewhere)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    start = model(idx, targets)[1].item()
    for _ in range(150):
        _, loss = model(idx, targets)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    print(f"6) overfitting one batch: loss {start:.2f} -> {loss.item():.2f}")
    assert loss.item() < 1.0

    # 7) size of the presets
    for name, preset in PRESETS.items():
        n = GPT(GPTConfig(vocab_size=V, **preset)).num_params()
        print(f"7) preset {name!r}: {n:,} parameters")
    print("All tests passed.")


if __name__ == "__main__":
    _self_test()