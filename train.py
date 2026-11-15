"""Train the character-level GPT on the train split, early-stop on the val split.

Steps:
  1. load the vocabulary (data/meta.json) and encode train.txt / val.txt into index tensors
  2. build the model from a preset (+ optional ablation switches)
  3. training loop: random batches -> forward -> loss -> backward -> clip -> AdamW step,
     learning rate with linear warmup + cosine decay
  4. every eval_interval steps: estimate train/val loss on FIXED batches, keep the best
     model (saved to checkpoints/<name>.pt), stop early if val loss stops improving
  5. print a short generated sample from the best model

Usage:
  python train.py                                # cpu preset, a few minutes on a laptop
  python train.py --preset colab                 # GPU run on Colab
  python train.py --activation relu              # ablation: ReLU instead of GELU
  python train.py --no-residual                  # ablation: no skip connections
  python train.py --attention diff               # main experiment: Differential Transformer
  python train.py --max-iters 200                # quick smoke test
"""
import argparse
import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import torch

from model import GPT, GPTConfig, PRESETS

# training hyperparameters per preset (the architecture is in model.PRESETS)
TRAIN_DEFAULTS = {
    "cpu": dict(batch_size=32, lr=2e-3, max_iters=3000, warmup_iters=100,
                eval_interval=100, eval_iters=40, patience=5),
    "colab": dict(batch_size=64, lr=1e-3, max_iters=6000, warmup_iters=200,
                  eval_interval=200, eval_iters=50, patience=5),
}


# --------------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------------
def load_vocab(data_dir: Path) -> list[str]:
    meta = json.loads((data_dir / "meta.json").read_text(encoding="utf-8"))
    return meta["vocab"]


def load_split(data_dir: Path, split: str, stoi: dict[str, int]) -> torch.Tensor:
    """The whole split as ONE long tensor of character indices."""
    text = (data_dir / f"{split}.txt").read_text(encoding="utf-8")
    return torch.tensor([stoi[c] for c in text], dtype=torch.long)


def get_batch(data: torch.Tensor, block_size: int, batch_size: int, device: str,
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """batch_size random windows: x = data[s : s+T], y = data[s+1 : s+T+1] (shifted by one)."""
    starts = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    x = torch.stack([data[s:s + block_size] for s in starts])
    y = torch.stack([data[s + 1:s + 1 + block_size] for s in starts])
    return x.to(device), y.to(device)


# --------------------------------------------------------------------------------------------
# training helpers
# --------------------------------------------------------------------------------------------
def get_lr(it: int, lr: float, warmup_iters: int, max_iters: int) -> float:
    """Linear warmup from 0 to lr, then cosine decay down to lr / 10 at max_iters."""
    if it < warmup_iters:
        return lr * (it + 1) / warmup_iters
    progress = min(1.0, (it - warmup_iters) / max(1, max_iters - warmup_iters))
    return lr / 10 + 0.5 * (lr - lr / 10) * (1 + math.cos(math.pi * progress))


def make_optimizer(model: GPT, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """AdamW; weight decay only on matrices (Linear/Embedding weights),
    not on biases and LayerNorm parameters (1-dimensional tensors)."""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict[str, torch.Tensor], args, device: str,
                  autocast) -> dict[str, float]:
    """Mean loss on eval_iters batches per split. The batches are the SAME at every
    evaluation (fixed generator seed), so the numbers are directly comparable."""
    model.eval()                                      # dropout off
    out = {}
    for name, data in splits.items():
        gen = torch.Generator().manual_seed(1234)
        losses = []
        for _ in range(args.eval_iters):
            x, y = get_batch(data, model.config.block_size, args.batch_size, device, gen)
            with autocast():
                _, loss = model(x, y)
            losses.append(loss.item())
        out[name] = sum(losses) / len(losses)
    model.train()                                     # dropout back on
    return out


def default_name(args) -> str:
    name = args.preset
    if args.activation != "gelu":
        name += f"_{args.activation}"
    if args.no_residual:
        name += "_nores"
    if args.attention != "standard":
        name += f"_{args.attention}"
    return name


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", choices=PRESETS, default="cpu")
    p.add_argument("--activation", choices=("gelu", "relu"), default="gelu")
    p.add_argument("--no-residual", action="store_true")
    p.add_argument("--attention", choices=("standard", "diff"), default="standard")
    p.add_argument("--dropout", type=float, default=None, help="override the preset's dropout")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--lr", type=float, help="peak learning rate")
    p.add_argument("--max-iters", type=int)
    p.add_argument("--warmup-iters", type=int)
    p.add_argument("--eval-interval", type=int)
    p.add_argument("--eval-iters", type=int)
    p.add_argument("--patience", type=int, help="evaluations without val improvement before stopping")
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--out-dir", default="checkpoints")
    p.add_argument("--name", default=None, help="checkpoint name (default: from preset/ablation)")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    for key, value in TRAIN_DEFAULTS[args.preset].items():   # fill unset values from the preset
        if getattr(args, key) is None:
            setattr(args, key, value)
    args.name = args.name or default_name(args)
    return args


# --------------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)                      # init weights, dropout, batch sampling
    device = args.device
    use_amp = device.startswith("cuda")               # float16 mixed precision on the GPU
    autocast = (lambda: torch.autocast("cuda", dtype=torch.float16)) if use_amp \
        else (lambda: torch.autocast("cpu", enabled=False))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)   # keeps tiny fp16 gradients from vanishing

    # 1) data
    data_dir = Path(args.data_dir)
    vocab = load_vocab(data_dir)
    stoi = {ch: i for i, ch in enumerate(vocab)}
    train_data = load_split(data_dir, "train", stoi)
    val_data = load_split(data_dir, "val", stoi)

    # 2) model
    arch = dict(PRESETS[args.preset])
    if args.dropout is not None:
        arch["dropout"] = args.dropout
    config = GPTConfig(vocab_size=len(vocab), activation=args.activation,
                       residual=not args.no_residual, attention=args.attention, **arch)
    model = GPT(config).to(device)
    optimizer = make_optimizer(model, args.lr, args.weight_decay)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{args.name}.pt"
    print(f"run '{args.name}' on {device}: {model.num_params():,} params, "
          f"{len(train_data):,} train / {len(val_data):,} val characters")
    print(f"config: {config}")

    # 3) + 4) training loop with periodic evaluation and early stopping
    best_val, best_iter, bad_evals, history = float("inf"), 0, 0, []
    t0 = time.time()
    for it in range(args.max_iters + 1):
        lr = get_lr(it, args.lr, args.warmup_iters, args.max_iters)
        for group in optimizer.param_groups:
            group["lr"] = lr

        if it % args.eval_interval == 0 or it == args.max_iters:
            losses = estimate_loss(model, {"train": train_data, "val": val_data}, args, device, autocast)
            history.append({"iter": it, "train": losses["train"], "val": losses["val"], "lr": lr})
            improved = losses["val"] < best_val
            print(f"iter {it:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"(ppl {math.exp(losses['val']):6.2f}) | lr {lr:.2e} | "
                  f"{time.time() - t0:6.1f}s{'  * best, saved' if improved else ''}")
            if improved:
                best_val, best_iter, bad_evals = losses["val"], it, 0
                torch.save({"model": model.state_dict(), "config": asdict(config), "vocab": vocab,
                            "args": vars(args), "iter": it, "best_val_loss": best_val,
                            "history": history}, ckpt_path)
            else:
                bad_evals += 1
                if bad_evals >= args.patience:
                    print(f"early stopping: no val improvement for {args.patience} evaluations")
                    break
        if it == args.max_iters:
            break

        x, y = get_batch(train_data, config.block_size, args.batch_size, device)
        with autocast():
            _, loss = model(x, y)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()                 # backprop: gradients of every weight
        scaler.unscale_(optimizer)                    # (no-op without AMP)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)                        # weights -= lr * (AdamW update)
        scaler.update()

    # keep the full history in the checkpoint of the best model
    ckpt = torch.load(ckpt_path, map_location=device)
    ckpt["history"] = history
    torch.save(ckpt, ckpt_path)

    # 5) sample from the best model
    model.load_state_dict(ckpt["model"])
    model.eval()
    print(f"\nbest val loss {best_val:.4f} (ppl {math.exp(best_val):.2f}) at iter {best_iter} "
          f"-> {ckpt_path}  [{time.time() - t0:.0f}s]")
    start = torch.tensor([[stoi["\n"]]], device=device)
    sample = model.generate(start, max_new_tokens=400, temperature=0.8, top_k=20)[0].tolist()
    print("\n--- sample (temperature 0.8, top-k 20) ---\n" + "".join(vocab[i] for i in sample))


if __name__ == "__main__":
    main()