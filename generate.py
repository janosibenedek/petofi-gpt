"""Generate Petőfi-style text with a trained checkpoint.

Usage:
  python generate.py                                        # checkpoints/cpu.pt, 500 characters
  python generate.py --prompt "Szabadság, szerelem!"        # continue a given beginning
  python generate.py --temperature 0.5 1.0 1.5              # compare temperatures (same seed)
  python generate.py --top-k 0 5 20 --temperature 1.0       # compare top-k values (0 = off)
  python generate.py --checkpoint checkpoints/colab.pt --out samples.txt
"""
import argparse
import itertools
from pathlib import Path

import torch

from model import GPT, GPTConfig


def load_model(path: str | Path, device: str = "cpu") -> tuple[GPT, list[str]]:
    """Rebuild the model from a checkpoint: config -> empty model -> trained weights."""
    ckpt = torch.load(path, map_location=device, weights_only=True)   # weights_only: safe loading
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()                                                       # dropout off
    return model, ckpt["vocab"]


def generate_text(model: GPT, vocab: list[str], prompt: str, max_new_tokens: int,
                  temperature: float, top_k: int | None, seed: int) -> str:
    stoi = {ch: i for i, ch in enumerate(vocab)}
    unknown = sorted(set(prompt) - set(stoi))
    if unknown:
        raise ValueError(f"characters not in the vocabulary: {unknown}")
    device = next(model.parameters()).device
    idx = torch.tensor([[stoi[c] for c in prompt]], dtype=torch.long, device=device)
    torch.manual_seed(seed)                       # same seed -> comparable samples
    out = model.generate(idx, max_new_tokens, temperature=temperature, top_k=top_k)
    return "".join(vocab[i] for i in out[0].tolist())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default="checkpoints/cpu.pt")
    p.add_argument("--prompt", default="\n", help="beginning of the text (default: a new line)")
    p.add_argument("--max-new-tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, nargs="+", default=[0.8])
    p.add_argument("--top-k", type=int, nargs="+", default=[0], help="0 = no top-k filtering")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None, help="also save the samples to this UTF-8 text file")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    model, vocab = load_model(args.checkpoint, args.device)
    prompt = args.prompt.replace("\\n", "\n")     # allow "\n" typed on the command line
    blocks = []
    for temperature, top_k in itertools.product(args.temperature, args.top_k):
        text = generate_text(model, vocab, prompt, args.max_new_tokens, temperature,
                             top_k or None, args.seed)
        header = (f"=== {Path(args.checkpoint).name} | temperature {temperature} | "
                  f"top-k {top_k or 'off'} | seed {args.seed} ===")
        blocks.append(f"{header}\n{text}\n")
        print(blocks[-1])
    if args.out:
        Path(args.out).write_text("\n".join(blocks), encoding="utf-8", newline="\n")
        print(f"saved to {args.out}")


if __name__ == "__main__":
    main()