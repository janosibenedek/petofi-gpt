"""Character-level N-gram baseline: the classical language model the GPT has to beat.

An N-gram model predicts the next character from the previous n-1 characters, using
counts from train.txt:   P(c | h) ~ count(h + c) / count(h)
Smoothing makes sure that combinations never seen in train.txt still get a probability > 0:
  wb    Witten-Bell interpolation (default): mixes the order-n estimate with the order n-1
        estimate, recursively down to the unigram; no parameter to tune
  add1  add-one (Laplace) smoothing: add 1 to every count; simple, but weak for large n

The order n is a hyperparameter: chosen on the VAL split, then reported once on TEST.

Usage:
  python ngram_baseline.py                     # n = 1..8, Witten-Bell
  python ngram_baseline.py --smoothing add1    # for comparison
"""
import argparse
import json
import math
import random
from collections import Counter
from pathlib import Path


class NGramModel:
    """Counts every substring of length 1..max_order of the training text once;
    afterwards any order n <= max_order can be evaluated."""

    def __init__(self, vocab: list[str], max_order: int, smoothing: str = "wb"):
        self.vocab, self.V = vocab, len(vocab)
        self.max_order, self.smoothing = max_order, smoothing

    def fit(self, text: str) -> "NGramModel":
        self.N = len(text)
        self.counts = Counter()                       # "ab" -> how many times "ab" occurs
        for k in range(1, self.max_order + 1):
            self.counts.update(text[i:i + k] for i in range(len(text) - k + 1))
        self.ctx_total = Counter()                    # "ab" -> how many times "ab" is followed by a char
        self.ctx_types = Counter()                    # "ab" -> how many DIFFERENT chars follow "ab"
        for gram, c in self.counts.items():
            if len(gram) >= 2:
                self.ctx_total[gram[:-1]] += c
                self.ctx_types[gram[:-1]] += 1
        return self

    def prob(self, context: str, ch: str, n: int) -> float:
        """P(ch | the last n-1 characters of context)."""
        context = context[-(n - 1):] if n > 1 else ""
        if self.smoothing == "add1":
            total = self.ctx_total[context] if context else self.N
            return (self.counts[context + ch] + 1) / (total + self.V)
        # Witten-Bell: start from the add-1 unigram, then refine with longer and longer contexts
        p = (self.counts[ch] + 1) / (self.N + self.V)
        for k in range(1, len(context) + 1):
            h = context[-k:]
            total = self.ctx_total[h]
            if total == 0:                            # unseen context: longer ones are unseen too
                break
            types = self.ctx_types[h]
            # many different followers -> trust the lower-order estimate more
            p = (self.counts[h + ch] + types * p) / (total + types)
        return p

    def token_losses(self, text: str, n: int) -> list[float]:
        """-log P(text[i] | previous characters) for every position i >= 1 (natural log)."""
        return [-math.log(self.prob(text[max(0, i - n + 1):i], text[i], n))
                for i in range(1, len(text))]

    def loss(self, text: str, n: int) -> float:
        losses = self.token_losses(text, n)
        return sum(losses) / len(losses)

    def generate(self, prompt: str, length: int, n: int, seed: int = 42) -> str:
        rng = random.Random(seed)
        out = prompt
        for _ in range(length):
            weights = [self.prob(out, c, n) for c in self.vocab]   # full distribution
            out += rng.choices(self.vocab, weights=weights)[0]
        return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data-dir", default="data")
    p.add_argument("--max-order", type=int, default=8)
    p.add_argument("--smoothing", choices=("wb", "add1"), default="wb")
    p.add_argument("--out", default="results/ngram_baseline.json")
    args = p.parse_args()

    data = Path(args.data_dir)
    vocab = json.loads((data / "meta.json").read_text(encoding="utf-8"))["vocab"]
    train, val, test = ((data / f"{s}.txt").read_text(encoding="utf-8") for s in ("train", "val", "test"))

    model = NGramModel(vocab, args.max_order, args.smoothing).fit(train)
    print(f"{args.smoothing} smoothing, {len(model.counts):,} distinct substrings counted "
          f"(length 1..{args.max_order})")
    print(f"{'n':>2} | {'train loss':>10} | {'val loss':>8} | {'val ppl':>7}")

    # 1) choose n on VAL (train loss only to show over-fitting: it keeps falling with n)
    val_losses = {}
    for n in range(1, args.max_order + 1):
        val_losses[n] = model.loss(val, n)
        train_loss = model.loss(train[:100_000], n)
        print(f"{n:>2} | {train_loss:10.4f} | {val_losses[n]:8.4f} | {math.exp(val_losses[n]):7.2f}")
    best_n = min(val_losses, key=val_losses.get)

    # 2) report the chosen model ONCE on TEST
    test_loss = model.loss(test, best_n)
    print(f"\nbest n on val: {best_n}  ->  test loss {test_loss:.4f}  (ppl {math.exp(test_loss):.2f}, "
          f"{test_loss / math.log(2):.3f} bits/char)")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"smoothing": args.smoothing, "val_loss": val_losses, "best_n": best_n,
                               "test_loss": test_loss, "test_ppl": math.exp(test_loss)}, indent=2),
                   encoding="utf-8", newline="\n")
    print(f"saved to {out}")

    print(f"\n--- sample from the {best_n}-gram model ---")
    print(model.generate("\n", 300, best_n))


if __name__ == "__main__":
    main()