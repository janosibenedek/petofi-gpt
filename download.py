"""Download the trained checkpoints (and optionally the source text) from the GitHub Release.

Every file is checked against its SHA-256 fingerprint, so a broken or partial download is
detected. Files that are already present and correct are skipped.

Usage:
  python download.py              # all 8 checkpoints (~310 MB) -> checkpoints/, needed by analysis.ipynb
  python download.py --quick      # only the small CPU model (cpu.pt, 3 MB) for a quick test
  python download.py --final      # only the two final models (GPT and Differential Transformer)
  python download.py --source     # also the source text -> data/raw/pg41504.txt,
                                  # in case gutenberg.org is unavailable (then run prepare_data.py)
"""
import argparse
import hashlib
import sys
import urllib.request
from pathlib import Path

BASE_URL = "https://github.com/janosibenedek/petofi-gpt/releases/latest/download"

# file name -> SHA-256 of the released file
CHECKPOINTS = {
    "cpu.pt":                "efe2de1db6cf0982dc3fd351037e0cf0020e907b431a5a46d260dd15e1f80e07",
    "colab.pt":              "de6a294d3ff06a8403dfb97b9b39f1558be0aac427b7d5aaec9c1287933ab237",
    "colab_drop03.pt":       "bf8ad417c055f812303e549ed1be6b6fb5b475ff73e90d3b8485ed45957317c8",
    "colab_drop04.pt":       "f5bd08b95a74b43a04bff8239e09de162f878feca08aef7c01da8fbab606173c",
    "colab_drop03_relu.pt":  "92db426c580c1e61a61447c0dc4922646e3486d3c62f2e81f678ad0f5185ec82",
    "colab_drop03_nores.pt": "4cf659757030985ad665da9a84b56ff40f132a764420d83e353cdbe0bf331385",
    "colab_drop02_diff.pt":  "54b8848f65dc83468276f9c991d6c514c0fb900331b0857c0c92aeeae4d8acb2",
    "colab_drop03_diff.pt":  "37d84b63f53e859826b1b593678a52871088a3f0f77fb1fe72d1f963e8a61b20",
}
SOURCE = ("pg41504.txt", "aa7655a76f0321eaf42d2a56f97b9ddbc7af9274ece654f36d3d3a119c5bffdf")
FINAL = ("colab_drop03.pt", "colab_drop02_diff.pt")


def sha256(path: Path) -> str:
    """Fingerprint of a file, read in 1 MB blocks (no need to hold 40 MB in memory)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(name: str, expected: str, target: Path) -> bool:
    """Download one release file to `target` unless it is already there; True if it is correct."""
    if target.exists() and sha256(target) == expected:
        print(f"  {name}: already present, OK")
        return True
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")    # only renamed once complete and verified
    print(f"  {name}: downloading ...", end=" ", flush=True)
    request = urllib.request.Request(f"{BASE_URL}/{name}", headers={"User-Agent": "petofi-gpt"})
    with urllib.request.urlopen(request) as response, partial.open("wb") as f:
        while block := response.read(1 << 20):
            f.write(block)
    if sha256(partial) != expected:
        partial.unlink()
        print("FAILED (fingerprint mismatch)")
        return False
    partial.replace(target)
    print(f"OK ({target.stat().st_size / 1e6:.1f} MB)")
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="only cpu.pt (3 MB)")
    group.add_argument("--final", action="store_true", help="only the two final models")
    p.add_argument("--source", action="store_true", help="also download the source text to data/raw/")
    p.add_argument("--out-dir", default="checkpoints")
    args = p.parse_args()

    names = ["cpu.pt"] if args.quick else list(FINAL) if args.final else list(CHECKPOINTS)
    print(f"checkpoints -> {args.out_dir}/")
    ok = all([fetch(n, CHECKPOINTS[n], Path(args.out_dir) / n) for n in names])
    if args.source:
        print("source text -> data/raw/")
        ok = fetch(SOURCE[0], SOURCE[1], Path("data") / "raw" / SOURCE[0]) and ok
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
