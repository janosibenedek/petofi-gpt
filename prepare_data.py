"""Download, clean and split Petőfi Sándor's collected poems for character-level language modeling.

Source: Project Gutenberg eBook #41504, "Petőfi Sándor összes költeményei" (public domain).

Steps:
  1. download the plain-text edition (cached in <out-dir>/raw/)
  2. cut the Project Gutenberg header/footer and the title page
  3. split the text into poems using the title lines (layout-based, see split_poems)
  4. clean the poems and drop exact duplicates
  5. split by POEM (not by random windows) into train / val / test to avoid leakage
  6. write the splits, the vocabulary and some statistics

Usage:
  python prepare_data.py                 # default: 80/10/10 split, seed 42
  python prepare_data.py --modernize-cz  # optional: old "cz" spelling -> "c"
"""
import argparse
import hashlib
import json
import random
import re
import unicodedata
import urllib.request
from collections import Counter
from pathlib import Path

URL = "https://www.gutenberg.org/ebooks/41504.txt.utf-8"
FIRST_HEADING = "ELBESZÉLŐ KÖLTEMÉNYEK."          # the poems start here (after the title page)
STOP_MARKERS = ("Jegyzetek.", "[Footnote", "BETÜRENDES")  # notes / index: stop parsing
SECTION_HEADINGS = {                               # book sections, not poems
    "ELBESZÉLŐ KÖLTEMÉNYEK", "LYRAI KÖLTEMÉNYEK", "TOLDALÉK",
    "IFJÚKORI KISÉRLETEK", "KIHAGYOTT KÖLTEMÉNYEK",
}
DOUBTFUL_TITLES = {"Ez a világ egy nagy kocsma, jó áros…"}  # authorship uncertain (footnote 4)
POEM_SEP = "\n\n\n"                                # separator between poems in the split files
NOT_TITLE = re.compile(
    r"^(?:[IVXLC]+\."                              # part numbers inside a poem, e.g. "XII."
    r"|[A-ZÁÉÍÓÖŐÚÜŰ]+ ÉNEK\."                      # canto headings, e.g. "ELSŐ ÉNEK."
    r"|\d{1,3}\."                                  # numbered parts, e.g. "12." (Az apostol)
    r"|.*§\."                                      # sections of A táblabíró, e.g. "II-dik §."
    r"|\(.*|\*.*|_.*|\[.*)$"                        # (place/date), * * *, _subtitle_, [note]
)
YEAR = re.compile(r"^\d{4}\.$")                    # "1844." -- a year heading OR the poem "1848."
FOOTNOTE_REF = re.compile(r"\[\d+\]")              # "[3]" footnote references in titles


def download(raw_path: Path, force: bool = False) -> None:
    """Download the raw text once and cache it, so later runs work offline."""
    if force or not raw_path.exists():
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {URL} ...")
        request = urllib.request.Request(URL, headers={"User-Agent": "petofi-gpt (student project)"})
        with urllib.request.urlopen(request) as response:
            raw_path.write_bytes(response.read())


def strip_gutenberg(text: str) -> str:
    """Keep only the book itself: drop the license header/footer and the title page."""
    start = text.find("\n", text.find("*** START OF")) + 1
    end = text.find("*** END OF")
    body = text[start:end if end != -1 else None]
    first = body.find(FIRST_HEADING)
    return body[first:] if first != -1 else body


def is_title(candidate: str, next_line: str) -> bool:
    """Decide whether a non-indented line block is a poem title.
    next_line is the next non-empty line after it (verse lines start with two spaces)."""
    if not candidate or NOT_TITLE.match(candidate):
        return False
    if YEAR.match(candidate) and next_line.startswith("*"):     # "1844." + "* *" = year heading
        return False
    if candidate.endswith(",") and not next_line.startswith("  "):  # "Mutató tábla," + prose
        return False
    return True


def split_poems(text: str) -> list[dict]:
    """Group the lines into poems.

    Layout of this edition:
      * verse lines are indented by two spaces,
      * a poem title is a non-indented line (sometimes two) after >= 2 empty lines,
        followed by an empty line -- narrative poems have UPPERCASE titles, lyric poems
        normal case ("A Dunán."), untitled poems use their first line ("Érik a gabona…"),
      * other non-indented lines (years, places, notes, section headings) are skipped.
    """
    lines = text.splitlines()
    poems, title, body = [], None, []
    i, blanks = 0, 0
    while i < len(lines):
        line = lines[i]
        if line.startswith(STOP_MARKERS):
            break
        if not line.strip():
            blanks += 1
            if title is not None:
                body.append("")
            i += 1
            continue
        if line.startswith("  "):                  # verse line
            if title is not None:
                body.append(line[2:].rstrip())
            blanks, i = 0, i + 1
            continue
        # non-indented line: collect a possible (multi-line) title
        j = i
        while j < len(lines) and lines[j].strip() and not lines[j].startswith(" "):
            j += 1
        candidate = FOOTNOTE_REF.sub("", " ".join(l.strip() for l in lines[i:j])).strip()
        k = j
        while k < len(lines) and not lines[k].strip():
            k += 1
        next_line = lines[k] if k < len(lines) else ""
        # usually 2+ empty lines before a title; one is enough if verse follows (typo in source)
        spaced = blanks >= 2 or (blanks == 1 and next_line.startswith("  "))
        if spaced and k > j and is_title(candidate, next_line):
            if title is not None:
                poems.append({"title": title, "lines": body})
            # a section heading may have a second line, e.g. "LYRAI KÖLTEMÉNYEK." + "(1842–1849.)"
            is_section = lines[i].strip().rstrip(".") in SECTION_HEADINGS
            title, body = (None, []) if is_section else (candidate.rstrip(".,"), [])
        blanks, i = 0, j
    if title is not None:
        poems.append({"title": title, "lines": body})
    return poems


def clean_poem(lines: list[str]) -> str:
    """Join the verse lines, remove italic markup, fix two rare characters
    (a stray tab, the single 'ë' in "Phaëton") and collapse long runs of empty lines."""
    text = "\n".join(lines).replace("_", "").replace("\t", " ").replace("ë", "e")
    text = re.sub(r"(?<=\S) {2,}", " ", text)     # inner runs only: keep stanza indentation
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def modernize_cz(text: str) -> str:
    """Optional: the 19th-century 'cz' spelling becomes the modern 'c'."""
    return text.replace("CZ", "C").replace("Cz", "C").replace("cz", "c")


def assign_splits(poems: list[dict], val_frac: float, test_frac: float, seed: int,
                  max_eval_share: float = 0.1) -> None:
    """Shuffle the poems and fill test and val up to their share of CHARACTERS;
    everything else goes to train. Whole poems never cross split boundaries.

    A poem longer than max_eval_share of a split's budget always goes to train:
    otherwise one long narrative poem (Az apostol is ~9% of the book) could fill
    the whole validation set, and val/test would consist of only a few poems."""
    order = list(range(len(poems)))
    random.Random(seed).shuffle(order)
    total = sum(len(p["text"]) for p in poems)
    budget = {"test": test_frac * total, "val": val_frac * total}
    filled = {"test": 0, "val": 0}
    for i in order:
        n, target = len(poems[i]["text"]), "train"
        for split in ("test", "val"):
            if n <= max_eval_share * budget[split] and filled[split] + n <= budget[split]:
                target = split
                break
        poems[i]["split"] = target
        if target != "train":
            filled[target] += n


def write_utf8(path: Path, text: str) -> None:
    """UTF-8 with "\\n" line endings on every OS (Windows would write "\\r\\n" by default)."""
    path.write_text(text, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--raw-file", default=None, help="use a local raw file instead of downloading")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--val-frac", type=float, default=0.1)
    parser.add_argument("--test-frac", type=float, default=0.1)
    parser.add_argument("--min-lines", type=int, default=2, help="drop fragments shorter than this")
    parser.add_argument("--modernize-cz", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    raw_path = Path(args.raw_file) if args.raw_file else out / "raw" / "pg41504.txt"
    if not args.raw_file:
        download(raw_path, args.force_download)
    raw_bytes = raw_path.read_bytes()
    raw_sha256 = hashlib.sha256(raw_bytes).hexdigest()   # fingerprint of the exact source file
    raw = raw_bytes.decode("utf-8")

    text = unicodedata.normalize("NFC", strip_gutenberg(raw)).replace("\r\n", "\n")
    if args.modernize_cz:
        text = modernize_cz(text)

    # 1) poems, 2) cleaning + length filter, 3) exact-duplicate removal
    poems, seen, n_short, n_dup, n_doubtful = [], set(), 0, 0, 0
    for p in split_poems(text):
        if p["title"] in DOUBTFUL_TITLES:
            n_doubtful += 1
            continue
        body = clean_poem(p["lines"])
        if sum(1 for line in body.splitlines() if line.strip()) < args.min_lines:
            n_short += 1
            continue
        key = re.sub(r"\W+", "", body.lower())
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        poems.append({"title": p["title"], "text": f"{p['title']}\n\n{body}"})

    assign_splits(poems, args.val_frac, args.test_frac, args.seed)

    # write the three splits (poems keep their original order inside a split)
    stats = {}
    for split in ("train", "val", "test"):
        part = [p for p in poems if p["split"] == split]
        joined = POEM_SEP.join(p["text"] for p in part)
        write_utf8(out / f"{split}.txt", joined)
        stats[split] = {"poems": len(part), "chars": len(joined)}

    all_text = POEM_SEP.join(p["text"] for p in poems)
    vocab = sorted(set(all_text))
    train_chars = set((out / "train.txt").read_text(encoding="utf-8"))
    meta = {
        "source": URL, "raw_sha256": raw_sha256, "settings": vars(args),
        "vocab": vocab, "vocab_size": len(vocab),
        "chars_missing_from_train": sorted(set(vocab) - train_chars),
        "splits": stats,
        "dropped_short": n_short, "dropped_duplicates": n_dup, "dropped_doubtful": n_doubtful,
    }
    write_utf8(out / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))
    write_utf8(out / "poems.json", json.dumps(poems, ensure_ascii=False, indent=1))

    # short report
    print(f"Poems kept: {len(poems)} (dropped: {n_short} too short, {n_dup} duplicates, "
          f"{n_doubtful} of doubtful authorship)")
    for split, s in stats.items():
        print(f"  {split:5}: {s['poems']:4d} poems, {s['chars']:9,d} characters")
    print(f"Vocabulary size: {len(vocab)}")
    print("Most common characters:", Counter(all_text).most_common(12))
    if meta["chars_missing_from_train"]:
        print("Characters that appear only in val/test:", meta["chars_missing_from_train"])
    longest = sorted(poems, key=lambda p: len(p["text"]), reverse=True)[:5]
    print("Longest poems:", [(p["title"], len(p["text"])) for p in longest])
    print("First titles:", [p["title"] for p in poems[:15]])


if __name__ == "__main__":
    main()