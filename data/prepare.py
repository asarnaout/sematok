"""
Prepare training data: tokenize C# corpus into memory-mapped .bin files.

Runs in two modes:
  - compressed: applies macro compression before tokenizing
  - baseline: tokenizes raw C# code directly

Uses repo-balanced splitting: train on ~21 repos, evaluate on 3 held-out repos.
This ensures evaluation measures generalization across codebases, not memorization.

Usage:
    python -m data.prepare --corpus data/raw_cs --mode compressed --output data/compressed
    python -m data.prepare --corpus data/raw_cs --mode baseline --output data/baseline
    python -m data.prepare --corpus data/raw_cs --mode compressed --output data/compressed --eval-repos ppy--osu JamesNK--Newtonsoft.Json nunit--nunit
"""

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sematok.dictionary import CompressionDictionary
from sematok.compressor import Compressor
from sematok.lexer import get_safe_ranges
from tokenizer.extended_tokenizer import ExtendedTokenizer


# Default held-out repos for evaluation (diverse domains, ~7% of corpus)
DEFAULT_EVAL_REPOS = [
    "ppy--osu",                 # Game code (4,707 files)
    "JamesNK--Newtonsoft.Json", # Library code (910 files)
    "nunit--nunit",             # Test framework (1,040 files)
]


def _load_file_repo_map(corpus_dir: Path) -> dict[str, str]:
    """Load metadata.jsonl to build filename -> repo mapping."""
    meta_path = corpus_dir / "metadata.jsonl"
    if not meta_path.exists():
        return {}
    file_to_repo = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            m = json.loads(line)
            file_to_repo[m["filename"]] = m["source"]
    return file_to_repo


def _split_files_by_repo(
    cs_files: list[Path],
    file_to_repo: dict[str, str],
    eval_repos: list[str],
) -> tuple[list[Path], list[Path]]:
    """Split files into train/eval based on repo membership."""
    eval_repo_set = set(eval_repos)
    train_files = []
    eval_files = []
    unmapped = 0

    for f in cs_files:
        repo = file_to_repo.get(f.name)
        if repo is None:
            unmapped += 1
            train_files.append(f)  # unmapped files go to train
        elif repo in eval_repo_set:
            eval_files.append(f)
        else:
            train_files.append(f)

    if unmapped > 0:
        print(f"  Warning: {unmapped} files not in metadata.jsonl (added to train)")

    return train_files, eval_files


def _tokenize_files(
    files: list[Path],
    tokenizer: ExtendedTokenizer,
    compressor: Compressor | None,
    use_safe_zones: bool,
    desc: str,
) -> tuple[list[int], int, int, int]:
    """Tokenize a list of files. Returns (tokens, original_count, output_count, errors)."""
    all_tokens = []
    total_original = 0
    total_output = 0
    errors = 0

    for f in tqdm(files, desc=desc):
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            errors += 1
            continue

        original_tokens = tokenizer.encode(source)
        total_original += len(original_tokens)

        if compressor:
            if use_safe_zones:
                try:
                    safe_ranges = get_safe_ranges(source)
                    text = compressor.compress(source, safe_ranges=safe_ranges)
                except Exception:
                    text = compressor.compress(source)
            else:
                text = compressor.compress(source)
        else:
            text = source

        tokens = tokenizer.encode(text)
        total_output += len(tokens)
        all_tokens.extend(tokens)

    return all_tokens, total_original, total_output, errors


def prepare_data(
    corpus_dir: Path,
    output_dir: Path,
    mode: str = "compressed",
    dictionary_path: Path | None = None,
    eval_repos: list[str] | None = None,
    use_safe_zones: bool = True,
):
    """
    Tokenize corpus and write train.bin and val.bin files.

    Uses repo-balanced splitting: files from eval_repos go to val.bin,
    all other files go to train.bin.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if eval_repos is None:
        eval_repos = DEFAULT_EVAL_REPOS

    # Load dictionary
    if dictionary_path and dictionary_path.exists():
        dictionary = CompressionDictionary.load(dictionary_path)
        print(f"Loaded dictionary from {dictionary_path} ({dictionary.size} patterns)")
    else:
        dictionary = CompressionDictionary.from_seed()
        print(f"Using seed dictionary ({dictionary.size} patterns)")

    tokenizer = ExtendedTokenizer(dictionary)
    compressor = Compressor(dictionary) if mode == "compressed" else None

    # Collect all .cs files
    cs_files = sorted(corpus_dir.glob("*.cs"))
    if not cs_files:
        raise FileNotFoundError(f"No .cs files found in {corpus_dir}")

    # Repo-balanced split
    file_to_repo = _load_file_repo_map(corpus_dir)
    if file_to_repo:
        train_files, eval_files = _split_files_by_repo(cs_files, file_to_repo, eval_repos)
        print(f"Repo-balanced split: {len(train_files)} train files ({len(cs_files) - len(eval_files)} repos), "
              f"{len(eval_files)} eval files ({len(eval_repos)} held-out repos)")
        print(f"  Eval repos: {', '.join(eval_repos)}")
    else:
        # Fallback: random 90/10 split if no metadata
        print("Warning: metadata.jsonl not found, falling back to random 90/10 split")
        n_eval = max(1, int(len(cs_files) * 0.1))
        train_files = cs_files[:-n_eval]
        eval_files = cs_files[-n_eval:]

    print(f"Mode: '{mode}' | Vocab size: {tokenizer.vocab_size}")

    # Tokenize train split
    train_tokens, train_orig, train_out, train_err = _tokenize_files(
        train_files, tokenizer, compressor, use_safe_zones, "Tokenizing train"
    )

    # Tokenize eval split
    eval_tokens, eval_orig, eval_out, eval_err = _tokenize_files(
        eval_files, tokenizer, compressor, use_safe_zones, "Tokenizing eval"
    )

    total_errors = train_err + eval_err
    if total_errors > 0:
        print(f"  ({total_errors} files failed to read)")

    # Print stats
    print(f"\nTrain tokens: {train_out:,}")
    print(f"Eval tokens:  {eval_out:,}")
    if compressor:
        total_orig = train_orig + eval_orig
        total_out = train_out + eval_out
        reduction = (1 - total_out / total_orig) * 100 if total_orig > 0 else 0
        print(f"Original tokens (total): {total_orig:,}")
        print(f"Token reduction: {reduction:.1f}%")

    # Write .bin files
    train_arr = np.array(train_tokens, dtype=np.uint16)
    eval_arr = np.array(eval_tokens, dtype=np.uint16)

    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    train_arr.tofile(train_path)
    eval_arr.tofile(val_path)

    print(f"\nTrain: {len(train_arr):,} tokens -> {train_path}")
    print(f"Val:   {len(eval_arr):,} tokens -> {val_path}")

    # Save metadata
    meta = {
        "mode": mode,
        "vocab_size": tokenizer.vocab_size,
        "n_train": len(train_arr),
        "n_val": len(eval_arr),
        "n_train_files": len(train_files),
        "n_eval_files": len(eval_files),
        "eval_repos": eval_repos,
        "split_method": "repo_balanced" if file_to_repo else "random",
        "token_reduction_pct": (
            (1 - (train_out + eval_out) / (train_orig + eval_orig)) * 100
            if compressor and (train_orig + eval_orig) > 0 else 0.0
        ),
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare training data")
    parser.add_argument("--corpus", type=str, required=True, help="Dir with .cs files")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument(
        "--mode", type=str, default="compressed", choices=["compressed", "baseline"],
    )
    parser.add_argument("--dictionary", type=str, default=None, help="Dictionary JSON path")
    parser.add_argument(
        "--eval-repos", type=str, nargs="+", default=None,
        help="Held-out repos for evaluation (default: osu, Newtonsoft.Json, nunit)",
    )
    parser.add_argument("--no-safe-zones", action="store_true")
    args = parser.parse_args()

    prepare_data(
        corpus_dir=Path(args.corpus),
        output_dir=Path(args.output),
        mode=args.mode,
        dictionary_path=Path(args.dictionary) if args.dictionary else None,
        eval_repos=args.eval_repos,
        use_safe_zones=not args.no_safe_zones,
    )


if __name__ == "__main__":
    main()
