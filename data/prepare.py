"""
Prepare fine-tuning data for Unsloth continued pre-training.

Compresses C# files using the 778-pattern macro dictionary and outputs JSONL
with a single "text" field per line. Unsloth handles tokenization, chunking,
and sequence packing internally.

Training data uses a 75/25 compressed/original mix (Token Sugar's validated ratio).
Eval data is 100% compressed (to measure macro comprehension).

Repo-balanced splitting: train on 21 repos, eval on 3 held-out repos.

Usage:
    python -m data.prepare --corpus data/raw_cs --output data/finetune
    python -m data.prepare --corpus data/raw_cs --output data/finetune --compress-ratio 0.75 --seed 42
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from sematok.compressor import Compressor
from sematok.dictionary import CompressionDictionary
from sematok.lexer import get_safe_ranges


DEFAULT_EVAL_REPOS = [
    "ppy--osu",
    "JamesNK--Newtonsoft.Json",
    "nunit--nunit",
]


def _load_file_repo_map(corpus_dir: Path) -> dict[str, str]:
    """Load metadata.jsonl to build filename -> repo mapping."""
    meta_path = corpus_dir / "metadata.jsonl"
    if not meta_path.exists():
        return {}
    file_to_repo = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            file_to_repo[entry["filename"]] = entry["source"]
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

    for f in cs_files:
        repo = file_to_repo.get(f.name)
        if repo is None:
            train_files.append(f)
        elif repo in eval_repo_set:
            eval_files.append(f)
        else:
            train_files.append(f)

    return train_files, eval_files


def _compress_file(source: str, compressor: Compressor) -> str:
    """Compress a C# source file using safe zones."""
    try:
        safe_ranges = get_safe_ranges(source)
        return compressor.compress(source, safe_ranges=safe_ranges)
    except Exception:
        return compressor.compress(source)


def prepare_data(
    corpus_dir: Path,
    output_dir: Path,
    dictionary_path: Path | None = None,
    eval_repos: list[str] | None = None,
    compress_ratio: float = 0.75,
    seed: int = 42,
):
    """
    Prepare JSONL training data for Unsloth continued pre-training.

    Args:
        corpus_dir: Directory containing .cs files and metadata.jsonl.
        output_dir: Where to write train.jsonl, eval.jsonl, meta.json.
        dictionary_path: Path to dictionary JSON. Defaults to sematok/dictionary.json.
        eval_repos: Held-out repos for evaluation split.
        compress_ratio: Fraction of train files to compress (rest kept original).
        seed: Random seed for reproducible train/original sampling.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    if eval_repos is None:
        eval_repos = DEFAULT_EVAL_REPOS

    # Load dictionary
    if dictionary_path and dictionary_path.exists():
        dictionary = CompressionDictionary.load(dictionary_path)
    else:
        default_path = Path("sematok/dictionary.json")
        if default_path.exists():
            dictionary = CompressionDictionary.load(default_path)
        else:
            dictionary = CompressionDictionary.from_seed()
    print(f"Dictionary: {dictionary.size} patterns")

    compressor = Compressor(dictionary)
    rng = random.Random(seed)

    # Collect .cs files
    cs_files = sorted(corpus_dir.glob("*.cs"))
    if not cs_files:
        raise FileNotFoundError(f"No .cs files found in {corpus_dir}")

    # Repo-balanced split
    file_to_repo = _load_file_repo_map(corpus_dir)
    if file_to_repo:
        train_files, eval_files = _split_files_by_repo(cs_files, file_to_repo, eval_repos)
        print(f"Repo-balanced split: {len(train_files)} train, {len(eval_files)} eval")
        print(f"  Eval repos: {', '.join(eval_repos)}")
    else:
        print("Warning: metadata.jsonl not found, using all files for train")
        train_files = cs_files
        eval_files = []

    # Process train files
    train_path = output_dir / "train.jsonl"
    n_compressed = 0
    n_original = 0
    errors = 0
    total_chars_original = 0
    total_chars_compressed = 0

    with open(train_path, "w", encoding="utf-8") as out:
        for f in tqdm(train_files, desc="Train"):
            try:
                source = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                errors += 1
                continue

            total_chars_original += len(source)

            if rng.random() < compress_ratio:
                text = _compress_file(source, compressor)
                n_compressed += 1
            else:
                text = source
                n_original += 1

            total_chars_compressed += len(text)
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Train: {n_compressed} compressed + {n_original} original = {n_compressed + n_original} files")

    # Process eval files (all compressed)
    eval_path = output_dir / "eval.jsonl"
    n_eval = 0
    eval_chars_original = 0
    eval_chars_compressed = 0

    with open(eval_path, "w", encoding="utf-8") as out:
        for f in tqdm(eval_files, desc="Eval"):
            try:
                source = f.read_text(encoding="utf-8", errors="replace")
            except Exception:
                errors += 1
                continue

            eval_chars_original += len(source)
            text = _compress_file(source, compressor)
            eval_chars_compressed += len(text)
            n_eval += 1
            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    print(f"Eval: {n_eval} files (all compressed)")

    if errors > 0:
        print(f"  ({errors} files failed to read)")

    # Compression stats
    all_orig = total_chars_original + eval_chars_original
    all_comp = total_chars_compressed + eval_chars_compressed
    char_reduction = (1 - all_comp / all_orig) * 100 if all_orig > 0 else 0
    print(f"Char reduction (compressed files): {char_reduction:.1f}%")

    # Write metadata
    meta = {
        "dictionary_size": dictionary.size,
        "compress_ratio": compress_ratio,
        "seed": seed,
        "split_method": "repo_balanced" if file_to_repo else "all_train",
        "eval_repos": eval_repos,
        "train_files": n_compressed + n_original,
        "train_compressed": n_compressed,
        "train_original": n_original,
        "eval_files": n_eval,
        "char_reduction_pct": round(char_reduction, 2),
        "errors": errors,
    }
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning data for Unsloth")
    parser.add_argument("--corpus", type=str, required=True, help="Dir with .cs files")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--dictionary", type=str, default=None, help="Dictionary JSON path")
    parser.add_argument(
        "--eval-repos", type=str, nargs="+", default=None,
        help="Held-out repos for evaluation",
    )
    parser.add_argument(
        "--compress-ratio", type=float, default=0.75,
        help="Fraction of train files to compress (default: 0.75)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    prepare_data(
        corpus_dir=Path(args.corpus),
        output_dir=Path(args.output),
        dictionary_path=Path(args.dictionary) if args.dictionary else None,
        eval_repos=args.eval_repos,
        compress_ratio=args.compress_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
