"""
Prepare evaluation data from the curated corpus.

Compresses source files from held-out repos using the macro dictionary
and outputs eval.jsonl (100% compressed) for measuring macro comprehension.

Training data is prepared separately using data.prepare_starcoder, which
streams from bigcode/starcoderdata for much larger scale training.

Usage:
    # Generate eval data only (recommended -- use prepare_starcoder for training):
    python -m data.prepare --corpus data/raw_csharp --output data/finetune/csharp --language csharp --eval-only

    # Generate both train and eval from curated corpus (small-scale testing):
    python -m data.prepare --corpus data/raw_csharp --output data/finetune/csharp --language csharp
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from sematok.compressor import Compressor
from sematok.dictionary import CompressionDictionary
from sematok.languages import LanguageConfig, get_language
from sematok.lexer import get_safe_ranges, set_language


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
    """Compress a source file using safe zones."""
    try:
        safe_ranges = get_safe_ranges(source)
        return compressor.compress(source, safe_ranges=safe_ranges)
    except Exception:
        return compressor.compress(source)


def prepare_data(
    corpus_dir: Path,
    output_dir: Path,
    language: str | LanguageConfig,
    dictionary_path: Path | None = None,
    eval_repos: list[str] | None = None,
    compress_ratio: float = 0.75,
    seed: int = 42,
    eval_only: bool = False,
):
    """
    Prepare fine-tuning data from the curated corpus.

    Args:
        corpus_dir: Directory containing source files and metadata.jsonl.
        output_dir: Where to write eval.jsonl (and optionally train.jsonl), meta.json.
        dictionary_path: Path to dictionary JSON. Defaults to language's built-in dictionary.
        eval_repos: Held-out repos for evaluation split.
        compress_ratio: Fraction of train files to compress (rest kept original).
        seed: Random seed for reproducible train/original sampling.
        language: Language name or LanguageConfig instance.
        eval_only: If True, only generate eval.jsonl (use prepare_starcoder for training).
    """
    lang = get_language(language) if isinstance(language, str) else language
    set_language(lang)
    output_dir.mkdir(parents=True, exist_ok=True)
    if eval_repos is None:
        eval_repos = lang.eval_repos

    # Load dictionary
    from sematok.languages import get_dictionary_path
    if dictionary_path and dictionary_path.exists():
        dictionary = CompressionDictionary.load(dictionary_path)
    else:
        default_path = get_dictionary_path(lang.name)
        if default_path and default_path.exists():
            dictionary = CompressionDictionary.load(default_path)
        else:
            dictionary = CompressionDictionary.from_seed(language=lang.name)
    print(f"Dictionary: {dictionary.size} patterns")

    compressor = Compressor(dictionary, language=lang)
    rng = random.Random(seed)

    # Collect source files
    source_files = sorted(corpus_dir.glob(f"*{lang.file_extension}"))
    if not source_files:
        raise FileNotFoundError(f"No {lang.file_extension} files found in {corpus_dir}")

    # Repo-balanced split
    file_to_repo = _load_file_repo_map(corpus_dir)
    if file_to_repo:
        train_files, eval_files = _split_files_by_repo(source_files, file_to_repo, eval_repos)
        print(f"Repo-balanced split: {len(train_files)} train, {len(eval_files)} eval")
        print(f"  Eval repos: {', '.join(eval_repos)}")
    else:
        print("Warning: metadata.jsonl not found, using all files for train")
        train_files = source_files
        eval_files = []

    # Process train files (skip if --eval-only)
    n_compressed = 0
    n_original = 0
    errors = 0
    total_chars_original = 0
    total_chars_compressed = 0

    if eval_only:
        print("Skipping training data (--eval-only). Use prepare_starcoder for training.")
    else:
        train_path = output_dir / "train.jsonl"
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
    if all_orig > 0:
        print(f"Char reduction (compressed files): {char_reduction:.1f}%")

    # Write metadata
    meta = {
        "dictionary_size": dictionary.size,
        "compress_ratio": compress_ratio,
        "seed": seed,
        "split_method": "repo_balanced" if file_to_repo else "all_train",
        "eval_only": eval_only,
        "eval_repos": eval_repos,
        "eval_files": n_eval,
        "errors": errors,
    }
    if not eval_only:
        meta["train_files"] = n_compressed + n_original
        meta["train_compressed"] = n_compressed
        meta["train_original"] = n_original
        meta["char_reduction_pct"] = round(char_reduction, 2)
    meta_path = output_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare fine-tuning data")
    parser.add_argument("--corpus", type=str, required=True, help="Dir with source files")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--language", type=str, required=True, help="Language config to use (e.g. csharp, python)")
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
    parser.add_argument(
        "--eval-only", action="store_true",
        help="Only generate eval.jsonl (use prepare_starcoder for training data)",
    )
    args = parser.parse_args()

    prepare_data(
        corpus_dir=Path(args.corpus),
        output_dir=Path(args.output),
        dictionary_path=Path(args.dictionary) if args.dictionary else None,
        eval_repos=args.eval_repos,
        compress_ratio=args.compress_ratio,
        seed=args.seed,
        language=args.language,
        eval_only=args.eval_only,
    )


if __name__ == "__main__":
    main()
