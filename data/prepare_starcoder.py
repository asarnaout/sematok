"""
Prepare training data from StarCoderData.

Streams a language subset of bigcode/starcoderdata, compresses each file
with the existing sematok dictionary, and writes training JSONL. This is
the primary source of training data -- eval data is prepared separately
from the curated corpus using data.prepare --eval-only.

Filters out files from eval repos to prevent data leakage.

Training data uses a 75/25 compressed/original mix (Token Sugar's ratio).

Usage:
    python -m data.prepare_starcoder \\
        --language csharp \\
        --output data/finetune/csharp/train.jsonl \\
        --max-files 1000000
"""

import argparse
import json
import random
from pathlib import Path

from tqdm import tqdm

from sematok.compressor import Compressor
from sematok.dictionary import CompressionDictionary
from sematok.languages import LanguageConfig, get_language, get_dictionary_path
from sematok.lexer import get_safe_ranges, set_language


STARCODER_DATASET = "bigcode/starcoderdata"


def _compress_file(source: str, compressor: Compressor) -> str:
    """Compress a source file using safe zones."""
    try:
        safe_ranges = get_safe_ranges(source)
        return compressor.compress(source, safe_ranges=safe_ranges)
    except Exception:
        return compressor.compress(source)


def _build_eval_repo_filter(eval_repos: list[str]) -> set[str]:
    """Build a set of repo names to exclude from training.

    Eval repos in language configs use "owner--repo" format.
    StarCoderData uses "owner/repo" in max_stars_repo_name.
    We normalize both to "owner/repo" for matching.
    """
    normalized = set()
    for repo in eval_repos:
        # "microsoft--garnet" -> "microsoft/garnet"
        normalized.add(repo.replace("--", "/"))
        # Also keep original in case format varies
        normalized.add(repo)
    return normalized


def prepare_starcoder(
    language: str,
    output_path: Path,
    max_files: int = 1_000_000,
    compress_ratio: float = 0.75,
    seed: int = 42,
):
    lang = get_language(language)
    set_language(lang)

    # Map language names to StarCoderData data_dir names
    lang_to_datadir = {
        "csharp": "c-sharp",
        "python": "python",
        "java": "java",
        "typescript": "typescript",
        "go": "go",
    }
    data_dir = lang_to_datadir.get(lang.name)
    if data_dir is None:
        raise ValueError(f"No StarCoderData mapping for language '{lang.name}'")

    # Load dictionary
    dict_path = get_dictionary_path(lang.name)
    if dict_path and dict_path.exists():
        dictionary = CompressionDictionary.load(dict_path)
    else:
        raise FileNotFoundError(f"No dictionary found for {lang.name}")
    print(f"Dictionary: {dictionary.size} patterns")

    compressor = Compressor(dictionary, language=lang)
    rng = random.Random(seed)

    # Build eval repo filter to prevent data leakage
    eval_repos = lang.eval_repos
    eval_filter = _build_eval_repo_filter(eval_repos)
    print(f"Filtering out {len(eval_repos)} eval repos to prevent data leakage")

    # Stream StarCoderData
    from datasets import load_dataset
    print(f"\nStreaming {STARCODER_DATASET} ({data_dir})...")
    ds = load_dataset(
        STARCODER_DATASET,
        data_dir=data_dir,
        split="train",
        streaming=True,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_compressed = 0
    n_original = 0
    n_skipped = 0
    n_filtered = 0
    n_seen = 0

    with open(output_path, "w", encoding="utf-8") as out:
        for record in tqdm(ds, desc="Processing", total=max_files):
            n_seen += 1

            content = record.get("content", "")
            if not content or len(content) < 50:
                n_skipped += 1
                continue

            # Filter out eval repos
            repo_name = record.get("max_stars_repo_name", "")
            if repo_name in eval_filter:
                n_filtered += 1
                continue

            if rng.random() < compress_ratio:
                text = _compress_file(content, compressor)
                n_compressed += 1
            else:
                text = content
                n_original += 1

            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

            if n_compressed + n_original >= max_files:
                break

    total = n_compressed + n_original
    print(f"\nDone: {total:,} files written to {output_path}")
    print(f"  Compressed: {n_compressed:,}")
    print(f"  Original: {n_original:,}")
    print(f"  Skipped (too short): {n_skipped:,}")
    print(f"  Filtered (eval repos): {n_filtered:,}")
    print(f"  Total streamed: {n_seen:,}")

    # Write metadata
    meta_path = output_path.parent / "meta_starcoder.json"
    meta = {
        "source": STARCODER_DATASET,
        "data_dir": data_dir,
        "max_files": max_files,
        "total_written": total,
        "compressed": n_compressed,
        "original": n_original,
        "skipped": n_skipped,
        "filtered_eval_repos": n_filtered,
        "total_streamed": n_seen,
        "compress_ratio": compress_ratio,
        "seed": seed,
        "eval_repos_excluded": eval_repos,
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"Metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare training data from StarCoderData"
    )
    parser.add_argument("--language", type=str, required=True,
                        help="Language config (e.g. csharp, python)")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL path (e.g. data/finetune/csharp/train.jsonl)")
    parser.add_argument("--max-files", type=int, default=1_000_000,
                        help="Maximum files to process (default: 1,000,000)")
    parser.add_argument("--compress-ratio", type=float, default=0.75,
                        help="Fraction of files to compress (default: 0.75)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    prepare_starcoder(
        language=args.language,
        output_path=Path(args.output),
        max_files=args.max_files,
        compress_ratio=args.compress_ratio,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
