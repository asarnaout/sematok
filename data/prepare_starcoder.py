"""
Download and compress StarCoderData for supplemental training.

Streams the C# subset of bigcode/starcoderdata, compresses each file
with the existing sematok dictionary, and appends to the training JSONL.
Does NOT touch eval data.

Usage:
    python -m data.prepare_starcoder \
        --language csharp \
        --output data/finetune/csharp/train_starcoder.jsonl \
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

    with open(output_path, "w", encoding="utf-8") as out:
        for i, record in enumerate(tqdm(ds, desc="Processing", total=max_files)):
            if i >= max_files:
                break

            content = record.get("content", "")
            if not content or len(content) < 50:
                n_skipped += 1
                continue

            if rng.random() < compress_ratio:
                text = _compress_file(content, compressor)
                n_compressed += 1
            else:
                text = content
                n_original += 1

            out.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")

    total = n_compressed + n_original
    print(f"\nDone: {total:,} files written to {output_path}")
    print(f"  Compressed: {n_compressed:,}")
    print(f"  Original: {n_original:,}")
    print(f"  Skipped (too short): {n_skipped:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Prepare StarCoderData for supplemental training"
    )
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--max-files", type=int, default=1_000_000)
    parser.add_argument("--compress-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
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
