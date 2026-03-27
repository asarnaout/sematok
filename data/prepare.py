"""
Prepare training data: tokenize C# corpus into memory-mapped .bin files.

Runs in two modes:
  - compressed: applies macro compression before tokenizing
  - baseline: tokenizes raw C# code directly

Produces train.bin and val.bin (numpy uint16 arrays) for each mode.

Usage:
    python -m data.prepare --corpus data/raw_cs --mode compressed --output data/compressed
    python -m data.prepare --corpus data/raw_cs --mode baseline --output data/baseline
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm

from sematok.dictionary import CompressionDictionary
from sematok.compressor import Compressor
from sematok.lexer import get_safe_ranges
from tokenizer.extended_tokenizer import ExtendedTokenizer


def prepare_data(
    corpus_dir: Path,
    output_dir: Path,
    mode: str = "compressed",
    dictionary_path: Path | None = None,
    val_split: float = 0.1,
    use_safe_zones: bool = True,
):
    """
    Tokenize corpus and write train/val .bin files.

    Args:
        corpus_dir: Directory containing .cs files
        output_dir: Where to write train.bin and val.bin
        mode: "compressed" or "baseline"
        dictionary_path: Path to dictionary JSON (uses seed if None)
        val_split: Fraction of data for validation
        use_safe_zones: Whether to use tree-sitter safe zones for compression
    """
    output_dir.mkdir(parents=True, exist_ok=True)

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

    print(f"Processing {len(cs_files)} files in '{mode}' mode...")
    print(f"Vocab size: {tokenizer.vocab_size}")

    # Tokenize all files
    all_tokens = []
    total_original_tokens = 0
    total_output_tokens = 0
    errors = 0

    for f in tqdm(cs_files, desc="Tokenizing"):
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            errors += 1
            continue

        # Count original tokens (before compression)
        original_tokens = tokenizer.encode(source)
        total_original_tokens += len(original_tokens)

        if compressor:
            # Apply safe-zone-aware compression
            if use_safe_zones:
                try:
                    safe_ranges = get_safe_ranges(source)
                    text = compressor.compress(source, safe_ranges=safe_ranges)
                except Exception:
                    # If tree-sitter fails, compress without safe zones
                    text = compressor.compress(source)
            else:
                text = compressor.compress(source)
        else:
            text = source

        tokens = tokenizer.encode(text)
        total_output_tokens += len(tokens)
        all_tokens.extend(tokens)

    if errors > 0:
        print(f"  ({errors} files failed to read)")

    print(f"Total tokens: {total_output_tokens:,}")
    if compressor:
        reduction = (1 - total_output_tokens / total_original_tokens) * 100
        print(f"Original tokens: {total_original_tokens:,}")
        print(f"Token reduction: {reduction:.1f}%")

    # Convert to numpy array
    all_tokens = np.array(all_tokens, dtype=np.uint16)

    # Split into train and val
    n_val = int(len(all_tokens) * val_split)
    n_train = len(all_tokens) - n_val

    train_tokens = all_tokens[:n_train]
    val_tokens = all_tokens[n_train:]

    # Write .bin files
    train_path = output_dir / "train.bin"
    val_path = output_dir / "val.bin"
    train_tokens.tofile(train_path)
    val_tokens.tofile(val_path)

    print(f"\nTrain: {n_train:,} tokens -> {train_path}")
    print(f"Val:   {n_val:,} tokens -> {val_path}")

    # Save metadata
    meta = {
        "mode": mode,
        "vocab_size": tokenizer.vocab_size,
        "n_train": n_train,
        "n_val": n_val,
        "n_files": len(cs_files),
        "token_reduction_pct": (
            (1 - total_output_tokens / total_original_tokens) * 100
            if compressor and total_original_tokens > 0 else 0.0
        ),
    }
    meta_path = output_dir / "meta.json"
    import json
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
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--no-safe-zones", action="store_true")
    args = parser.parse_args()

    prepare_data(
        corpus_dir=Path(args.corpus),
        output_dir=Path(args.output),
        mode=args.mode,
        dictionary_path=Path(args.dictionary) if args.dictionary else None,
        val_split=args.val_split,
        use_safe_zones=not args.no_safe_zones,
    )


if __name__ == "__main__":
    main()
