"""
Merge multiple per-language dictionaries into one with non-overlapping macro IDs.

Usage:
    python -m sematok.merge \
        sematok/languages/csharp/dictionary.json \
        sematok/languages/python/dictionary.json \
        --output merged_dictionary.json
"""

import argparse
from pathlib import Path

from sematok.dictionary import CompressionDictionary


def main():
    parser = argparse.ArgumentParser(
        description="Merge multiple sematok dictionaries into one"
    )
    parser.add_argument(
        "dictionaries", nargs="+",
        help="Paths to dictionary JSON files to merge",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output path for merged dictionary",
    )
    args = parser.parse_args()

    dicts = []
    for path in args.dictionaries:
        d = CompressionDictionary.load(path)
        print(f"Loaded {path}: {d.size} patterns, {d.template_count} templates")
        dicts.append(d)

    merged = CompressionDictionary.merge(*dicts)
    merged.save(args.output)

    total_input = sum(d.size + d.template_count for d in dicts)
    total_merged = merged.size + merged.template_count
    deduped = total_input - total_merged

    print(f"\nMerged: {merged.size} patterns, {merged.template_count} templates")
    if deduped > 0:
        print(f"Deduplicated: {deduped} entries shared across dictionaries")
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
