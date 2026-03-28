"""
Measure compression ratio: what fraction of BPE tokens are saved by macro replacement.

Samples files from the corpus, compresses each with the dictionary,
and counts net token savings (each macro replaces N BPE tokens with 1).

Usage:
    python -m sematok.measure --corpus data/raw_cs --dictionary sematok/dictionary.json
"""

import argparse
import random
import re
from pathlib import Path

import tiktoken

from sematok.compressor import Compressor
from sematok.dictionary import CompressionDictionary
from sematok.lexer import get_safe_ranges

MACRO_RE = re.compile(r"<\|M\d{3}\|>")
TEMPLATE_RE = re.compile(r"<\|T(\d{3}):([^|]*)\|>")


def measure_compression(
    corpus_dir: Path,
    dictionary_path: Path,
    sample_size: int = 2000,
    seed: int = 42,
) -> dict:
    """
    Measure compression ratio on a random sample of files.

    Returns dict with total_original, total_savings, total_macros,
    compression_ratio, avg_macros_per_file.
    """
    d = CompressionDictionary.load(dictionary_path)
    compressor = Compressor(d)
    enc = tiktoken.get_encoding("gpt2")

    files = sorted(corpus_dir.glob("*.cs"))
    random.seed(seed)
    sample = random.sample(files, min(sample_size, len(files)))

    total_original = 0
    total_savings = 0
    total_macros = 0
    total_template_macros = 0
    total_template_savings = 0

    for f in sample:
        source = f.read_text(encoding="utf-8", errors="replace")
        total_original += len(enc.encode(source))

        try:
            safe_ranges = get_safe_ranges(source, allow_xmldoc=True)
        except Exception:
            safe_ranges = [(0, len(source.encode("utf-8")))]

        compressed = compressor.compress(source, safe_ranges=safe_ranges)

        # Count exact macro savings
        for macro in MACRO_RE.findall(compressed):
            pattern = d.macro_to_pattern.get(macro, "")
            if pattern:
                total_macros += 1
                total_savings += len(enc.encode(pattern)) - 1

        # Count template macro savings
        for match in TEMPLATE_RE.finditer(compressed):
            macro_base = f"<|T{match.group(1)}|>"
            args = match.group(2).split(",")
            template = d.macro_to_template.get(macro_base, "")
            if template:
                # Reconstruct the original text to measure savings
                expanded = template
                for i, arg in enumerate(args):
                    expanded = expanded.replace(f"{{{i}}}", arg)
                expanded_tokens = len(enc.encode(expanded))
                # The T macro itself is 1 token, plus args are tokenized
                macro_tokens = 1 + len(enc.encode(":" + ",".join(args)))
                saved = expanded_tokens - macro_tokens
                if saved > 0:
                    total_template_macros += 1
                    total_template_savings += saved

    total_all_savings = total_savings + total_template_savings
    return {
        "sample_size": len(sample),
        "total_original_tokens": total_original,
        "exact_macros": total_macros,
        "exact_tokens_saved": total_savings,
        "template_macros": total_template_macros,
        "template_tokens_saved": total_template_savings,
        "total_tokens_saved": total_all_savings,
        "compression_ratio": total_all_savings / total_original if total_original else 0,
        "avg_macros_per_file": (total_macros + total_template_macros) / len(sample) if sample else 0,
    }


def main():
    parser = argparse.ArgumentParser(description="Measure compression ratio")
    parser.add_argument("--corpus", type=str, required=True, help="Directory with .cs files")
    parser.add_argument("--dictionary", type=str, default="sematok/dictionary.json")
    parser.add_argument("--sample", type=int, default=2000, help="Number of files to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    results = measure_compression(
        Path(args.corpus), Path(args.dictionary), args.sample, args.seed,
    )

    print(f"Sample: {results['sample_size']} files")
    print(f"Total original tokens: {results['total_original_tokens']:,}")
    print(f"Exact macros: {results['exact_macros']:,} ({results['exact_tokens_saved']:,} tokens saved)")
    print(f"Template macros: {results['template_macros']:,} ({results['template_tokens_saved']:,} tokens saved)")
    print(f"Total tokens saved: {results['total_tokens_saved']:,}")
    print(f"Compression ratio: {results['compression_ratio'] * 100:.2f}%")
    print(f"Avg macros per file: {results['avg_macros_per_file']:.1f}")


if __name__ == "__main__":
    main()
