"""
Measure compression ratio: what fraction of BPE tokens are saved by macro replacement.

Samples files from the corpus, compresses each with the dictionary,
and counts net token savings (each macro replaces N BPE tokens with 1).

Usage:
    python -m sematok.measure --corpus data/raw_cs --language csharp
"""

import argparse
import random
import re
from pathlib import Path

from transformers import AutoTokenizer

from sematok.compressor import Compressor
from sematok.dictionary import CompressionDictionary
from sematok.languages import LanguageConfig, get_language
from sematok.lexer import get_safe_ranges, set_language

MACRO_RE = re.compile(r"<\|M\d+\|>")
TEMPLATE_RE = re.compile(r"<\|T(\d+):([^|]*)\|>")

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def measure_compression(
    corpus_dir: Path,
    dictionary_path: Path,
    sample_size: int = 2000,
    seed: int = 42,
    language: str | LanguageConfig = "csharp",
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> dict:
    """
    Measure compression ratio on a random sample of files.

    Returns dict with total_original, total_savings, total_macros,
    compression_ratio, avg_macros_per_file.
    """
    lang = get_language(language) if isinstance(language, str) else language
    set_language(lang)
    d = CompressionDictionary.load(dictionary_path)
    compressor = Compressor(d, language=lang)
    enc = AutoTokenizer.from_pretrained(tokenizer_name)

    files = sorted(corpus_dir.glob(f"*{lang.file_extension}"))
    random.seed(seed)
    sample = random.sample(files, min(sample_size, len(files)))

    total_original = 0
    total_savings = 0
    total_macros = 0
    total_template_macros = 0
    total_template_savings = 0

    for f in sample:
        source = f.read_text(encoding="utf-8", errors="replace")
        total_original += len(enc.encode(source, add_special_tokens=False))

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
                total_savings += len(enc.encode(pattern, add_special_tokens=False)) - 1

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
                expanded_tokens = len(enc.encode(expanded, add_special_tokens=False))
                # The T macro itself is 1 token, plus args are tokenized
                macro_tokens = 1 + len(enc.encode(":" + ",".join(args), add_special_tokens=False))
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
    parser.add_argument("--corpus", type=str, required=True, help="Directory with source files")
    parser.add_argument("--dictionary", type=str, default=None, help="Dictionary JSON (default: auto-detect from language)")
    parser.add_argument("--language", type=str, default="csharp", help="Language config to use")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER, help="HuggingFace tokenizer for scoring")
    parser.add_argument("--sample", type=int, default=2000, help="Number of files to sample")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from sematok.languages import get_dictionary_path
    dict_path = args.dictionary
    if dict_path is None:
        resolved = get_dictionary_path(args.language)
        if resolved is None:
            raise FileNotFoundError(f"No dictionary found for language '{args.language}'")
        dict_path = str(resolved)

    results = measure_compression(
        Path(args.corpus), Path(dict_path), args.sample, args.seed,
        language=args.language, tokenizer_name=args.tokenizer,
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
