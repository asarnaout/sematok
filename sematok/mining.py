"""
Pattern mining: discover frequent multi-token C# boilerplate patterns from a corpus.

Scans C# files, extracts candidate patterns, scores them by
frequency * token_savings, and produces a refined compression dictionary.

Usage:
    python -m sematok.mining --corpus data/raw_cs --output sematok/dictionary.json --top 100
"""

import argparse
import re
from collections import Counter
from pathlib import Path

import tiktoken
from tqdm import tqdm

from sematok.dictionary import CompressionDictionary, SEED_PATTERNS
from sematok.lexer import get_safe_ranges


# Candidate pattern extraction regexes (applied to safe zones only)
CANDIDATE_PATTERNS = [
    # Using directives
    re.compile(r"using\s+[\w.]+;"),
    # Attribute patterns
    re.compile(r"\[[\w]+(?:\([^)]*\))?\]"),
    # Property accessors
    re.compile(r"\{\s*get;\s*(?:(?:private|protected|internal)\s+)?set;\s*\}"),
    re.compile(r"\{\s*get;\s*(?:init|internal\s+set|protected\s+set);\s*\}"),
    # Access modifier combos (2-4 keywords before a type/name)
    re.compile(r"(?:public|private|protected|internal)\s+(?:static\s+)?(?:readonly\s+)?(?:virtual\s+|override\s+|abstract\s+|sealed\s+|async\s+)?(?:partial\s+)?(?:class|struct|interface|enum|void|string|int|bool|Task|Task<\w+>)"),
    # Common method signatures
    re.compile(r"(?:public|private|protected)\s+(?:static\s+)?(?:override\s+)?(?:void|string|int|bool|Task)\s+\w+\([^)]*\)"),
    # Throw patterns
    re.compile(r"throw\s+new\s+\w+(?:Exception|Error)\([^)]*\);"),
    # Common expressions
    re.compile(r"Console\.(?:Write|WriteLine|ReadLine|Read)\("),
    re.compile(r"return\s+Task\.CompletedTask;"),
    re.compile(r"=\s*(?:string\.Empty|new\(\)|default!);"),
    # XML doc
    re.compile(r"///\s*<(?:summary|/summary|param\s+name=\"[^\"]*\"|returns|exception\s+cref=\"[^\"]*\")>"),
    # Generic types
    re.compile(r"(?:IEnumerable|IList|ICollection|IDictionary|Dictionary|List|HashSet|Task|ILogger|IOptions)<"),
    # Namespace/class scaffolding
    re.compile(r"namespace\s+[\w.]+\s*\{"),
]

# Minimum BPE tokens a pattern must span to be worth compressing
MIN_TOKEN_SPAN = 3
# Minimum frequency across corpus to consider a pattern
MIN_FREQUENCY = 5


def _get_bpe_token_count(text: str, enc: tiktoken.Encoding) -> int:
    """Count how many BPE tokens a text requires."""
    return len(enc.encode(text))


def extract_candidates_from_file(source: str) -> list[str]:
    """Extract candidate boilerplate patterns from a C# source file."""
    # Get safe zones to avoid mining from strings/comments
    try:
        safe_ranges = get_safe_ranges(source)
    except Exception:
        # If parsing fails, fall back to the full source
        safe_ranges = [(0, len(source.encode("utf-8")))]

    source_bytes = source.encode("utf-8")
    safe_text_parts = []
    for start, end in safe_ranges:
        safe_text_parts.append(source_bytes[start:end].decode("utf-8", errors="replace"))
    safe_text = "\n".join(safe_text_parts)

    candidates = []
    for pattern_re in CANDIDATE_PATTERNS:
        for match in pattern_re.finditer(safe_text):
            candidate = match.group(0).strip()
            if candidate:
                candidates.append(candidate)

    return candidates


def mine_patterns(
    corpus_dir: Path,
    top_n: int = 100,
    min_frequency: int = MIN_FREQUENCY,
    min_token_span: int = MIN_TOKEN_SPAN,
    max_files: int | None = None,
) -> list[tuple[str, int, int, float]]:
    """
    Mine frequent boilerplate patterns from a corpus of C# files.

    Returns:
        List of (pattern, frequency, token_count, score) tuples sorted by score descending.
        Score = frequency * (token_count - 1), representing total tokens saved across corpus.
    """
    enc = tiktoken.get_encoding("gpt2")

    # Count all candidate patterns across the corpus
    pattern_counter: Counter = Counter()
    cs_files = sorted(corpus_dir.glob("*.cs"))

    if max_files:
        cs_files = cs_files[:max_files]

    print(f"Mining patterns from {len(cs_files)} files...")
    for f in tqdm(cs_files, desc="Scanning"):
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        candidates = extract_candidates_from_file(source)
        pattern_counter.update(candidates)

    # Score and filter patterns
    scored = []
    for pattern, freq in pattern_counter.items():
        if freq < min_frequency:
            continue
        token_count = _get_bpe_token_count(pattern, enc)
        if token_count < min_token_span:
            continue
        # Score: total tokens saved across corpus
        score = freq * (token_count - 1)
        scored.append((pattern, freq, token_count, score))

    # Sort by score descending
    scored.sort(key=lambda x: x[3], reverse=True)
    return scored[:top_n]


def build_mined_dictionary(
    corpus_dir: Path,
    top_n: int = 100,
    include_seeds: bool = True,
    max_files: int | None = None,
) -> CompressionDictionary:
    """
    Build a compression dictionary by combining seed patterns with mined patterns.

    Seed patterns are always included first. Mined patterns fill the remaining
    slots up to top_n total patterns.
    """
    if include_seeds:
        d = CompressionDictionary.from_seed()
        remaining = top_n - d.size
    else:
        d = CompressionDictionary()
        remaining = top_n

    if remaining > 0:
        mined = mine_patterns(corpus_dir, top_n=remaining * 2, max_files=max_files)
        added = 0
        for pattern, freq, tok_count, score in mined:
            if pattern in d.pattern_to_macro:
                continue  # Already in seeds
            d.add_pattern(pattern, category="mined")
            added += 1
            if added >= remaining:
                break
        print(f"Added {added} mined patterns (total: {d.size})")

    return d


def main():
    parser = argparse.ArgumentParser(description="Mine C# boilerplate patterns")
    parser.add_argument("--corpus", type=str, required=True, help="Directory with .cs files")
    parser.add_argument("--output", type=str, default="sematok/dictionary.json")
    parser.add_argument("--top", type=int, default=100, help="Total patterns in dictionary")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--no-seeds", action="store_true", help="Don't include seed patterns")
    args = parser.parse_args()

    d = build_mined_dictionary(
        Path(args.corpus),
        top_n=args.top,
        include_seeds=not args.no_seeds,
        max_files=args.max_files,
    )

    d.save(args.output)
    print(f"\nDictionary saved to {args.output}")
    print(f"Stats: {d.stats()}")

    # Print top patterns
    print("\nTop 20 patterns:")
    mined = mine_patterns(Path(args.corpus), top_n=20, max_files=args.max_files)
    for pattern, freq, tok_count, score in mined[:20]:
        in_dict = "+" if pattern in d.pattern_to_macro else " "
        print(f"  [{in_dict}] freq={freq:5d} toks={tok_count:2d} score={score:6d} | {pattern!r}")


if __name__ == "__main__":
    main()
