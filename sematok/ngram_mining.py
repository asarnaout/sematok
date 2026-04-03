"""
N-gram substring frequency mining for boilerplate patterns.

Discovers patterns that hand-crafted regex templates miss by counting
all substrings of length 8-120 across the corpus using a two-pass
Apriori-pruned approach.

Pass 1: Count 8-character substrings at word boundaries. Keep survivors.
Pass 2: At each surviving position, extend to full-length patterns.
         Apply quality filters and score.

Usage:
    python -m sematok.ngram_mining --corpus data/raw_cs --language csharp \
        --exclude-repos ppy--osu JamesNK--Newtonsoft.Json nunit--nunit
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm

from sematok.languages import LanguageConfig, get_language
from sematok.lexer import get_safe_ranges, set_language
from sematok.mining import (
    DEFAULT_TOKENIZER,
    MIN_CHAR_LENGTH,
    MIN_FREQUENCY,
    MIN_REPOS,
    MIN_TOKEN_SPAN,
    _get_bpe_token_count,
    _load_file_repo_map,
)

# --- Constants ---

PASS1_THRESHOLD = 15
PASS1_PRUNE_INTERVAL = 5000
PASS2_PRUNE_INTERVAL = 2000
MAX_PATTERN_LENGTH = 120
MAX_WHITESPACE_RATIO = 0.5

_IDENT_CHARS = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_")


def extract_safe_segments(source: str) -> list[str]:
    """
    Extract single-line segments from safe zones.

    Each segment is one line from within one safe zone. N-grams will not
    cross segment or line boundaries.
    """
    try:
        safe_ranges = get_safe_ranges(source)
    except Exception:
        safe_ranges = [(0, len(source.encode("utf-8")))]

    source_bytes = source.encode("utf-8")
    segments = []
    for start, end in safe_ranges:
        chunk = source_bytes[start:end].decode("utf-8", errors="replace")
        for line in chunk.split("\n"):
            stripped = line.strip()
            if len(stripped) >= MIN_CHAR_LENGTH:
                segments.append(stripped)

    return segments


def _is_word_boundary_start(segment: str, pos: int) -> bool:
    """True if pos is at a word boundary (not mid-identifier)."""
    if pos == 0:
        return True
    return segment[pos - 1] not in _IDENT_CHARS


def _is_word_boundary_end(segment: str, pos: int) -> bool:
    """True if pos is at a word boundary for the end of a pattern."""
    if pos == len(segment):
        return True
    return segment[pos] not in _IDENT_CHARS


def _is_valid_ngram(pattern: str) -> bool:
    """Quality check for an n-gram candidate."""
    if len(pattern) < MIN_CHAR_LENGTH:
        return False
    if "\n" in pattern or "\r" in pattern:
        return False
    if not pattern.strip():
        return False
    # Not a single identifier
    if re.fullmatch(r"\w+", pattern):
        return False
    # No more than 50% whitespace
    ws = sum(1 for c in pattern if c in " \t")
    if ws > len(pattern) * MAX_WHITESPACE_RATIO:
        return False
    return True


def _get_corpus_files(
    corpus_dir: Path,
    file_to_repo: dict[str, str],
    exclude_set: set[str],
    max_files: int | None,
    file_extension: str,
) -> list[Path]:
    """Get sorted list of source files, excluding eval repos."""
    files = sorted(corpus_dir.glob(f"*{file_extension}"))
    if exclude_set and file_to_repo:
        files = [
            f for f in files
            if file_to_repo.get(f.name, "unknown") not in exclude_set
        ]
    if max_files:
        files = files[:max_files]
    return files


def ngram_pass1(
    corpus_dir: Path,
    file_to_repo: dict[str, str],
    exclude_set: set[str],
    max_files: int | None,
    file_extension: str,
    pass1_threshold: int = PASS1_THRESHOLD,
) -> set[str]:
    """
    Pass 1: Count 8-grams at word-boundary starts across corpus.

    Returns set of 8-grams appearing in >= pass1_threshold files.
    """
    src_files = _get_corpus_files(corpus_dir, file_to_repo, exclude_set, max_files, file_extension)
    counter: Counter = Counter()

    for file_idx, f in enumerate(tqdm(src_files, desc="N-gram pass 1")):
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        segments = extract_safe_segments(source)
        file_8grams: set[str] = set()
        for seg in segments:
            for i in range(len(seg) - 7):
                if _is_word_boundary_start(seg, i):
                    file_8grams.add(seg[i : i + 8])

        counter.update(file_8grams)

        # Periodic pruning to control memory
        if file_idx > 0 and file_idx % PASS1_PRUNE_INTERVAL == 0:
            cutoff = min(pass1_threshold, max(2, file_idx // (PASS1_PRUNE_INTERVAL // 2)))
            before = len(counter)
            counter = Counter({k: v for k, v in counter.items() if v >= cutoff})
            pruned = before - len(counter)
            if pruned > 0:
                tqdm.write(f"  Pruned {pruned:,} 8-grams at file {file_idx:,}")

    survivors = {gram for gram, count in counter.items() if count >= pass1_threshold}
    return survivors


def ngram_pass2(
    corpus_dir: Path,
    surviving_8grams: set[str],
    file_to_repo: dict[str, str],
    exclude_set: set[str],
    max_files: int | None,
    file_extension: str,
    max_length: int = MAX_PATTERN_LENGTH,
    min_frequency: int = MIN_FREQUENCY,
) -> tuple[Counter, dict[str, set[str]]]:
    """
    Pass 2: Extend surviving 8-grams to full-length patterns.

    At each position where a surviving 8-gram starts, try extending
    to lengths 8 through max_length (or end of line). Only candidates
    that end at word boundaries are kept.

    Returns (pattern_counter, pattern_repos) where counts are per-file.
    """
    src_files = _get_corpus_files(corpus_dir, file_to_repo, exclude_set, max_files, file_extension)
    pattern_counter: Counter = Counter()
    pattern_repos: dict[str, set[str]] = defaultdict(set)

    for file_idx, f in enumerate(tqdm(src_files, desc="N-gram pass 2")):
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        repo = file_to_repo.get(f.name, "unknown")
        segments = extract_safe_segments(source)

        file_patterns: set[str] = set()
        for seg in segments:
            for i in range(len(seg) - 7):
                if seg[i : i + 8] not in surviving_8grams:
                    continue
                if not _is_word_boundary_start(seg, i):
                    continue

                end_limit = min(i + max_length, len(seg))
                for j in range(i + 8, end_limit + 1):
                    if _is_word_boundary_end(seg, j):
                        file_patterns.add(seg[i:j])

        pattern_counter.update(file_patterns)
        for p in file_patterns:
            pattern_repos[p].add(repo)

        # Periodic pruning
        if file_idx > 0 and file_idx % PASS2_PRUNE_INTERVAL == 0:
            cutoff = min(min_frequency, max(3, file_idx // 1000))
            before = len(pattern_counter)
            pruned_keys = {k for k, v in pattern_counter.items() if v < cutoff}
            for k in pruned_keys:
                del pattern_counter[k]
                pattern_repos.pop(k, None)
            pruned = before - len(pattern_counter)
            if pruned > 0:
                tqdm.write(
                    f"  Pruned {pruned:,} patterns at file {file_idx:,} "
                    f"(keeping {len(pattern_counter):,})"
                )

    return pattern_counter, pattern_repos


def filter_and_score_ngrams(
    pattern_counter: Counter,
    pattern_repos: dict[str, set[str]],
    enc,
    min_frequency: int = MIN_FREQUENCY,
    min_repos: int = MIN_REPOS,
    min_token_span: int = MIN_TOKEN_SPAN,
) -> list[tuple[str, int, int, float, int]]:
    """
    Apply quality filters and score n-gram candidates.

    Returns list of (pattern, frequency, token_count, score, repo_count)
    sorted by score descending.
    """
    scored = []
    rejected = {"low_freq": 0, "few_repos": 0, "short_span": 0, "invalid": 0}

    for pattern, freq in pattern_counter.items():
        if freq < min_frequency:
            rejected["low_freq"] += 1
            continue
        if not _is_valid_ngram(pattern):
            rejected["invalid"] += 1
            continue
        repo_count = len(pattern_repos.get(pattern, set()))
        if repo_count < min_repos:
            rejected["few_repos"] += 1
            continue
        token_count = _get_bpe_token_count(pattern, enc)
        if token_count < min_token_span:
            rejected["short_span"] += 1
            continue
        score = freq * (token_count - 1)
        scored.append((pattern, freq, token_count, score, repo_count))

    scored.sort(key=lambda x: x[3], reverse=True)

    print(f"\nN-gram candidates after pass 2: {len(pattern_counter):,}")
    print(f"After filtering: {len(scored)} passed, rejected: {rejected}")

    return scored


def mine_ngram_patterns(
    corpus_dir: Path,
    language: str | LanguageConfig,
    top_n: int = 1000,
    min_frequency: int = MIN_FREQUENCY,
    min_token_span: int = MIN_TOKEN_SPAN,
    min_repos: int = MIN_REPOS,
    max_files: int | None = None,
    exclude_repos: list[str] | None = None,
    pass1_threshold: int = PASS1_THRESHOLD,
    max_pattern_length: int = MAX_PATTERN_LENGTH,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> list[tuple[str, int, int, float, int]]:
    """
    Mine n-gram patterns from a corpus of source files.

    Two-pass approach:
    1. Count 8-grams at word boundaries, keep frequent survivors
    2. Extend survivors to full-length patterns, filter and score

    Returns list of (pattern, frequency, token_count, score, repo_count)
    sorted by score descending.
    """
    lang = get_language(language) if isinstance(language, str) else language
    set_language(lang)
    file_to_repo = _load_file_repo_map(corpus_dir)
    exclude_set = set(exclude_repos) if exclude_repos else set()

    print("N-gram mining: Pass 1 (8-gram census)...")
    survivors = ngram_pass1(
        corpus_dir, file_to_repo, exclude_set, max_files,
        file_extension=lang.file_extension, pass1_threshold=pass1_threshold,
    )
    print(f"  {len(survivors):,} 8-grams survived (threshold={pass1_threshold})")

    if not survivors:
        print("  No survivors -- skipping pass 2")
        return []

    print("N-gram mining: Pass 2 (extend to full patterns)...")
    pattern_counter, pattern_repos = ngram_pass2(
        corpus_dir, survivors, file_to_repo, exclude_set,
        max_files, file_extension=lang.file_extension, max_length=max_pattern_length,
        min_frequency=min_frequency,
    )

    print("N-gram mining: Filtering and scoring...")
    enc = AutoTokenizer.from_pretrained(tokenizer_name)
    scored = filter_and_score_ngrams(
        pattern_counter, pattern_repos, enc,
        min_frequency, min_repos, min_token_span,
    )

    return scored[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Mine n-gram patterns")
    parser.add_argument("--corpus", type=str, required=True, help="Directory with source files")
    parser.add_argument("--language", type=str, required=True, help="Language config to use (e.g. csharp, python)")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER, help="HuggingFace tokenizer for scoring")
    parser.add_argument("--top", type=int, default=100, help="Show top N patterns")
    parser.add_argument("--min-freq", type=int, default=MIN_FREQUENCY)
    parser.add_argument("--min-repos", type=int, default=MIN_REPOS)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--pass1-threshold", type=int, default=PASS1_THRESHOLD)
    parser.add_argument("--max-length", type=int, default=MAX_PATTERN_LENGTH)
    parser.add_argument(
        "--exclude-repos", type=str, nargs="+", default=None,
        help="Repos to exclude from mining",
    )
    args = parser.parse_args()

    results = mine_ngram_patterns(
        Path(args.corpus),
        top_n=args.top,
        min_frequency=args.min_freq,
        min_repos=args.min_repos,
        max_files=args.max_files,
        exclude_repos=args.exclude_repos,
        pass1_threshold=args.pass1_threshold,
        max_pattern_length=args.max_length,
        language=args.language,
        tokenizer_name=args.tokenizer,
    )

    print(f"\nTop {min(args.top, len(results))} n-gram patterns by score:")
    for pattern, freq, tok_count, score, repo_count in results[: args.top]:
        print(
            f"  freq={freq:5d} toks={tok_count:2d} "
            f"score={score:7d} repos={repo_count:2d} | {pattern!r}"
        )


if __name__ == "__main__":
    main()
