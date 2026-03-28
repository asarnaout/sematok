"""
Pattern mining: discover frequent multi-token C# boilerplate patterns from a corpus.

Scans C# files, extracts candidate patterns, scores them by
frequency * token_savings, and produces a refined compression dictionary.

Quality filters:
- Must appear in at least 2 repos (no repo-specific patterns)
- Must be at least 8 characters (no short junk like [0], [i])
- Must span at least 3 BPE tokens (not worth compressing otherwise)
- Must appear at least 50 times across the corpus
- No embedded newlines (regex artifacts)
- No pure-logic patterns (only boilerplate/scaffolding)

Usage:
    python -m sematok.mining --corpus data/raw_cs --output sematok/dictionary.json
"""

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm

from sematok.dictionary import CompressionDictionary
from sematok.languages import get_language
from sematok.lexer import get_safe_ranges

QWEN_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


# Loaded from language config. Module-level ref kept for imports by other modules.
CANDIDATE_PATTERNS = get_language("csharp").candidate_patterns


# --- Quality filter thresholds ---

# Minimum BPE tokens a pattern must span to be worth compressing
MIN_TOKEN_SPAN = 3
# Minimum frequency across corpus to consider a pattern
MIN_FREQUENCY = 50
# Minimum character length for a pattern
MIN_CHAR_LENGTH = 8
# Minimum number of distinct repos a pattern must appear in
MIN_REPOS = 2


def _get_bpe_token_count(text: str, enc) -> int:
    """Count how many BPE tokens a text requires."""
    return len(enc.encode(text, add_special_tokens=False))


def _load_file_repo_map(corpus_dir: Path) -> dict[str, str]:
    """Load metadata.jsonl to map filename -> repo source."""
    meta_path = corpus_dir / "metadata.jsonl"
    file_to_repo = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                file_to_repo[entry["filename"]] = entry["source"]
    return file_to_repo


def _is_valid_pattern(pattern: str) -> bool:
    """Quick quality check before counting a candidate."""
    # No embedded newlines
    if "\n" in pattern or "\r" in pattern:
        return False
    # Minimum length
    if len(pattern) < MIN_CHAR_LENGTH:
        return False
    # Not pure whitespace
    if not pattern.strip():
        return False
    # Not a single word (too generic)
    if re.fullmatch(r"\w+", pattern):
        return False
    return True


def extract_candidates_from_file(source: str) -> list[str]:
    """Extract candidate boilerplate patterns from a C# source file."""
    try:
        safe_ranges = get_safe_ranges(source, allow_xmldoc=True)
    except Exception:
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
            if candidate and _is_valid_pattern(candidate):
                candidates.append(candidate)

    return candidates


def mine_patterns(
    corpus_dir: Path,
    top_n: int = 1000,
    min_frequency: int = MIN_FREQUENCY,
    min_token_span: int = MIN_TOKEN_SPAN,
    min_repos: int = MIN_REPOS,
    max_files: int | None = None,
    exclude_repos: list[str] | None = None,
) -> list[tuple[str, int, int, float, int]]:
    """
    Mine frequent boilerplate patterns from a corpus of C# files.

    Args:
        exclude_repos: Repo names to skip (e.g. held-out eval repos).
            This ensures the dictionary is built only from training data.

    Returns:
        List of (pattern, frequency, token_count, score, repo_count) tuples
        sorted by score descending.
        Score = frequency * (token_count - 1), representing total tokens saved.
    """
    enc = AutoTokenizer.from_pretrained(QWEN_MODEL)

    # Load repo mapping for cross-repo validation
    file_to_repo = _load_file_repo_map(corpus_dir)
    exclude_set = set(exclude_repos) if exclude_repos else set()

    # Track global frequency and per-repo presence
    pattern_counter: Counter = Counter()
    pattern_repos: dict[str, set[str]] = defaultdict(set)

    cs_files = sorted(corpus_dir.glob("*.cs"))

    # Exclude eval repo files from mining
    if exclude_set and file_to_repo:
        before = len(cs_files)
        cs_files = [f for f in cs_files if file_to_repo.get(f.name, "unknown") not in exclude_set]
        print(f"Excluded {before - len(cs_files)} files from {len(exclude_set)} eval repos")

    if max_files:
        cs_files = cs_files[:max_files]

    print(f"Mining patterns from {len(cs_files)} files...")
    for f in tqdm(cs_files, desc="Scanning"):
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        repo = file_to_repo.get(f.name, "unknown")
        candidates = extract_candidates_from_file(source)

        # Deduplicate per file: count each pattern at most once per file
        unique_candidates = set(candidates)
        pattern_counter.update(unique_candidates)
        for c in unique_candidates:
            pattern_repos[c].add(repo)

    # Score and filter patterns
    scored = []
    rejected = {"low_freq": 0, "few_repos": 0, "short_span": 0}

    for pattern, freq in pattern_counter.items():
        if freq < min_frequency:
            rejected["low_freq"] += 1
            continue
        repo_count = len(pattern_repos[pattern])
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

    print(f"\nCandidate patterns found: {len(pattern_counter)}")
    print(f"After filtering: {len(scored)} passed, rejected: {rejected}")

    return scored[:top_n]


def merge_mining_results(
    regex_results: list[tuple[str, int, int, float, int]],
    ngram_results: list[tuple[str, int, int, float, int]],
) -> list[tuple[str, int, int, float, int]]:
    """
    Merge regex-mined and n-gram-mined patterns.

    Deduplicates by pattern string, keeping the entry with higher frequency.
    Returns merged list sorted by score descending.
    """
    by_pattern: dict[str, tuple[str, int, int, float, int]] = {}
    for result_list in [regex_results, ngram_results]:
        for entry in result_list:
            pattern = entry[0]
            if pattern not in by_pattern or entry[1] > by_pattern[pattern][1]:
                by_pattern[pattern] = entry
    return sorted(by_pattern.values(), key=lambda x: x[3], reverse=True)


def deduplicate_substrings(
    results: list[tuple[str, int, int, float, int]],
) -> list[tuple[str, int, int, float, int]]:
    """
    Remove patterns that are substrings of longer patterns.

    Processes longest patterns first. A shorter pattern is dropped if it
    appears as a substring of any already-accepted longer pattern.
    This eliminates fragment overlap (e.g. five partial for-loop variants)
    and near-duplicates differing by trailing punctuation.

    Returns deduplicated list sorted by score descending.
    """
    # Process longest patterns first -- they subsume shorter ones
    by_length = sorted(results, key=lambda x: len(x[0]), reverse=True)

    kept = []
    kept_patterns: list[str] = []

    for entry in by_length:
        pattern = entry[0]
        if any(pattern in kp for kp in kept_patterns):
            continue
        kept.append(entry)
        kept_patterns.append(pattern)

    kept.sort(key=lambda x: x[3], reverse=True)
    before = len(results)
    print(f"Substring dedup: {before} -> {len(kept)} patterns ({before - len(kept)} removed)")
    return kept


def _score_on_corpus(
    d: CompressionDictionary,
    corpus_dir: Path,
    sample_size: int = 2000,
    seed: int = 42,
    exclude_repos: list[str] | None = None,
) -> dict[str, int]:
    """
    Score every dictionary entry by actual corpus impact using Qwen's tokenizer.

    Compresses a sample of files and counts how many Qwen BPE tokens each
    macro saves. Returns {macro_or_template: total_tokens_saved}.
    """
    from sematok.compressor import Compressor

    enc = AutoTokenizer.from_pretrained(QWEN_MODEL)
    compressor = Compressor(d)

    file_to_repo = _load_file_repo_map(corpus_dir)
    exclude_set = set(exclude_repos) if exclude_repos else set()

    files = sorted(corpus_dir.glob("*.cs"))
    if exclude_set and file_to_repo:
        files = [f for f in files if file_to_repo.get(f.name, "unknown") not in exclude_set]

    random.seed(seed)
    sample = random.sample(files, min(sample_size, len(files)))

    macro_re = re.compile(r"<\|M\d{3}\|>")
    template_re = re.compile(r"<\|T(\d{3}):([^|]*)\|>")

    scores: Counter = Counter()

    print(f"\nScoring {len(sample)} files for corpus impact...")
    for f in tqdm(sample, desc="Scoring"):
        source = f.read_text(encoding="utf-8", errors="replace")
        try:
            safe_ranges = get_safe_ranges(source, allow_xmldoc=True)
        except Exception:
            safe_ranges = [(0, len(source.encode("utf-8")))]

        compressed = compressor.compress(source, safe_ranges=safe_ranges)

        for macro in macro_re.findall(compressed):
            pattern = d.macro_to_pattern.get(macro, "")
            if pattern:
                saving = len(enc.encode(pattern, add_special_tokens=False)) - 1
                if saving > 0:
                    scores[macro] += saving

        for match in template_re.finditer(compressed):
            macro_base = f"<|T{match.group(1)}|>"
            args = match.group(2).split(",")
            template = d.macro_to_template.get(macro_base, "")
            if template:
                expanded = template
                for i, arg in enumerate(args):
                    expanded = expanded.replace(f"{{{i}}}", arg)
                expanded_tokens = len(enc.encode(expanded, add_special_tokens=False))
                macro_tokens = 1 + len(enc.encode(":" + ",".join(args), add_special_tokens=False))
                saving = expanded_tokens - macro_tokens
                if saving > 0:
                    scores[macro_base] += saving

    return dict(scores)


def _rebuild_top_n(
    d: CompressionDictionary,
    scores: dict[str, int],
    top_n: int,
) -> CompressionDictionary:
    """
    Rebuild a dictionary with only the top N entries by corpus impact score.

    Entries that never fired (score=0) are dropped. Remaining entries are
    ranked by total Qwen tokens saved and assigned fresh sequential macro IDs.
    """
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    new_d = CompressionDictionary()
    kept_patterns = 0
    kept_templates = 0

    for macro, score in ranked:
        if kept_patterns + kept_templates >= top_n:
            break

        if macro in d.macro_to_pattern:
            pattern = d.macro_to_pattern[macro]
            category = d.pattern_categories.get(pattern, "mined")
            new_d.add_pattern(pattern, category=category)
            kept_patterns += 1
        elif macro in d.macro_to_template:
            template = d.macro_to_template[macro]
            slots = d.template_slots[template]
            category = d.template_categories.get(template, "template")
            new_d.add_template(template, slots, category=category)
            kept_templates += 1

    total = kept_patterns + kept_templates
    print(f"\nTrimmed to top {total}: {kept_patterns} exact + {kept_templates} templates")
    if ranked:
        top_score = ranked[0][1]
        cutoff_score = ranked[min(top_n - 1, len(ranked) - 1)][1] if len(ranked) >= top_n else 0
        print(f"Score range: {top_score} (best) ... {cutoff_score} (cutoff)")

    return new_d


def build_mined_dictionary(
    corpus_dir: Path,
    top_n: int = 999,
    include_seeds: bool = True,
    min_repos: int = MIN_REPOS,
    max_files: int | None = None,
    exclude_repos: list[str] | None = None,
    use_ngrams: bool = True,
    use_templates: bool = True,
    max_templates: int = 500,
    use_ast_templates: bool = True,
    max_ast_templates: int = 1000,
    score_sample_size: int = 2000,
) -> tuple[CompressionDictionary, list[tuple[str, int, int, float, int]]]:
    """
    Build a compression dictionary by mining broadly, then scoring and trimming.

    Pipeline:
    1. Mine all candidates (seeds + regex + n-gram + templates + AST templates)
       with generous internal limits
    2. Score every entry by actual Qwen corpus impact on a file sample
    3. Keep only the top_n entries by total tokens saved

    Returns:
        (dictionary, mined_patterns) -- the final trimmed dictionary and the
        raw mined results for downstream display/analysis.
    """
    # --- Phase 1: Mine broadly ---
    INTERNAL_LIMIT = 10000  # mine many candidates, trim later

    if include_seeds:
        d = CompressionDictionary.from_seed()
    else:
        d = CompressionDictionary()

    regex_mined = mine_patterns(
        corpus_dir, top_n=INTERNAL_LIMIT,
        min_repos=min_repos, max_files=max_files,
        exclude_repos=exclude_repos,
    )

    mined = []
    if use_ngrams:
        from sematok.ngram_mining import mine_ngram_patterns

        ngram_mined = mine_ngram_patterns(
            corpus_dir, top_n=INTERNAL_LIMIT,
            min_repos=min_repos, max_files=max_files,
            exclude_repos=exclude_repos,
        )
        mined = merge_mining_results(regex_mined, ngram_mined)
    else:
        mined = regex_mined

    added = 0
    for pattern, freq, tok_count, score, repo_count in mined:
        if pattern in d.pattern_to_macro:
            continue
        d.add_pattern(pattern, category="mined")
        added += 1
        if added >= INTERNAL_LIMIT:
            break
    print(f"Added {added} mined patterns (total: {d.size})")

    if use_templates:
        from sematok.template_mining import mine_templates

        template_results = mine_templates(
            corpus_dir,
            top_n=max_templates,
            min_repos=min_repos,
            max_files=max_files,
            exclude_repos=exclude_repos,
        )
        added_templates = 0
        for template_str, freq, slot_count, score, repo_count in template_results:
            if template_str in d.template_to_macro:
                continue
            d.add_template(template_str, slot_count)
            added_templates += 1
            if added_templates >= max_templates:
                break
        print(f"Added {added_templates} templates (total: {d.template_count})")

    if use_ast_templates:
        from sematok.ast_mining import mine_ast_templates

        ast_template_results = mine_ast_templates(
            corpus_dir,
            top_n=max_ast_templates,
            min_repos=min_repos,
            max_files=max_files,
            exclude_repos=exclude_repos,
        )
        added_ast = 0
        for template_str, freq, slot_count, score, repo_count in ast_template_results:
            if template_str in d.template_to_macro:
                continue
            d.add_template(template_str, slot_count, category="ast_template")
            added_ast += 1
            if added_ast >= max_ast_templates:
                break
        print(f"Added {added_ast} AST-mined templates (total: {d.template_count})")

    total_before = d.size + d.template_count
    print(f"\nTotal candidates before scoring: {total_before} ({d.size} exact + {d.template_count} templates)")

    # --- Phase 2: Score on corpus and trim to top_n ---
    if total_before > top_n:
        scores = _score_on_corpus(d, corpus_dir, sample_size=score_sample_size, exclude_repos=exclude_repos)
        d = _rebuild_top_n(d, scores, top_n)
    else:
        print(f"Only {total_before} entries — no trimming needed (target: {top_n})")

    return d, mined


def main():
    parser = argparse.ArgumentParser(description="Mine C# boilerplate patterns")
    parser.add_argument("--corpus", type=str, required=True, help="Directory with .cs files")
    parser.add_argument("--output", type=str, default="sematok/dictionary.json")
    parser.add_argument("--top", type=int, default=999, help="Final dictionary size after corpus impact scoring")
    parser.add_argument("--min-repos", type=int, default=MIN_REPOS, help="Min repos a pattern must appear in")
    parser.add_argument("--min-freq", type=int, default=MIN_FREQUENCY, help="Min frequency across corpus")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--no-seeds", action="store_true", help="Don't include seed patterns")
    parser.add_argument(
        "--exclude-repos", type=str, nargs="+", default=None,
        help="Repos to exclude from mining (e.g. held-out eval repos)",
    )
    parser.add_argument("--no-ngrams", action="store_true", help="Skip n-gram mining (regex only)")
    parser.add_argument("--no-templates", action="store_true", help="Skip template mining")
    parser.add_argument("--max-templates", type=int, default=500, help="Max template patterns")
    parser.add_argument("--no-ast-templates", action="store_true", help="Skip AST subtree mining")
    parser.add_argument("--max-ast-templates", type=int, default=1000, help="Max AST-mined templates")
    parser.add_argument("--score-sample", type=int, default=2000, help="Number of files to sample for corpus impact scoring")
    parser.add_argument("--verbose", action="store_true", help="Print all accepted patterns")
    args = parser.parse_args()

    d, mined = build_mined_dictionary(
        Path(args.corpus),
        top_n=args.top,
        include_seeds=not args.no_seeds,
        min_repos=args.min_repos,
        max_files=args.max_files,
        exclude_repos=args.exclude_repos,
        use_ngrams=not args.no_ngrams,
        use_templates=not args.no_templates,
        max_templates=args.max_templates,
        use_ast_templates=not args.no_ast_templates,
        max_ast_templates=args.max_ast_templates,
        score_sample_size=args.score_sample,
    )

    d.save(args.output)
    print(f"\nDictionary saved to {args.output}")
    print(f"Stats: {d.stats()}")

    # Print top patterns (reuse results from build, no re-scan)
    print(f"\nTop 30 mined patterns by score:")
    for pattern, freq, tok_count, score, repo_count in mined[:30]:
        in_dict = "+" if pattern in d.pattern_to_macro else " "
        print(
            f"  [{in_dict}] freq={freq:5d} toks={tok_count:2d} "
            f"score={score:7d} repos={repo_count} | {pattern!r}"
        )


if __name__ == "__main__":
    main()
