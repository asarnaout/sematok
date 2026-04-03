"""
Pattern mining: discover frequent multi-token boilerplate patterns from a corpus.

Scans source files, extracts candidate patterns, scores them by
frequency * token_savings, and produces a refined compression dictionary.

Quality filters:
- Must appear in at least 2 repos (no repo-specific patterns)
- Must be at least 8 characters (no short junk like [0], [i])
- Must span at least 3 BPE tokens (not worth compressing otherwise)
- Must appear at least 50 times across the corpus
- No embedded newlines (regex artifacts)
- No pure-logic patterns (only boilerplate/scaffolding)

Usage:
    python -m sematok.mining --corpus data/raw_cs --language csharp
"""

import argparse
import json

import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm

from sematok.dictionary import CompressionDictionary
from sematok.languages import LanguageConfig, get_language
from sematok.lexer import get_safe_ranges, set_language

DEFAULT_TOKENIZER = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


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


def extract_candidates_from_file(
    source: str,
    candidate_patterns: list[re.Pattern] | None = None,
) -> list[str]:
    """Extract candidate boilerplate patterns from a source file."""
    if candidate_patterns is None:
        raise ValueError("candidate_patterns must be provided")

    try:
        safe_ranges = get_safe_ranges(source)
    except Exception:
        safe_ranges = [(0, len(source.encode("utf-8")))]

    source_bytes = source.encode("utf-8")
    safe_text_parts = []
    for start, end in safe_ranges:
        safe_text_parts.append(source_bytes[start:end].decode("utf-8", errors="replace"))
    safe_text = "\n".join(safe_text_parts)

    candidates = []
    for pattern_re in candidate_patterns:
        for match in pattern_re.finditer(safe_text):
            candidate = match.group(0).strip()
            if candidate and _is_valid_pattern(candidate):
                candidates.append(candidate)

    return candidates


def mine_patterns(
    corpus_dir: Path,
    language: str | LanguageConfig,
    top_n: int = 1000,
    min_frequency: int = MIN_FREQUENCY,
    min_token_span: int = MIN_TOKEN_SPAN,
    min_repos: int = MIN_REPOS,
    max_files: int | None = None,
    exclude_repos: list[str] | None = None,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> list[tuple[str, int, int, float, int]]:
    """
    Mine frequent boilerplate patterns from a corpus of source files.

    Args:
        exclude_repos: Repo names to skip (e.g. held-out eval repos).
            This ensures the dictionary is built only from training data.
        language: Language name or LanguageConfig instance.
        tokenizer_name: HuggingFace tokenizer for scoring token savings.

    Returns:
        List of (pattern, frequency, token_count, score, repo_count) tuples
        sorted by score descending.
        Score = frequency * (token_count - 1), representing total tokens saved.
    """
    lang = get_language(language) if isinstance(language, str) else language
    set_language(lang)
    enc = AutoTokenizer.from_pretrained(tokenizer_name)

    # Load repo mapping for cross-repo validation
    file_to_repo = _load_file_repo_map(corpus_dir)
    exclude_set = set(exclude_repos) if exclude_repos else set()

    # Track global frequency and per-repo presence
    pattern_counter: Counter = Counter()
    pattern_repos: dict[str, set[str]] = defaultdict(set)

    files = sorted(corpus_dir.glob(f"*{lang.file_extension}"))

    # Exclude eval repo files from mining
    if exclude_set and file_to_repo:
        before = len(files)
        files = [f for f in files if file_to_repo.get(f.name, "unknown") not in exclude_set]
        print(f"Excluded {before - len(files)} files from {len(exclude_set)} eval repos")

    if max_files:
        files = files[:max_files]

    print(f"Mining patterns from {len(files)} files...")
    for f in tqdm(files, desc="Scanning"):
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        repo = file_to_repo.get(f.name, "unknown")
        candidates = extract_candidates_from_file(source, lang.candidate_patterns)

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



def _score_on_corpus(
    d: CompressionDictionary,
    corpus_dir: Path,
    language: str | LanguageConfig,
    exclude_repos: list[str] | None = None,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
    """
    Score every dictionary entry by repo-weighted corpus impact.

    Each repo contributes equally to a macro's score regardless of repo size.
    For each macro, score = sum over repos of (savings_in_repo / files_in_repo).
    This prevents large repos from dominating the dictionary ranking.

    Returns:
        (scores, file_counts, repo_counts) where:
        - scores: {macro: repo_weighted_score}
        - file_counts: {macro: number_of_files_it_appeared_in}
        - repo_counts: {macro: number_of_distinct_repos_it_appeared_in}
    """
    from sematok.compressor import Compressor

    lang = get_language(language) if isinstance(language, str) else language
    set_language(lang)
    enc = AutoTokenizer.from_pretrained(tokenizer_name)
    compressor = Compressor(d, language=lang)

    file_to_repo = _load_file_repo_map(corpus_dir)
    exclude_set = set(exclude_repos) if exclude_repos else set()

    files = sorted(corpus_dir.glob(f"*{lang.file_extension}"))
    if exclude_set and file_to_repo:
        files = [f for f in files if file_to_repo.get(f.name, "unknown") not in exclude_set]

    macro_re = re.compile(r"<\|M\d+\|>")
    template_re = re.compile(r"<\|T(\d+):([^|]*)\|>")

    repo_total_files: Counter = Counter()
    repo_macro_savings: dict[str, Counter] = defaultdict(Counter)
    file_counts: Counter = Counter()
    repo_sets: dict[str, set[str]] = defaultdict(set)

    print(f"\nScoring {len(files)} files for corpus impact...")
    for f in tqdm(files, desc="Scoring"):
        repo = file_to_repo.get(f.name, "unknown")
        repo_total_files[repo] += 1
        source = f.read_text(encoding="utf-8", errors="replace")
        try:
            safe_ranges = get_safe_ranges(source)
        except Exception:
            safe_ranges = [(0, len(source.encode("utf-8")))]

        compressed = compressor.compress(source, safe_ranges=safe_ranges)

        # Track which macros appeared in this file (deduplicate per file)
        seen_in_file: set[str] = set()

        for macro in macro_re.findall(compressed):
            pattern = d.macro_to_pattern.get(macro, "")
            if pattern:
                saving = len(enc.encode(pattern, add_special_tokens=False)) - 1
                if saving > 0:
                    repo_macro_savings[macro][repo] += saving
                    seen_in_file.add(macro)

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
                    repo_macro_savings[macro_base][repo] += saving
                    seen_in_file.add(macro_base)

        for macro in seen_in_file:
            file_counts[macro] += 1
            repo_sets[macro].add(repo)

    # Compute repo-weighted scores: each repo contributes equally
    scores: dict[str, float] = {}
    for macro, per_repo in repo_macro_savings.items():
        score = 0.0
        for repo, total_savings in per_repo.items():
            score += total_savings / repo_total_files[repo]
        scores[macro] = score

    repo_counts = {macro: len(repos) for macro, repos in repo_sets.items()}
    return scores, dict(file_counts), repo_counts


def _rebuild_top_n(
    d: CompressionDictionary,
    scores: dict[str, float],
    file_counts: dict[str, int],
    min_file_count: int = 0,
    repo_counts: dict[str, int] | None = None,
    min_repo_count: int = 0,
    max_entries: int = 0,
) -> CompressionDictionary:
    """
    Rebuild a dictionary keeping only entries that pass quality filters.

    Entries that never fired (score=0) are dropped. If min_file_count > 0,
    entries appearing in fewer than that many files are also dropped.
    If min_repo_count > 0, entries appearing in fewer than that many distinct
    repos are also dropped (prevents repo-specific patterns from surviving).
    If max_entries > 0, at most that many entries are kept (by score rank).
    Remaining entries are ranked by repo-weighted score and assigned
    fresh sequential macro IDs. The 5-digit macro ID format caps exact macros
    and templates at 99,999 each.
    """
    ranked = sorted(scores.items(), key=lambda x: -x[1])

    new_d = CompressionDictionary()
    kept_patterns = 0
    kept_templates = 0
    skipped_low_freq = 0
    skipped_low_repos = 0
    skipped_max_entries = 0
    skipped_format_limit = 0

    from sematok.dictionary import MAX_MACROS, MAX_TEMPLATES
    for macro, score in ranked:

        fc = file_counts.get(macro, 0)
        if min_file_count > 0 and fc < min_file_count:
            skipped_low_freq += 1
            continue

        if min_repo_count > 0 and repo_counts:
            rc = repo_counts.get(macro, 0)
            if rc < min_repo_count:
                skipped_low_repos += 1
                continue

        if max_entries > 0 and (kept_patterns + kept_templates) >= max_entries:
            skipped_max_entries += 1
            continue

        if macro in d.macro_to_pattern:
            if kept_patterns >= MAX_MACROS:
                skipped_format_limit += 1
                continue
            pattern = d.macro_to_pattern[macro]
            category = d.pattern_categories.get(pattern, "mined")
            new_d.add_pattern(pattern, category=category)
            kept_patterns += 1
        elif macro in d.macro_to_template:
            if kept_templates >= MAX_TEMPLATES:
                skipped_format_limit += 1
                continue
            template = d.macro_to_template[macro]
            slots = d.template_slots[template]
            category = d.template_categories.get(template, "template")
            new_d.add_template(template, slots, category=category)
            kept_templates += 1

    total = kept_patterns + kept_templates
    print(f"\nKept {total} entries: {kept_patterns} exact + {kept_templates} templates")
    if min_file_count > 0:
        print(f"Skipped {skipped_low_freq} entries below min file count ({min_file_count})")
    if min_repo_count > 0:
        print(f"Skipped {skipped_low_repos} entries below min repo count ({min_repo_count})")
    if skipped_max_entries > 0:
        print(f"WARNING: {skipped_max_entries} eligible entries dropped by --max-entries {max_entries}. "
              f"Increase --max-entries or raise --min-files to keep only the most impactful patterns.")
    if skipped_format_limit > 0:
        print(f"WARNING: {skipped_format_limit} eligible entries dropped by macro ID limit "
              f"(max {MAX_MACROS} exact + {MAX_TEMPLATES} templates). Raise --min-files to reduce count.")
    if ranked:
        top_score = ranked[0][1]
        print(f"Score range: {top_score:.1f} (best) ... {ranked[-1][1]:.1f} (worst)")

    # --- Frequency distribution report ---
    print(f"\nFrequency distribution (all {len(ranked)} scored entries):")
    print(f"{'Rank':<6} {'Score':>8} {'Files':>8} {'Repos':>6} {'Avg/Repo':>9} {'Type':<5} Pattern")
    print("-" * 110)
    for i, (macro, score) in enumerate(ranked):
        fc = file_counts.get(macro, 0)
        rc = repo_counts.get(macro, 0) if repo_counts else 0
        avg = score / rc if rc > 0 else 0
        if macro in d.macro_to_pattern:
            ptype = "exact"
            label = repr(d.macro_to_pattern[macro])
        elif macro in d.macro_to_template:
            ptype = "tmpl"
            label = repr(d.macro_to_template[macro])
        else:
            ptype = "?"
            label = macro
        # Truncate long patterns for display
        if len(label) > 50:
            label = label[:47] + "..."
        print(f"{i+1:<6} {score:>8.1f} {fc:>8} {rc:>6} {avg:>9.1f} {ptype:<5} {label}")

    # Summary buckets
    all_fc = [file_counts.get(m, 0) for m, _ in ranked]
    buckets = [
        ("0 files (never seen)", lambda x: x == 0),
        ("1-9 files", lambda x: 1 <= x <= 9),
        ("10-49 files", lambda x: 10 <= x <= 49),
        ("50-99 files", lambda x: 50 <= x <= 99),
        ("100-499 files", lambda x: 100 <= x <= 499),
        ("500-999 files", lambda x: 500 <= x <= 999),
        ("1000-4999 files", lambda x: 1000 <= x <= 4999),
        ("5000+ files", lambda x: x >= 5000),
    ]
    print(f"\nFile count distribution:")
    for label, pred in buckets:
        count = sum(1 for fc in all_fc if pred(fc))
        if count > 0:
            print(f"  {label:<25} {count:>5} entries")

    # --- Threshold analysis: help user choose --min-files ---
    total_impact = sum(scores.values())
    all_entries = [(m, scores.get(m, 0), file_counts.get(m, 0)) for m, _ in ranked if scores.get(m, 0) > 0]
    thresholds = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000]
    print(f"\n--min-files threshold analysis (use this to choose --min-files):")
    print(f"  {'Threshold':<12} {'Entries':<10} {'Total impact':>14} {'% of impact':>12}")
    print(f"  {'-'*12} {'-'*10} {'-'*14} {'-'*12}")
    for t in thresholds:
        surviving = [(m, s, fc) for m, s, fc in all_entries if fc >= t]
        impact = sum(s for _, s, _ in surviving)
        pct = impact / total_impact * 100 if total_impact > 0 else 0
        count = len(surviving)
        marker = "  <-- current" if t == min_file_count else ""
        print(f"  >= {t:<8} {count:<10} {impact:>14.1f} {pct:>11.1f}%{marker}")

    return new_d


def _auto_select_min_files(
    scores: dict[str, float],
    file_counts: dict[str, int],
) -> int:
    """Find the highest --min-files threshold retaining >=90% of total impact."""
    total_impact = sum(scores.values())
    if total_impact == 0:
        return 0
    thresholds = [0, 10, 50, 100, 200, 500, 1000, 2000, 5000]
    best = 0
    for t in thresholds:
        impact = sum(s for m, s in scores.items() if file_counts.get(m, 0) >= t)
        if impact / total_impact >= 0.90:
            best = t
    return best


def _save_scores(
    d: CompressionDictionary,
    scores: dict[str, float],
    file_counts: dict[str, int],
    repo_counts: dict[str, int],
    output_path: str,
) -> None:
    """Save scoring data to a sidecar JSON file for later re-filtering."""
    entries = {}
    for macro in scores:
        if macro in d.macro_to_pattern:
            entry_type = "pattern"
            content = d.macro_to_pattern[macro]
            category = d.pattern_categories.get(content, "mined")
            slots = None
        elif macro in d.macro_to_template:
            entry_type = "template"
            content = d.macro_to_template[macro]
            category = d.template_categories.get(content, "template")
            slots = d.template_slots[content]
        else:
            continue
        entries[macro] = {
            "type": entry_type,
            "content": content,
            "category": category,
            "score": scores[macro],
            "file_count": file_counts.get(macro, 0),
            "repo_count": repo_counts.get(macro, 0),
        }
        if slots is not None:
            entries[macro]["slots"] = slots

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(entries, f, indent=2, ensure_ascii=False)
    print(f"Scoring data saved to {output_path} ({len(entries)} entries)")


def _load_scores(
    scores_path: str,
) -> tuple[CompressionDictionary, dict[str, float], dict[str, int], dict[str, int]]:
    """Load scoring data from a sidecar file. Returns (dictionary, scores, file_counts, repo_counts)."""
    with open(scores_path, encoding="utf-8") as f:
        entries = json.load(f)

    d = CompressionDictionary()
    scores = {}
    file_counts = {}
    repo_counts = {}

    for macro, entry in entries.items():
        if entry["type"] == "pattern":
            d.add_pattern(entry["content"], category=entry.get("category", "mined"))
        elif entry["type"] == "template":
            d.add_template(entry["content"], entry["slots"], category=entry.get("category", "template"))
        scores[macro] = entry["score"]
        file_counts[macro] = entry["file_count"]
        repo_counts[macro] = entry["repo_count"]

    return d, scores, file_counts, repo_counts


def refilter(
    scores_path: str,
    output_path: str,
    min_file_count: int = 0,
    max_entries: int = 999,
) -> CompressionDictionary:
    """Re-filter a scored dictionary with new --min-files / --max-entries thresholds.

    Loads scoring data from a sidecar file and applies filters without
    re-mining or re-scoring. Note: --min-repos cannot be changed here
    because it affects the mining phase.
    """
    d, scores, file_counts, repo_counts = _load_scores(scores_path)
    print(f"Loaded {len(scores)} scored entries from {scores_path}")

    new_d = _rebuild_top_n(
        d, scores, file_counts,
        min_file_count=min_file_count,
        repo_counts=repo_counts,
        min_repo_count=0,
        max_entries=max_entries,
    )

    new_d.save(output_path)
    print(f"\nDictionary saved to {output_path}")
    print(f"Stats: {new_d.stats()}")
    return new_d


def build_mined_dictionary(
    corpus_dir: Path,
    language: str | LanguageConfig,
    include_seeds: bool = True,
    min_repos: int = MIN_REPOS,
    max_files: int | None = None,
    exclude_repos: list[str] | None = None,
    use_ngrams: bool = True,
    use_templates: bool = True,
    max_templates: int = 500,
    use_ast_templates: bool = True,
    max_ast_templates: int = 1000,
    min_file_count: int = 0,
    max_entries: int = 999,
    tokenizer_name: str = DEFAULT_TOKENIZER,
    scores_output_path: str | None = None,
    auto_min_files: bool = False,
) -> tuple[CompressionDictionary, list[tuple[str, int, int, float, int]]]:
    """
    Build a compression dictionary by mining broadly, then scoring and trimming.

    Pipeline:
    1. Mine all candidates (seeds + regex + n-gram + templates + AST templates)
       with generous internal limits
    2. Score every entry by actual corpus impact on a file sample
    3. Keep entries that pass --min-files, --min-repos, and --max-entries filters

    If auto_min_files=True, --min-files is auto-selected from the scoring
    data (highest threshold retaining >=90% of total impact).

    Dictionary size is determined by the quality filters. An optional
    --max-entries cap limits the total count. The 5-digit macro ID format
    (M00001-M99999, T00001-T99999) enforces a hard ceiling of 99,999 exact
    macros and 99,999 templates regardless.

    Args:
        language: Language name or LanguageConfig instance.
        tokenizer_name: HuggingFace tokenizer for scoring token savings.

    Returns:
        (dictionary, mined_patterns) -- the final trimmed dictionary and the
        raw mined results for downstream display/analysis.
    """
    lang = get_language(language) if isinstance(language, str) else language

    # --- Phase 1: Mine broadly ---
    INTERNAL_LIMIT = 10000  # mine many candidates, trim later

    if include_seeds:
        d = CompressionDictionary.from_seed(language=lang.name)
    else:
        d = CompressionDictionary()

    regex_mined = mine_patterns(
        corpus_dir, top_n=INTERNAL_LIMIT,
        min_repos=min_repos, max_files=max_files,
        exclude_repos=exclude_repos,
        language=lang, tokenizer_name=tokenizer_name,
    )

    mined = []
    if use_ngrams:
        from sematok.ngram_mining import mine_ngram_patterns

        ngram_mined = mine_ngram_patterns(
            corpus_dir, top_n=INTERNAL_LIMIT,
            min_repos=min_repos, max_files=max_files,
            exclude_repos=exclude_repos,
            language=lang, tokenizer_name=tokenizer_name,
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
            language=lang, tokenizer_name=tokenizer_name,
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
            language=lang, tokenizer_name=tokenizer_name,
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

    # --- Phase 2: Score on corpus and trim ---
    needs_trim = min_file_count > 0 or min_repos > 0 or max_entries > 0 or auto_min_files
    if needs_trim:
        scores, file_counts, repo_counts = _score_on_corpus(
            d, corpus_dir, exclude_repos=exclude_repos,
            language=lang, tokenizer_name=tokenizer_name,
        )

        # Save scoring data so we can re-filter without re-mining
        if scores_output_path:
            _save_scores(d, scores, file_counts, repo_counts, scores_output_path)

        # Auto-select --min-files if requested
        if auto_min_files:
            min_file_count = _auto_select_min_files(scores, file_counts)
            # Calculate stats for the auto-selected threshold
            surviving = sum(1 for m in scores if file_counts.get(m, 0) >= min_file_count)
            total_impact = sum(scores.values())
            retained_impact = sum(s for m, s in scores.items() if file_counts.get(m, 0) >= min_file_count)
            pct = retained_impact / total_impact * 100 if total_impact > 0 else 0
            print(f"\nAuto-selected --min-files={min_file_count} "
                  f"(retains {pct:.1f}% of impact, {surviving} entries)")

        d = _rebuild_top_n(d, scores, file_counts, min_file_count=min_file_count,
                           repo_counts=repo_counts, min_repo_count=min_repos,
                           max_entries=max_entries)
    else:
        print(f"{total_before} entries, no quality filters specified — skipping scoring phase")

    return d, mined


def main():
    parser = argparse.ArgumentParser(description="Mine boilerplate patterns")
    parser.add_argument("--corpus", type=str, required=True, help="Directory with source files")
    parser.add_argument("--output", type=str, default=None, help="Output dictionary path (default: sematok/languages/<lang>/dictionary.json)")
    parser.add_argument("--language", type=str, required=True, help="Language config to use (e.g. csharp, python)")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER, help="HuggingFace tokenizer for scoring")
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
    parser.add_argument("--min-files", type=int, default=0, help="Min training files a pattern must appear in (0 = no threshold)")
    parser.add_argument("--max-entries", type=int, default=0, help="Max entries in final dictionary (0 = no cap)")
    parser.add_argument("--verbose", action="store_true", help="Print all accepted patterns")
    parser.add_argument("--scores-output", type=str, default=None,
                        help="Save scoring data to this sidecar JSON (for later re-filtering)")
    parser.add_argument("--refilter", type=str, default=None, metavar="SCORES_JSON",
                        help="Re-filter from saved scoring data instead of mining "
                             "(only --min-files, --max-entries, and --output are used)")
    parser.add_argument("--auto", action="store_true",
                        help="Auto-select --min-files from corpus data. "
                             "Uses --min-repos 2 by default (override with --min-repos).")
    args = parser.parse_args()

    if args.refilter:
        output_path = args.output
        if output_path is None:
            from sematok.languages import get_dictionary_path
            lang_dict = get_dictionary_path(args.language)
            output_path = str(lang_dict) if lang_dict else f"sematok/languages/{args.language}/dictionary.json"
        refilter(args.refilter, output_path,
                 min_file_count=args.min_files,
                 max_entries=args.max_entries)
        return

    # In auto mode: default min-repos to 2, force scoring, auto-derive scores path
    if args.auto:
        import sys
        min_repos = args.min_repos if "--min-repos" in sys.argv else 2
        min_files = 1  # force scoring; auto-selection overrides this
        scores_output = args.scores_output
        if not scores_output:
            from sematok.languages import get_dictionary_path
            output_path = args.output
            if output_path is None:
                lang_dict = get_dictionary_path(args.language)
                output_path = str(lang_dict) if lang_dict else f"sematok/languages/{args.language}/dictionary.json"
            base = output_path.rsplit(".", 1)[0]
            scores_output = f"{base}_scores.json"
    else:
        min_repos = args.min_repos
        min_files = args.min_files
        scores_output = args.scores_output

    d, mined = build_mined_dictionary(
        Path(args.corpus),
        include_seeds=not args.no_seeds,
        min_repos=min_repos,
        max_files=args.max_files,
        exclude_repos=args.exclude_repos,
        use_ngrams=not args.no_ngrams,
        use_templates=not args.no_templates,
        max_templates=args.max_templates,
        use_ast_templates=not args.no_ast_templates,
        max_ast_templates=args.max_ast_templates,
        min_file_count=min_files,
        max_entries=args.max_entries,
        language=args.language,
        tokenizer_name=args.tokenizer,
        scores_output_path=scores_output,
        auto_min_files=args.auto,
    )

    from sematok.languages import get_dictionary_path
    output_path = args.output
    if output_path is None:
        lang_dict = get_dictionary_path(args.language)
        output_path = str(lang_dict) if lang_dict else f"sematok/languages/{args.language}/dictionary.json"
    d.save(output_path)
    print(f"\nDictionary saved to {output_path}")
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
