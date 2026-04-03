"""
Template mining: discover parameterized boilerplate patterns via AST-guided
identifier normalization.

Takes regex-matched candidates, parses them with tree-sitter to find
identifier leaf nodes, replaces normalizable identifiers with positional
placeholders {0}, {1}, ..., and counts template frequency across the corpus.

This collapses the long tail of identifier variants into high-frequency
templates.  E.g. ``this._logger = logger;`` and ``this._options = options;``
both become ``this.{0} = {1};``.

Usage:
    python -m sematok.template_mining --corpus data/raw_cs \
        --exclude-repos ppy--osu JamesNK--Newtonsoft.Json nunit--nunit
"""

import argparse
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm
from tree_sitter import Node

from sematok.lexer import get_safe_ranges, parse_source, set_language
from sematok.languages import LanguageConfig, get_language
from sematok.mining import (
    DEFAULT_TOKENIZER,
    MIN_CHAR_LENGTH,
    MIN_REPOS,
    _get_bpe_token_count,
    _load_file_repo_map,
)

# --- Constants ---

MIN_TEMPLATE_FREQUENCY = 30
MAX_SLOTS = 6
MIN_SLOTS = 1


def find_identifiers_in_range(
    root_node: Node,
    source_bytes: bytes,
    start: int,
    end: int,
) -> list[tuple[str, int, int, str]]:
    """Find `identifier` leaf nodes within [start, end) byte range.

    Returns [(text, start_byte, end_byte, parent_type), ...] sorted by position.
    """
    results = []
    stack = [root_node]

    while stack:
        node = stack.pop()
        # Skip nodes entirely outside the range
        if node.end_byte <= start or node.start_byte >= end:
            continue
        if node.type == "identifier" and node.start_byte >= start and node.end_byte <= end:
            text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            parent_type = node.parent.type if node.parent else ""
            results.append((text, node.start_byte, node.end_byte, parent_type))
        stack.extend(reversed(node.children))

    results.sort(key=lambda x: x[1])
    return results


def _is_last_identifier_child(node: Node) -> bool:
    """True if this node is the last `identifier` child of its parent."""
    parent = node.parent
    if parent is None:
        return True
    last_ident = None
    for child in parent.children:
        if child.type == "identifier":
            last_ident = child
    return last_ident is not None and last_ident.id == node.id


def should_normalize(
    text: str,
    parent_type: str,
    node: Node | None = None,
    lang: LanguageConfig | None = None,
) -> bool:
    """True if this identifier should become a placeholder."""
    if lang is None:
        from sematok.lexer import _get_lang
        lang = _get_lang()
    if text in lang.structural_names:
        return False
    if parent_type in lang.fixed_parent_types:
        return False
    if parent_type in lang.normalize_parent_types:
        return True
    # Ambiguous: parameter can hold both type and name
    if parent_type == "parameter":
        if node is not None:
            return _is_last_identifier_child(node)
        return False
    # Default: don't normalize unknown parent types
    return False


def normalize_candidate(
    candidate_text: str,
    candidate_start: int,
    root_node: Node,
    source_bytes: bytes,
    lang: LanguageConfig | None = None,
) -> tuple[str, list[str]] | None:
    """Replace normalizable identifiers with {0}, {1}, ...

    Assigns same placeholder to repeated identical identifiers.
    Returns (template, [unique_args]) or None if no identifiers normalized.
    """
    candidate_end = candidate_start + len(candidate_text.encode("utf-8"))
    idents = find_identifiers_in_range(root_node, source_bytes, candidate_start, candidate_end)

    # Find the actual tree-sitter nodes for should_normalize with node arg
    # We need to re-walk to get the nodes (find_identifiers_in_range returns tuples)
    nodes_by_pos: dict[int, Node] = {}

    stack = [root_node]
    while stack:
        node = stack.pop()
        if node.end_byte <= candidate_start or node.start_byte >= candidate_end:
            continue
        if node.type == "identifier" and node.start_byte >= candidate_start:
            nodes_by_pos[node.start_byte] = node
        stack.extend(reversed(node.children))

    # Determine which identifiers to normalize
    normalizable: list[tuple[str, int, int]] = []  # (text, rel_start, rel_end)
    for text, abs_start, abs_end, parent_type in idents:
        node = nodes_by_pos.get(abs_start)
        if should_normalize(text, parent_type, node, lang=lang):
            rel_start = abs_start - candidate_start
            rel_end = abs_end - candidate_start
            normalizable.append((text, rel_start, rel_end))

    if not normalizable:
        return None

    # Assign slot indices: same text -> same slot
    text_to_slot: dict[str, int] = {}
    unique_args: list[str] = []
    for text, _, _ in normalizable:
        if text not in text_to_slot:
            text_to_slot[text] = len(unique_args)
            unique_args.append(text)

    if len(unique_args) > MAX_SLOTS:
        return None

    # Build template by replacing identifiers right-to-left (preserves positions)
    template = candidate_text
    for text, rel_start, rel_end in reversed(normalizable):
        slot_idx = text_to_slot[text]
        template = template[:rel_start] + f"{{{slot_idx}}}" + template[rel_end:]

    # Reject if template equals the candidate (nothing normalized)
    if template == candidate_text:
        return None

    # Reject adjacent placeholders with no literal separator
    if "}{" in template:
        return None

    # Reject pure-placeholder templates (need at least one literal keyword/type)
    without_placeholders = re.sub(r"\{\d+\}", "", template)
    without_punctuation = re.sub(r"[^a-zA-Z]", "", without_placeholders)
    if len(without_punctuation) < 2:
        return None

    return template, unique_args


def extract_template_candidates(
    source: str,
    lang: LanguageConfig | None = None,
) -> list[tuple[str, list[str]]]:
    """Extract (template, args) pairs from safe zones of a source file.

    Parses once, applies candidate pattern regexes, normalizes identifiers.
    """
    if lang is None:
        from sematok.lexer import _get_lang
        lang = _get_lang()

    try:
        root_node, source_bytes = parse_source(source)
    except Exception:
        return []

    try:
        safe_ranges = get_safe_ranges(source)
    except Exception:
        safe_ranges = [(0, len(source.encode("utf-8")))]

    results: list[tuple[str, list[str]]] = []

    for range_start, range_end in safe_ranges:
        chunk_bytes = source_bytes[range_start:range_end]
        chunk_text = chunk_bytes.decode("utf-8", errors="replace")

        for pattern_re in lang.candidate_patterns:
            for match in pattern_re.finditer(chunk_text):
                candidate_text = match.group()
                if len(candidate_text) < MIN_CHAR_LENGTH:
                    continue
                # File byte position = safe range start + match byte offset
                file_byte_start = range_start + match.start()

                result = normalize_candidate(
                    candidate_text, file_byte_start, root_node, source_bytes,
                    lang=lang,
                )
                if result is not None:
                    results.append(result)

    return results


def mine_templates(
    corpus_dir: Path,
    language: str | LanguageConfig,
    top_n: int = 2000,
    min_frequency: int = MIN_TEMPLATE_FREQUENCY,
    min_repos: int = MIN_REPOS,
    min_slots: int = MIN_SLOTS,
    max_slots: int = MAX_SLOTS,
    max_files: int | None = None,
    exclude_repos: list[str] | None = None,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> list[tuple[str, int, int, float, int]]:
    """Mine template patterns from corpus.

    Returns [(template, frequency, slot_count, score, repo_count), ...]
    sorted by score descending.
    """
    lang = get_language(language) if isinstance(language, str) else language
    set_language(lang)
    file_to_repo = _load_file_repo_map(corpus_dir)
    exclude_set = set(exclude_repos) if exclude_repos else set()

    files = sorted(corpus_dir.glob(f"*{lang.file_extension}"))
    if exclude_set and file_to_repo:
        files = [
            f for f in files
            if file_to_repo.get(f.name, "unknown") not in exclude_set
        ]
    if max_files:
        files = files[:max_files]

    template_counter: Counter = Counter()
    template_repos: dict[str, set[str]] = defaultdict(set)
    ident_freq: Counter = Counter()  # frequency-weighted identifier census

    for f in tqdm(files, desc="Template mining"):
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        repo = file_to_repo.get(f.name, "unknown")
        candidates = extract_template_candidates(source, lang=lang)

        # Deduplicate per file
        file_templates: set[str] = set()
        for template, args in candidates:
            file_templates.add(template)
            for arg in args:
                ident_freq[arg] += 1

        template_counter.update(file_templates)
        for t in file_templates:
            template_repos[t].add(repo)

    # Filter and score
    enc = AutoTokenizer.from_pretrained(tokenizer_name)

    # Compute frequency-weighted average BPE cost per identifier
    if ident_freq:
        bpe_cache: dict[str, int] = {}
        for ident in ident_freq:
            bpe_cache[ident] = _get_bpe_token_count(ident, enc)
        total_weighted = sum(bpe_cache[ident] * count for ident, count in ident_freq.items())
        total_count = sum(ident_freq.values())
        avg_ident_bpe = total_weighted / total_count
        # Pick the real identifier closest to the average as the scoring proxy
        representative_ident = min(ident_freq, key=lambda x: abs(bpe_cache[x] - avg_ident_bpe))
        print(f"Identifier census: {len(ident_freq):,} unique, {total_count:,} total, "
              f"avg BPE cost={avg_ident_bpe:.2f}, proxy={representative_ident!r} "
              f"({bpe_cache[representative_ident]} tokens)")
    else:
        representative_ident = "_varName"

    scored = []
    rejected = {"low_freq": 0, "few_repos": 0, "few_slots": 0, "many_slots": 0}

    for template, freq in template_counter.items():
        if freq < min_frequency:
            rejected["low_freq"] += 1
            continue
        repo_count = len(template_repos.get(template, set()))
        if repo_count < min_repos:
            rejected["few_repos"] += 1
            continue
        # Count slots
        slot_indices = set(int(m.group(1)) for m in re.finditer(r"\{(\d+)\}", template))
        slot_count = len(slot_indices)
        if slot_count < min_slots:
            rejected["few_slots"] += 1
            continue
        if slot_count > max_slots:
            rejected["many_slots"] += 1
            continue

        # Score using corpus-derived representative identifier
        estimated = template
        for i in range(slot_count):
            estimated = estimated.replace(f"{{{i}}}", representative_ident)
        token_count = _get_bpe_token_count(estimated, enc)
        # Template cost: prefix (1 token) + BPE(args) + closer (1 token)
        args_str = ",".join([representative_ident] * slot_count)
        macro_cost = 2 + _get_bpe_token_count(args_str, enc)
        savings_per_match = token_count - macro_cost
        score = freq * max(savings_per_match, 0)
        scored.append((template, freq, slot_count, score, repo_count))

    scored.sort(key=lambda x: x[3], reverse=True)

    print(f"\nTemplate candidates: {len(template_counter):,}")
    print(f"After filtering: {len(scored)} passed, rejected: {rejected}")

    return scored[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Mine template patterns")
    parser.add_argument("--corpus", type=str, required=True, help="Directory with source files")
    parser.add_argument("--language", type=str, required=True, help="Language config to use (e.g. csharp, python)")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER, help="HuggingFace tokenizer for scoring")
    parser.add_argument("--top", type=int, default=100, help="Show top N templates")
    parser.add_argument("--min-freq", type=int, default=MIN_TEMPLATE_FREQUENCY)
    parser.add_argument("--min-repos", type=int, default=MIN_REPOS)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--exclude-repos", type=str, nargs="+", default=None,
        help="Repos to exclude from mining",
    )
    args = parser.parse_args()

    results = mine_templates(
        Path(args.corpus),
        top_n=args.top,
        min_frequency=args.min_freq,
        min_repos=args.min_repos,
        max_files=args.max_files,
        exclude_repos=args.exclude_repos,
        language=args.language,
        tokenizer_name=args.tokenizer,
    )

    print(f"\nTop {min(args.top, len(results))} templates by score:")
    for template, freq, slot_count, score, repo_count in results[:args.top]:
        print(
            f"  freq={freq:5d} slots={slot_count} "
            f"score={score:7.0f} repos={repo_count:2d} | {template!r}"
        )


if __name__ == "__main__":
    main()
