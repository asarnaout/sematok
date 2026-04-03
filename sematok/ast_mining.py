"""
AST subtree mining: discover parameterized boilerplate patterns by walking
tree-sitter CSTs and extracting frequent normalized subtree strings.

Replaces regex-driven candidate discovery with structural subtree extraction,
enabling discovery of patterns that no hand-crafted regex anticipated.

Usage:
    python -m sematok.ast_mining --corpus data/raw_cs \
        --exclude-repos ppy--osu JamesNK--Newtonsoft.Json nunit--nunit
"""

import argparse
import bisect
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer
from tqdm import tqdm

from sematok.languages import LanguageConfig, get_language
from sematok.lexer import get_safe_ranges, parse_source, set_language
from sematok.mining import DEFAULT_TOKENIZER, MIN_CHAR_LENGTH, MIN_REPOS, _get_bpe_token_count, _load_file_repo_map
from sematok.template_mining import MAX_SLOTS, should_normalize

# --- Constants ---

MIN_DEPTH = 2
MAX_DEPTH = 6
MAX_SOURCE_LENGTH = 200
MIN_AST_TEMPLATE_FREQUENCY = 30
MIN_TOKEN_SPAN = 3
PRUNE_INTERVAL = 5000


def _subtree_depth(node) -> int:
    """Compute max depth of a subtree (iterative, capped at MAX_DEPTH)."""
    max_depth = 0
    stack = [(node, 0)]
    while stack:
        n, d = stack.pop()
        if d > max_depth:
            max_depth = d
        if d < MAX_DEPTH:
            for child in n.children:
                stack.append((child, d + 1))
    return max_depth


def _is_in_safe_range(byte_offset: int, safe_starts: list[int], safe_ends: list[int]) -> bool:
    """Check if byte_offset falls within any safe range using binary search."""
    if not safe_starts:
        return False
    idx = bisect.bisect_right(safe_starts, byte_offset) - 1
    if idx < 0:
        return False
    return byte_offset < safe_ends[idx]


def normalize_subtree(
    node, source_bytes: bytes, lang: LanguageConfig | None = None,
) -> tuple[str, list[str]] | None:
    """Extract source text of a subtree, normalize identifiers, collapse whitespace.

    Returns (template, [unique_args]) or None if nothing was normalized.
    """
    if lang is None:
        from sematok.lexer import _get_lang
        lang = _get_lang()
    raw = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")

    # Collect normalizable identifier leaves within this subtree
    ident_replacements = []  # (rel_start, rel_end, text)
    stack = [node]
    while stack:
        n = stack.pop()
        if n.type == "identifier":
            text = source_bytes[n.start_byte:n.end_byte].decode("utf-8", errors="replace")
            parent_type = n.parent.type if n.parent else ""
            if should_normalize(text, parent_type, n, lang=lang):
                rel_start = n.start_byte - node.start_byte
                rel_end = n.end_byte - node.start_byte
                ident_replacements.append((rel_start, rel_end, text))
        stack.extend(reversed(n.children))

    if not ident_replacements:
        return None

    # Sort by position for slot assignment
    ident_replacements.sort(key=lambda x: x[0])

    # Assign slot indices: same text -> same slot
    text_to_slot: dict[str, int] = {}
    unique_args: list[str] = []
    slotted: list[tuple[int, int, int]] = []  # (rel_start, rel_end, slot_idx)

    for rel_start, rel_end, text in ident_replacements:
        if text not in text_to_slot:
            if len(unique_args) >= MAX_SLOTS:
                return None
            text_to_slot[text] = len(unique_args)
            unique_args.append(text)
        slotted.append((rel_start, rel_end, text_to_slot[text]))

    # Replace identifiers right-to-left (preserves byte positions)
    template = raw
    for rel_start, rel_end, slot_idx in reversed(slotted):
        template = template[:rel_start] + f"{{{slot_idx}}}" + template[rel_end:]

    # Collapse whitespace
    template = " ".join(template.split())
    raw_collapsed = " ".join(raw.split())

    if template == raw_collapsed:
        return None

    # Reject adjacent placeholders with no literal separator
    if re.search(r"\}\s*\{", template):
        stripped = template.replace(" ", "").replace("\t", "")
        if "}{" in stripped:
            return None

    # Reject pure-placeholder templates (need at least one literal keyword/type)
    without_placeholders = re.sub(r"\{\d+\}", "", template)
    without_punctuation = re.sub(r"[^a-zA-Z]", "", without_placeholders)
    if len(without_punctuation) < 2:
        return None

    return template, unique_args


def extract_ast_candidates(
    source: str, lang: LanguageConfig | None = None,
) -> list[tuple[str, list[str]]]:
    """Extract (template, args) pairs from AST subtrees of a source file.

    Parses once, walks the full AST, extracts and normalizes subtrees
    at target node types.
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

    if not safe_ranges:
        return []

    # Build sorted arrays for binary search
    safe_starts = [s for s, _ in safe_ranges]
    safe_ends = [e for _, e in safe_ranges]

    results: list[tuple[str, list[str]]] = []

    # Single iterative DFS over the full tree
    stack = [root_node]
    while stack:
        cur = stack.pop()

        if cur.type in lang.subtree_root_types:
            # Check safe zone
            if not _is_in_safe_range(cur.start_byte, safe_starts, safe_ends):
                # Still recurse into children — nested targets may be in safe zones
                stack.extend(reversed(cur.children))
                continue

            # Check source length
            src_len = cur.end_byte - cur.start_byte
            if src_len > MAX_SOURCE_LENGTH or src_len < MIN_CHAR_LENGTH:
                stack.extend(reversed(cur.children))
                continue

            # Check depth
            depth = _subtree_depth(cur)
            if depth < MIN_DEPTH or depth > MAX_DEPTH:
                stack.extend(reversed(cur.children))
                continue

            # Normalize
            result = normalize_subtree(cur, source_bytes, lang=lang)
            if result is not None:
                template, args = result
                # Final length check after whitespace collapse
                if len(template) >= MIN_CHAR_LENGTH:
                    results.append((template, args))

        # Always recurse into children to find nested target nodes
        stack.extend(reversed(cur.children))

    return results


def mine_ast_templates(
    corpus_dir: Path,
    language: str | LanguageConfig,
    top_n: int = 1000,
    min_frequency: int = MIN_AST_TEMPLATE_FREQUENCY,
    min_repos: int = MIN_REPOS,
    max_files: int | None = None,
    exclude_repos: list[str] | None = None,
    tokenizer_name: str = DEFAULT_TOKENIZER,
) -> list[tuple[str, int, int, float, int]]:
    """Mine template patterns from AST subtrees across the corpus.

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

    for file_idx, f in enumerate(tqdm(files, desc="AST subtree mining")):
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        repo = file_to_repo.get(f.name, "unknown")
        candidates = extract_ast_candidates(source, lang=lang)

        # Per-file dedup
        file_templates: set[str] = set()
        for template, args in candidates:
            file_templates.add(template)

        template_counter.update(file_templates)
        for t in file_templates:
            template_repos[t].add(repo)

        # Periodic pruning to control memory
        if file_idx > 0 and file_idx % PRUNE_INTERVAL == 0:
            cutoff = max(2, file_idx // 2000)
            template_counter = Counter(
                {k: v for k, v in template_counter.items() if v >= cutoff}
            )

    # Filter and score
    enc = AutoTokenizer.from_pretrained(tokenizer_name)
    scored = []
    rejected = {"low_freq": 0, "few_repos": 0, "few_slots": 0, "many_slots": 0, "few_tokens": 0}

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
        if slot_count < 1:
            rejected["few_slots"] += 1
            continue
        if slot_count > MAX_SLOTS:
            rejected["many_slots"] += 1
            continue

        # Estimate tokens saved per match
        estimated = template
        for i in range(slot_count):
            estimated = estimated.replace(f"{{{i}}}", "_varName")
        token_count = _get_bpe_token_count(estimated, enc)
        if token_count < MIN_TOKEN_SPAN:
            rejected["few_tokens"] += 1
            continue

        # Template cost: prefix (1 token) + BPE(args) + closer (1 token)
        args_str = ",".join(["_varName"] * slot_count)
        macro_cost = 2 + _get_bpe_token_count(args_str, enc)
        savings_per_match = token_count - macro_cost
        score = freq * max(savings_per_match, 0)
        scored.append((template, freq, slot_count, score, repo_count))

    scored.sort(key=lambda x: x[3], reverse=True)

    print(f"\nAST subtree candidates: {len(template_counter):,}")
    print(f"After filtering: {len(scored)} passed, rejected: {rejected}")

    return scored[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Mine AST subtree patterns")
    parser.add_argument("--corpus", type=str, required=True, help="Directory with source files")
    parser.add_argument("--language", type=str, required=True, help="Language config to use (e.g. csharp, python)")
    parser.add_argument("--tokenizer", type=str, default=DEFAULT_TOKENIZER, help="HuggingFace tokenizer for scoring")
    parser.add_argument("--top", type=int, default=100, help="Show top N templates")
    parser.add_argument("--min-freq", type=int, default=MIN_AST_TEMPLATE_FREQUENCY)
    parser.add_argument("--min-repos", type=int, default=MIN_REPOS)
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument(
        "--exclude-repos", type=str, nargs="+", default=None,
        help="Repos to exclude from mining",
    )
    args = parser.parse_args()

    results = mine_ast_templates(
        Path(args.corpus),
        top_n=args.top,
        min_frequency=args.min_freq,
        min_repos=args.min_repos,
        max_files=args.max_files,
        exclude_repos=args.exclude_repos,
        language=args.language,
        tokenizer_name=args.tokenizer,
    )

    print(f"\nTop {min(args.top, len(results))} AST templates by score:")
    for template, freq, slot_count, score, repo_count in results[:args.top]:
        print(
            f"  freq={freq:5d} slots={slot_count} "
            f"score={score:7.0f} repos={repo_count:2d} | {template!r}"
        )


if __name__ == "__main__":
    main()
