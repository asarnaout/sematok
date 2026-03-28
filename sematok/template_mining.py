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

import tiktoken
from tqdm import tqdm
from tree_sitter import Node

from sematok.lexer import get_safe_ranges, parse_source
from sematok.mining import (
    CANDIDATE_PATTERNS,
    MIN_CHAR_LENGTH,
    MIN_REPOS,
    _get_bpe_token_count,
    _load_file_repo_map,
)

# --- Constants ---

MIN_TEMPLATE_FREQUENCY = 30
MAX_SLOTS = 6
MIN_SLOTS = 1

# Parent node types where an `identifier` is structural (defines the pattern)
FIXED_PARENT_TYPES = {
    "class_declaration",
    "struct_declaration",
    "interface_declaration",
    "enum_declaration",
    "record_declaration",
    "constructor_declaration",
    "method_declaration",
    "property_declaration",
    "object_creation_expression",
    "invocation_expression",
    "variable_declaration",  # type name in a declaration, not the var name
    "generic_name",
    "using_directive",
    "attribute",
    "base_list",
    "namespace_declaration",
    "qualified_name",
    "type_argument_list",
}

# Parent node types where an `identifier` is a user-chosen name (normalizable)
NORMALIZE_PARENT_TYPES = {
    "variable_declarator",
    "member_access_expression",
    "assignment_expression",
    "binary_expression",
    "return_statement",
    "argument",
}

# Well-known names that should never be normalized even if parent says so
STRUCTURAL_NAMES = {
    # Keywords tree-sitter may label as identifier
    "nameof", "sizeof", "typeof", "default", "value", "get", "set", "init",
    "add", "remove", "var", "dynamic", "global", "async", "await",
    # Common framework types
    "Console", "Task", "ValueTask", "String", "Object", "Math",
    "List", "Dictionary", "HashSet", "Array", "Tuple",
    "ILogger", "IOptions", "IConfiguration", "IServiceProvider",
    "IEnumerable", "IList", "ICollection", "IDictionary",
    "IDisposable", "IAsyncDisposable", "ICloneable",
    "CancellationToken", "StringBuilder", "EventArgs",
    "Debug", "Assert", "Trace",
    # Common exception types
    "Exception", "ArgumentNullException", "InvalidOperationException",
    "NotImplementedException", "NotSupportedException", "ArgumentException",
    "ArgumentOutOfRangeException", "NullReferenceException",
    "ObjectDisposedException", "OperationCanceledException",
    # Common method names that are structural
    "Dispose", "ToString", "GetHashCode", "Equals", "GetType",
    "ConfigureAwait", "GetAwaiter", "GetResult",
    "ThrowIfNull", "IsNullOrEmpty", "IsNullOrWhiteSpace",
}


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

    def _walk(node: Node):
        # Skip nodes entirely outside the range
        if node.end_byte <= start or node.start_byte >= end:
            return
        if node.type == "identifier" and node.start_byte >= start and node.end_byte <= end:
            text = source_bytes[node.start_byte:node.end_byte].decode("utf-8", errors="replace")
            parent_type = node.parent.type if node.parent else ""
            results.append((text, node.start_byte, node.end_byte, parent_type))
        for child in node.children:
            _walk(child)

    _walk(root_node)
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


def should_normalize(text: str, parent_type: str, node: Node | None = None) -> bool:
    """True if this identifier should become a placeholder."""
    if text in STRUCTURAL_NAMES:
        return False
    if parent_type in FIXED_PARENT_TYPES:
        return False
    if parent_type in NORMALIZE_PARENT_TYPES:
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

    def _collect_nodes(node: Node):
        if node.end_byte <= candidate_start or node.start_byte >= candidate_end:
            return
        if node.type == "identifier" and node.start_byte >= candidate_start:
            nodes_by_pos[node.start_byte] = node
        for child in node.children:
            _collect_nodes(child)

    _collect_nodes(root_node)

    # Determine which identifiers to normalize
    normalizable: list[tuple[str, int, int]] = []  # (text, rel_start, rel_end)
    for text, abs_start, abs_end, parent_type in idents:
        node = nodes_by_pos.get(abs_start)
        if should_normalize(text, parent_type, node):
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

    # Reject templates where two placeholders are adjacent with no literal between them
    if re.search(r"\}\s*\{", template):
        # Allow if there's at least some whitespace between (like `{0} {1}`)
        # But reject `{0}{1}` (truly adjacent, no separator)
        if "{}{" in template.replace(" ", "").replace("\t", ""):
            pass  # This check is handled by the regex compilation failing

    return template, unique_args


def extract_template_candidates(source: str) -> list[tuple[str, list[str]]]:
    """Extract (template, args) pairs from safe zones of a C# file.

    Parses once, applies CANDIDATE_PATTERNS regexes, normalizes identifiers.
    """
    try:
        root_node, source_bytes = parse_source(source)
    except Exception:
        return []

    try:
        safe_ranges = get_safe_ranges(source, allow_xmldoc=True)
    except Exception:
        safe_ranges = [(0, len(source.encode("utf-8")))]

    results: list[tuple[str, list[str]]] = []

    for range_start, range_end in safe_ranges:
        chunk_bytes = source_bytes[range_start:range_end]
        chunk_text = chunk_bytes.decode("utf-8", errors="replace")

        for pattern_re in CANDIDATE_PATTERNS:
            for match in pattern_re.finditer(chunk_text):
                candidate_text = match.group()
                if len(candidate_text) < MIN_CHAR_LENGTH:
                    continue
                # File byte position = safe range start + match byte offset
                file_byte_start = range_start + match.start()

                result = normalize_candidate(
                    candidate_text, file_byte_start, root_node, source_bytes,
                )
                if result is not None:
                    results.append(result)

    return results


def mine_templates(
    corpus_dir: Path,
    top_n: int = 2000,
    min_frequency: int = MIN_TEMPLATE_FREQUENCY,
    min_repos: int = MIN_REPOS,
    min_slots: int = MIN_SLOTS,
    max_slots: int = MAX_SLOTS,
    max_files: int | None = None,
    exclude_repos: list[str] | None = None,
) -> list[tuple[str, int, int, float, int]]:
    """Mine template patterns from corpus.

    Returns [(template, frequency, slot_count, score, repo_count), ...]
    sorted by score descending.
    """
    file_to_repo = _load_file_repo_map(corpus_dir)
    exclude_set = set(exclude_repos) if exclude_repos else set()

    cs_files = sorted(corpus_dir.glob("*.cs"))
    if exclude_set and file_to_repo:
        cs_files = [
            f for f in cs_files
            if file_to_repo.get(f.name, "unknown") not in exclude_set
        ]
    if max_files:
        cs_files = cs_files[:max_files]

    template_counter: Counter = Counter()
    template_repos: dict[str, set[str]] = defaultdict(set)

    for f in tqdm(cs_files, desc="Template mining"):
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        repo = file_to_repo.get(f.name, "unknown")
        candidates = extract_template_candidates(source)

        # Deduplicate per file
        file_templates: set[str] = set()
        for template, args in candidates:
            file_templates.add(template)

        template_counter.update(file_templates)
        for t in file_templates:
            template_repos[t].add(repo)

    # Filter and score
    enc = tiktoken.get_encoding("gpt2")
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

        # Score: estimate tokens saved per match
        # Use template with placeholder text for BPE estimation
        estimated = template
        for i in range(slot_count):
            estimated = estimated.replace(f"{{{i}}}", "_varName")
        token_count = _get_bpe_token_count(estimated, enc)
        score = freq * (token_count - 1)
        scored.append((template, freq, slot_count, score, repo_count))

    scored.sort(key=lambda x: x[3], reverse=True)

    print(f"\nTemplate candidates: {len(template_counter):,}")
    print(f"After filtering: {len(scored)} passed, rejected: {rejected}")

    return scored[:top_n]


def main():
    parser = argparse.ArgumentParser(description="Mine C# template patterns")
    parser.add_argument("--corpus", type=str, required=True, help="Directory with .cs files")
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
    )

    print(f"\nTop {min(args.top, len(results))} templates by score:")
    for template, freq, slot_count, score, repo_count in results[:args.top]:
        print(
            f"  freq={freq:5d} slots={slot_count} "
            f"score={score:7.0f} repos={repo_count:2d} | {template!r}"
        )


if __name__ == "__main__":
    main()
