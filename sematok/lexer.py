"""
Tree-sitter based lexer for safe-zone detection.

Identifies regions in source code where compression is safe (not inside
string literals, comments, or other contexts where pattern replacement could
break semantics).
"""

from tree_sitter import Parser, Node

from sematok.languages import LanguageConfig, get_language

_lang: LanguageConfig | None = None
_parser: Parser | None = None


def _get_lang() -> LanguageConfig:
    global _lang
    if _lang is None:
        _lang = get_language("csharp")
    return _lang


def _get_parser() -> Parser:
    global _parser
    if _parser is None:
        _parser = Parser(_get_lang().tree_sitter_language)
    return _parser


def set_language(lang: LanguageConfig) -> None:
    """Switch the lexer to a different language."""
    global _lang, _parser
    _lang = lang
    _parser = Parser(lang.tree_sitter_language)


def _collect_unsafe_ranges(
    node: Node, source_bytes: bytes | None = None,
) -> list[tuple[int, int]]:
    """Collect byte ranges of unsafe nodes (iterative to handle deep ASTs)."""
    lang = _get_lang()
    ranges = []
    stack = [node]
    while stack:
        cur = stack.pop()
        if cur.type in lang.unsafe_node_types:
            if (lang.is_safe_override is not None
                    and source_bytes is not None
                    and lang.is_safe_override(cur, source_bytes)):
                stack.extend(reversed(cur.children))
                continue  # language says this node is safe
            ranges.append((cur.start_byte, cur.end_byte))
            continue  # Don't recurse into unsafe nodes
        stack.extend(reversed(cur.children))
    return ranges


def _invert_ranges(
    unsafe_ranges: list[tuple[int, int]], total_length: int
) -> list[tuple[int, int]]:
    """Convert unsafe ranges to safe ranges (the complement)."""
    if not unsafe_ranges:
        return [(0, total_length)] if total_length > 0 else []

    # Sort and merge overlapping unsafe ranges
    sorted_ranges = sorted(unsafe_ranges)
    merged = [sorted_ranges[0]]
    for start, end in sorted_ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Build safe ranges from the gaps
    safe = []
    prev_end = 0
    for start, end in merged:
        if prev_end < start:
            safe.append((prev_end, start))
        prev_end = end
    if prev_end < total_length:
        safe.append((prev_end, total_length))

    return safe


def get_safe_ranges(source: str) -> list[tuple[int, int]]:
    """
    Parse source code and return byte ranges where compression is safe.

    Safe = everything EXCEPT string literals, comments, and character literals.
    Languages can override specific unsafe nodes via is_safe_override (e.g.
    C# treats ``///`` doc comments as safe, Python could treat docstrings as safe).

    These ranges can be passed to Compressor.compress(source, safe_ranges=...).

    Returns:
        List of (start, end) byte ranges where compression can be applied.
        Ranges refer to byte offsets in the UTF-8 encoded source.
    """
    source_bytes = source.encode("utf-8")
    tree = _get_parser().parse(source_bytes)
    unsafe = _collect_unsafe_ranges(tree.root_node, source_bytes)
    return _invert_ranges(unsafe, len(source_bytes))


def get_unsafe_ranges(source: str) -> list[tuple[int, int]]:
    """
    Parse source code and return byte ranges where compression is NOT safe.

    Returns ranges covering string literals, comments, and character literals.
    """
    source_bytes = source.encode("utf-8")
    tree = _get_parser().parse(source_bytes)
    return _collect_unsafe_ranges(tree.root_node, source_bytes)


def parse_source(source: str) -> tuple[Node, bytes]:
    """Parse source, return (root_node, source_bytes).

    Callers can walk the AST to find identifier nodes within byte ranges.
    """
    source_bytes = source.encode("utf-8")
    tree = _get_parser().parse(source_bytes)
    return tree.root_node, source_bytes


def classify_source(source: str) -> dict:
    """
    Classify source into safe/unsafe regions with statistics.
    Useful for debugging and understanding compression coverage.
    """
    source_bytes = source.encode("utf-8")
    total = len(source_bytes)
    safe = get_safe_ranges(source)
    unsafe = get_unsafe_ranges(source)

    safe_bytes = sum(end - start for start, end in safe)
    unsafe_bytes = sum(end - start for start, end in unsafe)

    return {
        "total_bytes": total,
        "safe_bytes": safe_bytes,
        "unsafe_bytes": unsafe_bytes,
        "safe_pct": safe_bytes / total * 100 if total > 0 else 0,
        "safe_ranges": safe,
        "unsafe_ranges": unsafe,
    }
