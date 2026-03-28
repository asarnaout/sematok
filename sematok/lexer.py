"""
Tree-sitter based C# lexer for safe-zone detection.

Identifies regions in C# source code where compression is safe (not inside
string literals, comments, or other contexts where pattern replacement could
break semantics).
"""

import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Parser, Node

CS_LANGUAGE = Language(tscsharp.language())

# Node types that are UNSAFE for compression (content should not be modified)
UNSAFE_NODE_TYPES = {
    "comment",
    "string_literal",
    "verbatim_string_literal",
    "interpolated_string_expression",
    "raw_string_literal",
    "character_literal",
}


def _create_parser() -> Parser:
    return Parser(CS_LANGUAGE)


# Module-level parser (reusable, not thread-safe)
_parser = _create_parser()


def _collect_unsafe_ranges(
    node: Node, source_bytes: bytes | None = None, allow_xmldoc: bool = False,
) -> list[tuple[int, int]]:
    """Recursively collect byte ranges of unsafe nodes."""
    ranges = []
    if node.type in UNSAFE_NODE_TYPES:
        if allow_xmldoc and node.type == "comment" and source_bytes is not None:
            text = source_bytes[node.start_byte:node.end_byte]
            if text.startswith(b"///"):
                return ranges  # XML doc comment -- treat as safe
        ranges.append((node.start_byte, node.end_byte))
        return ranges  # Don't recurse into unsafe nodes
    for child in node.children:
        ranges.extend(_collect_unsafe_ranges(child, source_bytes, allow_xmldoc))
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


def get_safe_ranges(source: str, allow_xmldoc: bool = False) -> list[tuple[int, int]]:
    """
    Parse C# source code and return byte ranges where compression is safe.

    Safe = everything EXCEPT string literals, comments, and character literals.
    These ranges can be passed to Compressor.compress(source, safe_ranges=...).

    Args:
        source: C# source code as a string.
        allow_xmldoc: If True, ``///`` XML doc comments are treated as safe
            (compressible). Regular ``//`` and ``/* */`` comments remain unsafe.

    Returns:
        List of (start, end) byte ranges where compression can be applied.
        Ranges refer to byte offsets in the UTF-8 encoded source.
    """
    source_bytes = source.encode("utf-8")
    tree = _parser.parse(source_bytes)
    unsafe = _collect_unsafe_ranges(tree.root_node, source_bytes, allow_xmldoc)
    return _invert_ranges(unsafe, len(source_bytes))


def get_unsafe_ranges(source: str, allow_xmldoc: bool = False) -> list[tuple[int, int]]:
    """
    Parse C# source code and return byte ranges where compression is NOT safe.

    Returns ranges covering string literals, comments, and character literals.
    """
    source_bytes = source.encode("utf-8")
    tree = _parser.parse(source_bytes)
    return _collect_unsafe_ranges(tree.root_node, source_bytes, allow_xmldoc)


def parse_source(source: str) -> tuple[Node, bytes]:
    """Parse C# source, return (root_node, source_bytes).

    Callers can walk the AST to find identifier nodes within byte ranges.
    """
    source_bytes = source.encode("utf-8")
    tree = _parser.parse(source_bytes)
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
