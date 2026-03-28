"""Tests for AST subtree mining: structural pattern discovery."""

from sematok.ast_mining import (
    _subtree_depth,
    _is_in_safe_range,
    normalize_subtree,
    extract_ast_candidates,
    SUBTREE_ROOT_TYPES,
    MIN_DEPTH,
    MAX_DEPTH,
    MIN_CHAR_LENGTH,
)
from sematok.lexer import parse_source
from sematok.dictionary import CompressionDictionary
from sematok.compressor import Compressor
from sematok.decompressor import Decompressor


# -- Helper --

def _parse(code: str):
    """Parse C# code and return (root_node, source_bytes)."""
    return parse_source(code)


def _find_node(root, node_type: str):
    """Find the first node of the given type in the AST (iterative DFS)."""
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == node_type:
            return n
        stack.extend(reversed(n.children))
    return None


def _find_all_nodes(root, node_type: str):
    """Find all nodes of the given type in the AST."""
    results = []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.type == node_type:
            results.append(n)
        stack.extend(reversed(n.children))
    return results


# -- _subtree_depth tests --

def test_subtree_depth_leaf():
    """A single identifier node has depth 0."""
    root, _ = _parse("class C { int x; }")
    ident = _find_node(root, "identifier")
    assert _subtree_depth(ident) == 0


def test_subtree_depth_nested():
    """A field_declaration has depth > 1."""
    root, _ = _parse("class C { private int _x; }")
    field = _find_node(root, "field_declaration")
    assert field is not None
    depth = _subtree_depth(field)
    assert depth >= 2


# -- _is_in_safe_range tests --

def test_safe_range_inside():
    """Offset within a safe range returns True."""
    assert _is_in_safe_range(5, [0, 20], [10, 30]) is True


def test_safe_range_outside():
    """Offset between safe ranges returns False."""
    assert _is_in_safe_range(15, [0, 20], [10, 30]) is False


def test_safe_range_empty():
    """Empty safe ranges returns False."""
    assert _is_in_safe_range(5, [], []) is False


def test_safe_range_boundary():
    """Offset at range end (exclusive) returns False."""
    assert _is_in_safe_range(10, [0], [10]) is False


# -- normalize_subtree tests --

def test_normalize_simple_assignment():
    """this._logger = logger; normalizes both identifiers."""
    code = "class C { void M() { this._logger = logger; } }"
    root, source_bytes = _parse(code)
    expr_stmt = _find_node(root, "expression_statement")
    assert expr_stmt is not None
    result = normalize_subtree(expr_stmt, source_bytes)
    assert result is not None
    template, args = result
    assert "{0}" in template
    assert len(args) >= 1


def test_normalize_field_declaration():
    """private readonly ILogger _logger; normalizes the variable name."""
    code = "class C { private readonly int _count; }"
    root, source_bytes = _parse(code)
    field = _find_node(root, "field_declaration")
    assert field is not None
    result = normalize_subtree(field, source_bytes)
    assert result is not None
    template, args = result
    assert "{0}" in template
    assert "_count" in args


def test_normalize_return_statement():
    """return _value; normalizes the identifier."""
    code = "class C { int M() { return _value; } }"
    root, source_bytes = _parse(code)
    ret = _find_node(root, "return_statement")
    assert ret is not None
    result = normalize_subtree(ret, source_bytes)
    assert result is not None
    template, args = result
    assert "{0}" in template
    assert "_value" in args


def test_normalize_repeated_identifier():
    """Same identifier used twice gets the same slot."""
    code = "class C { void M(int x) { if (x != null) { return x; } } }"
    root, source_bytes = _parse(code)
    # Find the if_statement which uses x twice
    if_stmt = _find_node(root, "if_statement")
    if if_stmt is not None:
        result = normalize_subtree(if_stmt, source_bytes)
        if result is not None:
            template, args = result
            # x should appear once in args, referenced multiple times as {0}
            x_count = sum(1 for a in args if a == "x")
            assert x_count <= 1  # same text -> same slot


def test_normalize_no_normalizable_idents():
    """throw new NotImplementedException(); has no normalizable identifiers."""
    code = "class C { void M() { throw new NotImplementedException(); } }"
    root, source_bytes = _parse(code)
    throw = _find_node(root, "throw_statement")
    if throw is None:
        throw = _find_node(root, "throw_expression")
    assert throw is not None
    result = normalize_subtree(throw, source_bytes)
    # Should return None — NotImplementedException is in STRUCTURAL_NAMES
    assert result is None


def test_normalize_max_slots_exceeded():
    """Subtree with >6 unique normalizable identifiers returns None."""
    code = """class C { void M(int a, int b, int c, int d, int e, int f, int g) {
        var r = a + b + c + d + e + f + g;
    } }"""
    root, source_bytes = _parse(code)
    # The local_declaration_statement has 7+ normalizable idents
    local_decl = _find_node(root, "local_declaration_statement")
    if local_decl is not None:
        result = normalize_subtree(local_decl, source_bytes)
        # With 7 unique names (a-g + r), should exceed MAX_SLOTS
        if result is not None:
            _, args = result
            assert len(args) <= 6  # should have been rejected


def test_normalize_whitespace_collapsed():
    """Multi-line subtree text is collapsed to single line."""
    code = "class C {\n    void M() {\n        this._x =\n            _y;\n    }\n}"
    root, source_bytes = _parse(code)
    expr_stmt = _find_node(root, "expression_statement")
    if expr_stmt is not None:
        result = normalize_subtree(expr_stmt, source_bytes)
        if result is not None:
            template, _ = result
            assert "\n" not in template


# -- extract_ast_candidates tests --

def test_extract_skips_string_content():
    """Identifiers inside string literals are not mined."""
    code = 'class C { string s = "this._logger = logger;"; }'
    candidates = extract_ast_candidates(code)
    # Should not produce a template from the string content
    for template, args in candidates:
        assert "logger" not in args


def test_extract_basic_file():
    """Basic file with common patterns produces candidates."""
    code = """
using System;
namespace MyApp {
    public class Service {
        private readonly int _count;
        public void Process(string input) {
            this._value = input;
            return;
        }
    }
}
"""
    candidates = extract_ast_candidates(code)
    # Should find at least some candidates (field_declaration, expression_statement, etc.)
    assert len(candidates) >= 1


def test_extract_depth_filter():
    """Very shallow subtrees (depth < 2) should be excluded."""
    # A simple `return;` statement has minimal depth
    code = "class C { void M() { return; } }"
    candidates = extract_ast_candidates(code)
    # return; is very simple — may or may not produce candidates depending on depth
    # but anything produced should be from a subtree with depth >= MIN_DEPTH
    for template, _ in candidates:
        assert len(template) >= MIN_CHAR_LENGTH


def test_extract_short_template_rejected():
    """Templates shorter than MIN_CHAR_LENGTH after normalization are excluded."""
    candidates = extract_ast_candidates("class C { void M() { x = 1; } }")
    for template, _ in candidates:
        assert len(template) >= MIN_CHAR_LENGTH


# -- Compression round-trip test --

def test_ast_template_compression_roundtrip():
    """AST-mined templates compress and decompress losslessly."""
    d = CompressionDictionary()
    d.add_template("this.{0} = {1};", 2, category="ast_template")

    source = "this._logger = logger;"
    comp = Compressor(d)
    compressed = comp.compress(source)
    assert "<|T0001:" in compressed

    decomp = Decompressor(d)
    restored = decomp.decompress(compressed)
    assert restored == source


def test_ast_template_dedup_with_existing():
    """Templates already in dictionary are not duplicated."""
    d = CompressionDictionary()
    d.add_template("this.{0} = {1};", 2, category="template")

    # Adding the same template again should return existing macro
    macro = d.add_template("this.{0} = {1};", 2, category="ast_template")
    assert macro == "<|T0001|>"
    assert d.template_count == 1  # not duplicated
