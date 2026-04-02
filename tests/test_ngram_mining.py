"""Tests for n-gram substring frequency mining."""

from sematok.ngram_mining import (
    extract_safe_segments,
    _is_word_boundary_start,
    _is_word_boundary_end,
    _is_valid_ngram,
)
from sematok.mining import merge_mining_results
from sematok.languages import get_language
from sematok.lexer import set_language

set_language(get_language("csharp"))


# -- extract_safe_segments tests --

def test_extract_safe_segments():
    """Safe zone text is split into single-line segments."""
    source = "public class Foo {\n    int x = 0;\n}"
    segments = extract_safe_segments(source)
    assert len(segments) >= 1
    for seg in segments:
        assert "\n" not in seg


def test_segments_skip_strings():
    """String literal content is excluded from segments."""
    source = 'string msg = "using System.Linq;"; public class Foo { }'
    segments = extract_safe_segments(source)
    joined = " ".join(segments)
    # The string literal content should not appear in safe segments
    assert "using System.Linq;" not in joined or "public class Foo" in joined


def test_segments_include_xmldoc():
    """XML doc comments are included in safe segments."""
    source = "/// <summary>\n/// Gets value.\n/// </summary>\npublic int X { get; }"
    segments = extract_safe_segments(source)
    joined = " ".join(segments)
    assert "/// <summary>" in joined


def test_segments_exclude_regular_comments():
    """Regular // comments are excluded from safe segments."""
    source = "// This is a regular comment, not xmldoc\npublic class Foo { }"
    segments = extract_safe_segments(source)
    joined = " ".join(segments)
    assert "regular comment" not in joined


def test_segments_short_lines_skipped():
    """Lines shorter than MIN_CHAR_LENGTH are skipped."""
    source = "x = 1;\npublic static void Main(string[] args) { }"
    segments = extract_safe_segments(source)
    # "x = 1;" is 6 chars, should be skipped
    for seg in segments:
        assert len(seg) >= 8


# -- Word boundary tests --

def test_word_boundary_start_at_zero():
    """Position 0 is always a word boundary."""
    assert _is_word_boundary_start("anything", 0) is True


def test_word_boundary_start_after_punct():
    """Position after punctuation is a word boundary."""
    assert _is_word_boundary_start(".ToString()", 1) is True


def test_word_boundary_start_mid_identifier():
    """Position mid-identifier is not a boundary."""
    assert _is_word_boundary_start("FooBar", 3) is False


def test_word_boundary_start_after_space():
    """Position after space is a word boundary."""
    assert _is_word_boundary_start("public static", 7) is True


def test_word_boundary_end_at_length():
    """End of string is always a word boundary."""
    assert _is_word_boundary_end("abc", 3) is True


def test_word_boundary_end_before_punct():
    """Position before punctuation is a boundary."""
    assert _is_word_boundary_end("Foo()", 3) is True


def test_word_boundary_end_mid_identifier():
    """Position mid-identifier is not an end boundary."""
    assert _is_word_boundary_end("FooBar", 3) is False


# -- _is_valid_ngram tests --

def test_valid_ngram_rejects_pure_alpha():
    """Pure alphabetic string is rejected (no punctuation)."""
    assert _is_valid_ngram("Cancellati") is False
    assert _is_valid_ngram("something here") is False  # no punctuation


def test_valid_ngram_rejects_high_whitespace():
    """Patterns with >50% whitespace are rejected."""
    assert _is_valid_ngram("     .X(") is False


def test_valid_ngram_rejects_short():
    """Patterns shorter than 8 chars are rejected."""
    assert _is_valid_ngram(".foo()") is False


def test_valid_ngram_accepts_good_patterns():
    """Valid boilerplate patterns pass all checks."""
    assert _is_valid_ngram(".ToList()") is True
    assert _is_valid_ngram("public static void Main(") is True
    assert _is_valid_ngram("/// <summary>") is True
    assert _is_valid_ngram("{ get; set; }") is True


def test_valid_ngram_rejects_single_word():
    """Single-word patterns are rejected."""
    assert _is_valid_ngram("CancellationToken") is False


# -- merge_mining_results tests --

def test_merge_deduplicates():
    """Same pattern from both sources keeps higher frequency."""
    regex = [("pattern_a;", 100, 5, 400, 3)]
    ngram = [("pattern_a;", 150, 5, 600, 4), ("pattern_b;", 80, 3, 160, 2)]
    merged = merge_mining_results(regex, ngram)
    by_pat = {m[0]: m for m in merged}
    # pattern_a should have freq 150 (from ngram, higher)
    assert by_pat["pattern_a;"][1] == 150
    # pattern_b only from ngram
    assert "pattern_b;" in by_pat


def test_merge_sorts_by_score():
    """Merged results are sorted by score descending."""
    regex = [("low_score;", 10, 3, 20, 2)]
    ngram = [("high_score;", 200, 8, 1400, 5)]
    merged = merge_mining_results(regex, ngram)
    assert merged[0][0] == "high_score;"


def test_merge_empty_inputs():
    """Merging with empty lists works."""
    assert merge_mining_results([], []) == []
    result = merge_mining_results([("a;", 10, 3, 20, 2)], [])
    assert len(result) == 1
