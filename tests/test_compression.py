"""Tests for compression dictionary, compressor, and decompressor."""

import json
import tempfile
from pathlib import Path

from sematok.dictionary import CompressionDictionary
from sematok.languages import get_language
from sematok.compressor import Compressor
from sematok.decompressor import Decompressor
from sematok.lexer import get_safe_ranges, get_unsafe_ranges, set_language

set_language(get_language("csharp"))


# -- Dictionary tests --

def test_dictionary_from_seed():
    d = CompressionDictionary.from_seed("csharp")
    assert d.size == len(get_language("csharp").seed_patterns)
    assert d.size > 0


def test_dictionary_bidirectional():
    d = CompressionDictionary.from_seed("csharp")
    for pattern, macro in d.pattern_to_macro.items():
        assert d.macro_to_pattern[macro] == pattern


def test_dictionary_add_pattern():
    d = CompressionDictionary()
    macro = d.add_pattern("new pattern", "test")
    assert d.size == 1
    assert d.pattern_to_macro["new pattern"] == macro
    assert d.macro_to_pattern[macro] == "new pattern"


def test_dictionary_add_duplicate():
    d = CompressionDictionary()
    m1 = d.add_pattern("pattern", "test")
    m2 = d.add_pattern("pattern", "test")
    assert m1 == m2
    assert d.size == 1


def test_dictionary_remove():
    d = CompressionDictionary.from_seed("csharp")
    original_size = d.size
    pattern = list(d.pattern_to_macro.keys())[0]
    d.remove_pattern(pattern)
    assert d.size == original_size - 1
    assert pattern not in d.pattern_to_macro


def test_dictionary_save_load():
    d = CompressionDictionary.from_seed("csharp")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name

    d.save(path)
    d2 = CompressionDictionary.load(path)
    assert d2.size == d.size
    assert d2.pattern_to_macro == d.pattern_to_macro
    assert d2.macro_to_pattern == d.macro_to_pattern

    Path(path).unlink()


def test_dictionary_patterns_by_length():
    d = CompressionDictionary.from_seed("csharp")
    patterns = d.patterns_by_length
    for i in range(len(patterns) - 1):
        assert len(patterns[i]) >= len(patterns[i + 1])


# -- Round-trip compression tests --

SAMPLE_CS_CODE = """\
using System;
using System.Collections.Generic;
using System.Linq;

namespace MyApp
{
    public sealed class Program
    {
        public static void Main(string[] args)
        {
            Console.WriteLine("Hello, World!");
        }

        public string Name { get; set; }
        public int Age { get; private set; }

        public override string ToString()
        {
            throw new NotImplementedException();
        }
    }
}
"""


def test_roundtrip_lossless():
    """Core invariant: decompress(compress(source)) == source."""
    d = CompressionDictionary.from_seed("csharp")
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    compressed = compressor.compress(SAMPLE_CS_CODE)
    decompressed = decompressor.decompress(compressed)
    assert decompressed == SAMPLE_CS_CODE


def test_compression_reduces_size():
    """Compressed text should be shorter than original (for code with boilerplate)."""
    d = CompressionDictionary.from_seed("csharp")
    compressor = Compressor(d)

    compressed = compressor.compress(SAMPLE_CS_CODE)
    assert len(compressed) < len(SAMPLE_CS_CODE)


def test_compression_contains_macros():
    """Compressed output should contain macro tokens."""
    d = CompressionDictionary.from_seed("csharp")
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    compressed = compressor.compress(SAMPLE_CS_CODE)
    assert decompressor.contains_macros(compressed)


def test_empty_input():
    d = CompressionDictionary.from_seed("csharp")
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    assert compressor.compress("") == ""
    assert decompressor.decompress("") == ""


def test_no_matching_patterns():
    """Text with no C# boilerplate should pass through unchanged."""
    d = CompressionDictionary.from_seed("csharp")
    compressor = Compressor(d)

    plain_text = "Hello, this is just plain English text with no C# patterns."
    assert compressor.compress(plain_text) == plain_text


def test_compression_stats():
    d = CompressionDictionary.from_seed("csharp")
    compressor = Compressor(d)

    stats = compressor.compression_stats(SAMPLE_CS_CODE)
    assert stats["original_chars"] > stats["compressed_chars"]
    assert stats["patterns_matched"] > 0
    assert stats["total_replacements"] > 0
    assert stats["char_reduction_pct"] > 0


def test_multiple_occurrences():
    """Pattern appearing multiple times should all be replaced."""
    d = CompressionDictionary.from_seed("csharp")
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    source = "{ get; set; } and also { get; set; } and another { get; set; }"
    compressed = compressor.compress(source)
    assert compressed.count("{ get; set; }") == 0  # All replaced
    assert decompressor.decompress(compressed) == source


def test_safe_zones_compression():
    """Compression with safe zones should skip unsafe regions."""
    d = CompressionDictionary.from_seed("csharp")
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    source = 'string msg = "using System;"; using System;'
    # Mark only the part after the string literal as safe (chars 30+)
    safe_ranges = [(30, len(source))]
    compressed = compressor.compress(source, safe_ranges=safe_ranges)

    # The "using System;" inside the string should NOT be compressed
    # The one outside should be compressed
    decompressed = decompressor.decompress(compressed)
    assert decompressed == source


# -- XML doc safe zone tests --

def test_xmldoc_becomes_safe():
    """C# is_safe_override makes /// comments safe zones."""
    source = '/// <summary>\npublic class Foo { }'
    safe = get_safe_ranges(source)
    source_bytes = source.encode("utf-8")
    safe_text = b"".join(source_bytes[s:e] for s, e in safe)
    assert b"/// <summary>" in safe_text


def test_regular_comment_stays_unsafe():
    """Regular // comments remain unsafe (is_safe_override only matches ///)."""
    source = '// Regular comment\npublic class Foo { }'
    unsafe = get_unsafe_ranges(source)
    source_bytes = source.encode("utf-8")
    unsafe_text = b"".join(source_bytes[s:e] for s, e in unsafe)
    assert b"// Regular comment" in unsafe_text


def test_block_comment_stays_unsafe():
    """Block comments remain unsafe (is_safe_override only matches ///)."""
    source = '/* block */\npublic class Foo { }'
    unsafe = get_unsafe_ranges(source)
    source_bytes = source.encode("utf-8")
    unsafe_text = b"".join(source_bytes[s:e] for s, e in unsafe)
    assert b"/* block */" in unsafe_text


def test_xmldoc_roundtrip_with_compression():
    """Compressing XML doc patterns in safe zones is lossless."""
    d = CompressionDictionary.from_seed("csharp")
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    source = '/// <summary>\n/// </summary>\npublic class Foo { get; set; }'
    safe = get_safe_ranges(source)
    compressed = compressor.compress(source, safe_ranges=safe)
    decompressed = decompressor.decompress(compressed)
    assert decompressed == source


# -- Template macro tests --


def test_template_add_and_lookup():
    """Templates get assigned T macros and can be looked up."""
    d = CompressionDictionary()
    macro = d.add_template("this.{0} = {1};", slot_count=2)
    assert macro == "<|T00001|>"
    assert d.template_to_macro["this.{0} = {1};"] == "<|T00001|>"
    assert d.macro_to_template["<|T00001|>"] == "this.{0} = {1};"
    assert d.template_slots["this.{0} = {1};"] == 2
    assert d.template_count == 1


def test_template_add_duplicate():
    """Adding the same template twice returns the existing macro."""
    d = CompressionDictionary()
    m1 = d.add_template("this.{0} = {1};", slot_count=2)
    m2 = d.add_template("this.{0} = {1};", slot_count=2)
    assert m1 == m2
    assert d.template_count == 1


def test_template_save_load_roundtrip(tmp_path):
    """Templates survive JSON serialization."""
    d = CompressionDictionary.from_seed("csharp")
    d.add_template("this.{0} = {1};", slot_count=2)
    d.add_template("return {0};", slot_count=1)
    path = tmp_path / "dict.json"
    d.save(path)

    loaded = CompressionDictionary.load(path)
    assert loaded.template_count == 2
    assert loaded.macro_to_template["<|T00001|>"] == "this.{0} = {1};"
    assert loaded.macro_to_template["<|T00002|>"] == "return {0};"
    assert loaded.template_slots["this.{0} = {1};"] == 2


def test_template_save_load_no_templates(tmp_path):
    """Loading a dictionary without templates key works (backward compat)."""
    d = CompressionDictionary.from_seed("csharp")
    path = tmp_path / "dict.json"
    d.save(path)
    loaded = CompressionDictionary.load(path)
    assert loaded.template_count == 0
    assert loaded.size == d.size


def test_template_decompression():
    """Template macro with args decompresses to original text."""
    d = CompressionDictionary()
    d.add_template("this.{0} = {1};", slot_count=2)
    decompressor = Decompressor(d)

    compressed = "public Foo() { <|T00001:_logger,logger|> }"
    result = decompressor.decompress(compressed)
    assert result == "public Foo() { this._logger = logger; }"


def test_template_compression_roundtrip():
    """decompress(compress(source)) == source for template-compressible code."""
    d = CompressionDictionary()
    d.add_template("this.{0} = {1};", slot_count=2)
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    source = "public Foo(ILogger logger) { this._logger = logger; }"
    compressed = compressor.compress(source)
    assert "<|T00001:" in compressed
    decompressed = decompressor.decompress(compressed)
    assert decompressed == source


def test_exact_before_template():
    """Exact macros match first; templates only apply to remaining text."""
    d = CompressionDictionary()
    d.add_pattern("throw new NotImplementedException();", "exception")
    d.add_template("throw new {0}();", slot_count=1)
    compressor = Compressor(d)

    source = "throw new NotImplementedException(); throw new ArgumentException();"
    compressed = compressor.compress(source)
    # NotImplementedException caught by exact macro
    assert "<|M00001|>" in compressed
    # ArgumentException caught by template
    assert "<|T00001:ArgumentException|>" in compressed


def test_template_with_safe_zones():
    """Templates skip unsafe regions (strings)."""
    d = CompressionDictionary()
    d.add_template("this.{0} = {1};", slot_count=2)
    compressor = Compressor(d)

    source = 'string s = "this._x = x;"; this._y = y;'
    safe = get_safe_ranges(source)
    compressed = compressor.compress(source, safe_ranges=safe)
    # The template inside the string should NOT be compressed
    assert '"this._x = x;"' in compressed
    # The one in code SHOULD be compressed
    assert "<|T00001:_y,y|>" in compressed


def test_template_repeated_slot():
    """Repeated {0} uses backreference -- same identifier at both positions."""
    d = CompressionDictionary()
    d.add_template("{0} ?? throw new ArgumentNullException(nameof({0}))", slot_count=1)
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    source = "x ?? throw new ArgumentNullException(nameof(x))"
    compressed = compressor.compress(source)
    assert "<|T00001:x|>" in compressed
    assert decompressor.decompress(compressed) == source


def test_no_templates_backward_compatible():
    """Empty template dictionary = exact-only compression, no errors."""
    d = CompressionDictionary.from_seed("csharp")
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    source = "using System; public class Foo { get; set; }"
    compressed = compressor.compress(source)
    assert "<|T" not in compressed
    assert decompressor.decompress(compressed) == source
