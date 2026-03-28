"""Tests for compression dictionary, compressor, and decompressor."""

import json
import tempfile
from pathlib import Path

from sematok.dictionary import CompressionDictionary, SEED_PATTERNS
from sematok.compressor import Compressor
from sematok.decompressor import Decompressor
from sematok.lexer import get_safe_ranges, get_unsafe_ranges


# -- Dictionary tests --

def test_dictionary_from_seed():
    d = CompressionDictionary.from_seed()
    assert d.size == len(SEED_PATTERNS)
    assert d.size > 0


def test_dictionary_bidirectional():
    d = CompressionDictionary.from_seed()
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
    d = CompressionDictionary.from_seed()
    original_size = d.size
    pattern = list(d.pattern_to_macro.keys())[0]
    d.remove_pattern(pattern)
    assert d.size == original_size - 1
    assert pattern not in d.pattern_to_macro


def test_dictionary_save_load():
    d = CompressionDictionary.from_seed()
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w") as f:
        path = f.name

    d.save(path)
    d2 = CompressionDictionary.load(path)
    assert d2.size == d.size
    assert d2.pattern_to_macro == d.pattern_to_macro
    assert d2.macro_to_pattern == d.macro_to_pattern

    Path(path).unlink()


def test_dictionary_patterns_by_length():
    d = CompressionDictionary.from_seed()
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
    d = CompressionDictionary.from_seed()
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    compressed = compressor.compress(SAMPLE_CS_CODE)
    decompressed = decompressor.decompress(compressed)
    assert decompressed == SAMPLE_CS_CODE


def test_compression_reduces_size():
    """Compressed text should be shorter than original (for code with boilerplate)."""
    d = CompressionDictionary.from_seed()
    compressor = Compressor(d)

    compressed = compressor.compress(SAMPLE_CS_CODE)
    assert len(compressed) < len(SAMPLE_CS_CODE)


def test_compression_contains_macros():
    """Compressed output should contain macro tokens."""
    d = CompressionDictionary.from_seed()
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    compressed = compressor.compress(SAMPLE_CS_CODE)
    assert decompressor.contains_macros(compressed)


def test_empty_input():
    d = CompressionDictionary.from_seed()
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    assert compressor.compress("") == ""
    assert decompressor.decompress("") == ""


def test_no_matching_patterns():
    """Text with no C# boilerplate should pass through unchanged."""
    d = CompressionDictionary.from_seed()
    compressor = Compressor(d)

    plain_text = "Hello, this is just plain English text with no C# patterns."
    assert compressor.compress(plain_text) == plain_text


def test_compression_stats():
    d = CompressionDictionary.from_seed()
    compressor = Compressor(d)

    stats = compressor.compression_stats(SAMPLE_CS_CODE)
    assert stats["original_chars"] > stats["compressed_chars"]
    assert stats["patterns_matched"] > 0
    assert stats["total_replacements"] > 0
    assert stats["char_reduction_pct"] > 0


def test_multiple_occurrences():
    """Pattern appearing multiple times should all be replaced."""
    d = CompressionDictionary.from_seed()
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    source = "{ get; set; } and also { get; set; } and another { get; set; }"
    compressed = compressor.compress(source)
    assert compressed.count("{ get; set; }") == 0  # All replaced
    assert decompressor.decompress(compressed) == source


def test_safe_zones_compression():
    """Compression with safe zones should skip unsafe regions."""
    d = CompressionDictionary.from_seed()
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

def test_xmldoc_default_unsafe():
    """By default, /// XML doc comments are unsafe (not compressible)."""
    source = '/// <summary>\npublic class Foo { }'
    safe = get_safe_ranges(source)
    source_bytes = source.encode("utf-8")
    safe_text = b"".join(source_bytes[s:e] for s, e in safe)
    assert b"/// <summary>" not in safe_text


def test_xmldoc_becomes_safe():
    """With allow_xmldoc=True, /// comments become safe zones."""
    source = '/// <summary>\npublic class Foo { }'
    safe = get_safe_ranges(source, allow_xmldoc=True)
    source_bytes = source.encode("utf-8")
    safe_text = b"".join(source_bytes[s:e] for s, e in safe)
    assert b"/// <summary>" in safe_text


def test_regular_comment_stays_unsafe():
    """Regular // comments remain unsafe even with allow_xmldoc=True."""
    source = '// Regular comment\npublic class Foo { }'
    unsafe = get_unsafe_ranges(source, allow_xmldoc=True)
    source_bytes = source.encode("utf-8")
    unsafe_text = b"".join(source_bytes[s:e] for s, e in unsafe)
    assert b"// Regular comment" in unsafe_text


def test_block_comment_stays_unsafe():
    """Block comments remain unsafe even with allow_xmldoc=True."""
    source = '/* block */\npublic class Foo { }'
    unsafe = get_unsafe_ranges(source, allow_xmldoc=True)
    source_bytes = source.encode("utf-8")
    unsafe_text = b"".join(source_bytes[s:e] for s, e in unsafe)
    assert b"/* block */" in unsafe_text


def test_xmldoc_roundtrip_with_compression():
    """Compressing XML doc patterns in safe zones is lossless."""
    d = CompressionDictionary.from_seed()
    compressor = Compressor(d)
    decompressor = Decompressor(d)

    source = '/// <summary>\n/// </summary>\npublic class Foo { get; set; }'
    safe = get_safe_ranges(source, allow_xmldoc=True)
    compressed = compressor.compress(source, safe_ranges=safe)
    decompressed = decompressor.decompress(compressed)
    assert decompressed == source
