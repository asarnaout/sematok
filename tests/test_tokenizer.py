"""Tests for the extended tokenizer."""

from sematok.dictionary import CompressionDictionary
from tokenizer.extended_tokenizer import ExtendedTokenizer, BASE_VOCAB_SIZE


def test_vocab_size_includes_macros():
    d = CompressionDictionary.from_seed()
    tok = ExtendedTokenizer(d)
    assert tok.vocab_size == BASE_VOCAB_SIZE + d.size


def test_macro_token_encodes_to_single_id():
    """Each macro token should encode to exactly one token ID."""
    d = CompressionDictionary.from_seed()
    tok = ExtendedTokenizer(d)

    for macro_str in d.macro_tokens:
        ids = tok.encode(macro_str)
        assert len(ids) == 1, f"{macro_str} encoded to {len(ids)} tokens, expected 1"
        assert ids[0] >= BASE_VOCAB_SIZE, f"{macro_str} got ID {ids[0]}, expected >= {BASE_VOCAB_SIZE}"


def test_macro_roundtrip():
    """Encoding then decoding a macro token should return the original string."""
    d = CompressionDictionary.from_seed()
    tok = ExtendedTokenizer(d)

    for macro_str in d.macro_tokens:
        ids = tok.encode(macro_str)
        decoded = tok.decode(ids)
        assert decoded == macro_str


def test_mixed_text_encoding():
    """Text mixing regular content and macro tokens should encode correctly."""
    d = CompressionDictionary.from_seed()
    tok = ExtendedTokenizer(d)

    # Get the first macro token
    macro = d.macro_tokens[0]
    text = f"Hello world {macro} more text"

    ids = tok.encode(text)
    decoded = tok.decode(ids)
    assert decoded == text


def test_regular_text_unchanged():
    """Regular text (no macros) should encode the same as base GPT-2."""
    import tiktoken

    d = CompressionDictionary.from_seed()
    tok = ExtendedTokenizer(d)
    base = tiktoken.get_encoding("gpt2")

    text = "Hello, world! This is a test."
    assert tok.encode(text) == base.encode(text)


def test_is_macro_id():
    d = CompressionDictionary.from_seed()
    tok = ExtendedTokenizer(d)

    assert not tok.is_macro_id(0)
    assert not tok.is_macro_id(50256)
    assert tok.is_macro_id(BASE_VOCAB_SIZE)
    assert tok.is_macro_id(BASE_VOCAB_SIZE + 1)


def test_count_tokens():
    d = CompressionDictionary.from_seed()
    tok = ExtendedTokenizer(d)

    text = "Hello world"
    assert tok.count_tokens(text) == len(tok.encode(text))


def test_compressed_code_fewer_tokens():
    """Compressed C# code should produce fewer tokens than uncompressed."""
    from sematok.compressor import Compressor

    d = CompressionDictionary.from_seed()
    tok = ExtendedTokenizer(d)
    compressor = Compressor(d)

    source = """\
using System;
using System.Collections.Generic;

public static void Main(string[] args)
{
    Console.WriteLine("Hello");
}
"""
    compressed = compressor.compress(source)

    tokens_original = tok.count_tokens(source)
    tokens_compressed = tok.count_tokens(compressed)

    assert tokens_compressed < tokens_original, (
        f"Compressed ({tokens_compressed}) should be fewer than original ({tokens_original})"
    )
