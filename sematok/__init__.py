"""
sematok -- Semantic token compression for code LLMs.

Replaces recurring multi-token boilerplate patterns with single macro tokens,
reducing context window usage without losing information.
"""

from sematok.compressor import Compressor
from sematok.decompressor import Decompressor
from sematok.dictionary import CompressionDictionary
from sematok.languages import (
    LanguageConfig,
    available_languages,
    get_dictionary_path,
    get_language,
)
from sematok.lexer import get_safe_ranges

__all__ = [
    "CompressionDictionary",
    "Compressor",
    "Decompressor",
    "LanguageConfig",
    "available_languages",
    "get_dictionary_path",
    "get_language",
    "get_safe_ranges",
]
