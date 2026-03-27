"""
Decompresses macro tokens back to original C# source code.

This is the inverse of compressor.py. Decompression is always lossless
and deterministic -- each macro token maps to exactly one pattern.
"""

import re

from sematok.dictionary import CompressionDictionary

# Regex to match macro tokens like <|M001|>, <|M002|>, etc.
MACRO_PATTERN = re.compile(r"<\|M(\d{3})\|>")


class Decompressor:
    """Expands macro tokens back to their original C# patterns."""

    def __init__(self, dictionary: CompressionDictionary):
        self.dictionary = dictionary

    def decompress(self, compressed: str) -> str:
        """
        Decompress a string by replacing all macro tokens with original patterns.

        Args:
            compressed: String containing macro tokens like <|M001|>.

        Returns:
            Original C# source code with all macro tokens expanded.
        """

        def _replace_macro(match: re.Match) -> str:
            macro = match.group(0)  # e.g., <|M001|>
            if macro in self.dictionary.macro_to_pattern:
                return self.dictionary.macro_to_pattern[macro]
            # Unknown macro token -- leave it as-is (shouldn't happen with a valid dictionary)
            return macro

        return MACRO_PATTERN.sub(_replace_macro, compressed)

    def contains_macros(self, text: str) -> bool:
        """Check if a string contains any macro tokens."""
        return bool(MACRO_PATTERN.search(text))

    def list_macros(self, text: str) -> list[str]:
        """Return all macro tokens found in the text."""
        return MACRO_PATTERN.findall(text)
