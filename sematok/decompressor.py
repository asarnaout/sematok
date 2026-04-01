"""
Decompresses macro tokens back to original source code.

This is the inverse of compressor.py. Decompression is always lossless
and deterministic -- each macro token maps to exactly one pattern.
"""

import re

from sematok.dictionary import CompressionDictionary

# Regex to match exact macro tokens like <|M00001|>
MACRO_PATTERN = re.compile(r"<\|M(\d+)\|>")

# Regex to match template macro tokens like <|T00001:_logger,logger|>
TEMPLATE_MACRO_PATTERN = re.compile(r"<\|T(\d+):([^|]*)\|>")

# Matches either M or T macros (for contains_macros / list_macros)
ANY_MACRO_PATTERN = re.compile(r"<\|[MT]\d+(?::[^|]*)?\|>")


class Decompressor:
    """Expands macro tokens back to their original patterns."""

    def __init__(self, dictionary: CompressionDictionary):
        self.dictionary = dictionary

    def decompress(self, compressed: str) -> str:
        """
        Decompress a string by replacing all macro tokens with original patterns.

        Expands template macros (T) first, then exact macros (M).
        """

        def _replace_template(match: re.Match) -> str:
            macro_base = f"<|T{match.group(1)}|>"
            args = match.group(2).split(",")
            template = self.dictionary.macro_to_template.get(macro_base)
            if template is None:
                return match.group(0)
            result = template
            for i, arg in enumerate(args):
                result = result.replace(f"{{{i}}}", arg)
            return result

        def _replace_macro(match: re.Match) -> str:
            macro = match.group(0)
            if macro in self.dictionary.macro_to_pattern:
                return self.dictionary.macro_to_pattern[macro]
            return macro

        result = TEMPLATE_MACRO_PATTERN.sub(_replace_template, compressed)
        result = MACRO_PATTERN.sub(_replace_macro, result)
        return result

    def contains_macros(self, text: str) -> bool:
        """Check if a string contains any macro tokens (M or T)."""
        return bool(ANY_MACRO_PATTERN.search(text))

    def list_macros(self, text: str) -> list[str]:
        """Return all macro tokens found in the text (M and T)."""
        return ANY_MACRO_PATTERN.findall(text)
