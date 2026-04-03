"""
Compresses source code by replacing boilerplate patterns with macro tokens.

Uses longest-match-first strategy to avoid partial replacements.
During the replacement cascade, null-byte delimited placeholders are used
instead of final <|M...|> tokens to prevent patterns starting with '>'
or ending with '<' from colliding with macro delimiters.

Two-pass pipeline:
  1. Exact macros (<|M...|>) via str.replace, longest-first
  2. Template macros (<|T...:args|>) via compiled regex capture
"""

import re

from sematok.dictionary import CompressionDictionary
from sematok.languages import LanguageConfig, get_language

# During replacement, use \x00 delimiters that cannot appear in source text.
# Converted to final <|...|> format after all replacements are done.
_PLACEHOLDER_PREFIX = "\x00M"
_PLACEHOLDER_SUFFIX = "\x00"

_SLOT_RE = re.compile(r"\{(\d+)\}")


class Compressor:
    """Replaces boilerplate patterns with macro tokens."""

    def __init__(
        self,
        dictionary: CompressionDictionary,
        language: str | LanguageConfig,
        safe_zones: bool = False,
    ):
        self.dictionary = dictionary
        self.safe_zones = safe_zones
        # Resolve language config
        if isinstance(language, str):
            lang = get_language(language)
        else:
            lang = language
        self._ident_pattern = lang.ident_pattern
        # Pre-sort patterns by length descending for longest-match-first
        self._sorted_patterns = dictionary.patterns_by_length
        # Pre-compile template regexes
        self._template_regexes: list[tuple[str, re.Pattern, str]] = []
        self._compile_template_regexes()

    def compress(self, source: str, safe_ranges: list[tuple[int, int]] | None = None) -> str:
        """
        Compress source code by replacing patterns with macro tokens.

        Args:
            source: Raw source code.
            safe_ranges: Optional list of (start, end) byte ranges where compression
                         is allowed. If None, compress everywhere.
                         Provided by lexer.py's get_safe_ranges().

        Returns:
            Compressed source with macro tokens replacing matched patterns.
        """
        if not self._sorted_patterns and not self._template_regexes:
            return source

        if safe_ranges is not None:
            return self._compress_with_safe_zones(source, safe_ranges)
        return self._compress_simple(source)

    def _compile_template_regexes(self):
        """Pre-compile a regex for each template, sorted by length descending."""
        for template in self.dictionary.templates_by_length:
            macro_base = self.dictionary.template_to_macro[template]
            slot_count = self.dictionary.template_slots[template]
            # Split template on {N} placeholders and build regex
            pieces = _SLOT_RE.split(template)
            # Track which slot indices we've seen (for backreferences)
            seen_groups: dict[int, int] = {}  # slot_index -> regex group number
            regex_parts = []
            group_num = 0
            for i, piece in enumerate(pieces):
                if i % 2 == 0:
                    # Literal text
                    regex_parts.append(re.escape(piece))
                else:
                    # Slot reference
                    slot_idx = int(piece)
                    if slot_idx not in seen_groups:
                        group_num += 1
                        seen_groups[slot_idx] = group_num
                        regex_parts.append(self._ident_pattern)
                    else:
                        # Backreference to first occurrence of this slot
                        regex_parts.append(f"\\{seen_groups[slot_idx]}")
            try:
                compiled = re.compile("".join(regex_parts))
                self._template_regexes.append((template, compiled, macro_base))
            except re.error:
                pass  # Skip invalid regexes

    def _apply_templates(self, text: str) -> str:
        """Apply template macros, replacing matches with <|T001:arg1,arg2|>."""
        for template, regex, macro_base in self._template_regexes:
            slot_count = self.dictionary.template_slots[template]

            def _replace(m, _macro_base=macro_base):
                args = ",".join(m.groups())
                # macro_base is "<|T001|>", convert to "<|T001:args|>"
                return _macro_base[:-2] + ":" + args + "|>"

            text = regex.sub(_replace, text)
        return text

    @staticmethod
    def _finalize_placeholders(text: str) -> str:
        """Convert null-byte placeholders to final <|M...|> macro tokens."""
        return re.sub(
            _PLACEHOLDER_PREFIX + r"(\d+)" + _PLACEHOLDER_SUFFIX,
            r"<|M\1|>",
            text,
        )

    def _compress_simple(self, source: str) -> str:
        """Naive compression: replace patterns everywhere (no lexer awareness)."""
        result = source
        for pattern in self._sorted_patterns:
            macro = self.dictionary.pattern_to_macro[pattern]
            # Use null-byte placeholder during cascade to prevent delimiter collisions
            idx = macro[3:-2]  # extract "00001" from "<|M00001|>"
            placeholder = _PLACEHOLDER_PREFIX + idx + _PLACEHOLDER_SUFFIX
            result = result.replace(pattern, placeholder)
        result = self._finalize_placeholders(result)
        if self._template_regexes:
            result = self._apply_templates(result)
        return result

    def _compress_with_safe_zones(
        self, source: str, safe_ranges: list[tuple[int, int]]
    ) -> str:
        """
        Compress only within safe ranges (skip strings, comments, etc.).

        Strategy: process the source in chunks. For each safe range, apply
        compression. For unsafe ranges, pass through unchanged.
        """
        if not safe_ranges:
            return source

        # Convert byte offsets (from tree-sitter) to character offsets for str slicing
        source_bytes = source.encode("utf-8")
        char_ranges = []
        for byte_start, byte_end in sorted(safe_ranges):
            char_start = len(source_bytes[:byte_start].decode("utf-8", errors="replace"))
            char_end = len(source_bytes[:byte_end].decode("utf-8", errors="replace"))
            char_ranges.append((char_start, char_end))

        # Process chunks: compress safe ranges, pass through unsafe regions unchanged
        result_parts = []
        prev_end = 0

        for start, end in char_ranges:
            # Unsafe region before this safe range: pass through unchanged
            if prev_end < start:
                result_parts.append(source[prev_end:start])

            # Safe region: apply compression (exact macros first, then templates)
            chunk = source[start:end]
            for pattern in self._sorted_patterns:
                macro = self.dictionary.pattern_to_macro[pattern]
                idx = macro[3:-2]  # extract "00001" from "<|M00001|>"
                placeholder = _PLACEHOLDER_PREFIX + idx + _PLACEHOLDER_SUFFIX
                chunk = chunk.replace(pattern, placeholder)
            chunk = self._finalize_placeholders(chunk)
            if self._template_regexes:
                chunk = self._apply_templates(chunk)
            result_parts.append(chunk)

            prev_end = end

        # Trailing unsafe region
        if prev_end < len(source):
            result_parts.append(source[prev_end:])

        return "".join(result_parts)

    def compression_stats(self, source: str) -> dict:
        """Compute compression statistics for a given source."""
        compressed = self.compress(source)
        patterns_found = {}
        for pattern in self._sorted_patterns:
            count = source.count(pattern)
            if count > 0:
                patterns_found[pattern] = count

        return {
            "original_chars": len(source),
            "compressed_chars": len(compressed),
            "char_reduction": len(source) - len(compressed),
            "char_reduction_pct": (
                (1 - len(compressed) / len(source)) * 100 if len(source) > 0 else 0
            ),
            "patterns_matched": len(patterns_found),
            "total_replacements": sum(patterns_found.values()),
            "pattern_counts": patterns_found,
        }
