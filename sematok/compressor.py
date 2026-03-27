"""
Compresses C# source code by replacing boilerplate patterns with macro tokens.

Uses longest-match-first strategy to avoid partial replacements.
Optionally uses tree-sitter safe zones to skip strings and comments.
"""

from sematok.dictionary import CompressionDictionary


class Compressor:
    """Replaces C# boilerplate patterns with macro tokens."""

    def __init__(self, dictionary: CompressionDictionary, safe_zones: bool = False):
        self.dictionary = dictionary
        self.safe_zones = safe_zones
        # Pre-sort patterns by length descending for longest-match-first
        self._sorted_patterns = dictionary.patterns_by_length

    def compress(self, source: str, safe_ranges: list[tuple[int, int]] | None = None) -> str:
        """
        Compress C# source code by replacing patterns with macro tokens.

        Args:
            source: Raw C# source code.
            safe_ranges: Optional list of (start, end) byte ranges where compression
                         is allowed. If None, compress everywhere.
                         Provided by lexer.py's get_safe_ranges().

        Returns:
            Compressed source with macro tokens replacing matched patterns.
        """
        if not self._sorted_patterns:
            return source

        if safe_ranges is not None:
            return self._compress_with_safe_zones(source, safe_ranges)
        return self._compress_simple(source)

    def _compress_simple(self, source: str) -> str:
        """Naive compression: replace patterns everywhere (no lexer awareness)."""
        result = source
        for pattern in self._sorted_patterns:
            macro = self.dictionary.pattern_to_macro[pattern]
            result = result.replace(pattern, macro)
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

        # Merge safe ranges with gaps (unsafe regions) and process in order
        result_parts = []
        prev_end = 0

        for start, end in sorted(safe_ranges):
            # Unsafe region before this safe range: pass through unchanged
            if prev_end < start:
                result_parts.append(source[prev_end:start])

            # Safe region: apply compression
            chunk = source[start:end]
            for pattern in self._sorted_patterns:
                macro = self.dictionary.pattern_to_macro[pattern]
                chunk = chunk.replace(pattern, macro)
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
