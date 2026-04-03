"""
Compression dictionary mapping boilerplate patterns to single macro tokens.

Each pattern is a multi-token string that gets replaced by a single macro token
like <|M001|>. The dictionary is bidirectional and supports JSON serialization.
"""

import json
from pathlib import Path

from sematok.languages import get_language

# 5-digit macro IDs (M00001-M99999, T00001-T99999) cap each type at 99,999 entries.
MAX_MACROS = 99999
MAX_TEMPLATES = 99999


def _make_macro_token(index: int) -> str:
    """Generate a macro token string like <|M00001|>, <|M00002|>, etc."""
    return f"<|M{index:05d}|>"


def _make_template_token(index: int) -> str:
    """Generate a template token string like <|T00001|>, <|T00002|>, etc."""
    return f"<|T{index:05d}|>"


class CompressionDictionary:
    """Bidirectional mapping between patterns and macro tokens."""

    def __init__(self):
        self.pattern_to_macro: dict[str, str] = {}
        self.macro_to_pattern: dict[str, str] = {}
        self.pattern_categories: dict[str, str] = {}
        # Template macro fields
        self.template_to_macro: dict[str, str] = {}
        self.macro_to_template: dict[str, str] = {}
        self.template_slots: dict[str, int] = {}
        self.template_categories: dict[str, str] = {}
        self._pattern_index: int = 0
        self._template_index: int = 0

    @classmethod
    def from_seed(cls, language: str) -> "CompressionDictionary":
        """Create a dictionary from the hand-picked seed patterns."""
        lang = get_language(language)
        d = cls()
        for i, (pattern, category) in enumerate(lang.seed_patterns, start=1):
            macro = _make_macro_token(i)
            d.pattern_to_macro[pattern] = macro
            d.macro_to_pattern[macro] = pattern
            d.pattern_categories[pattern] = category
        d._pattern_index = len(lang.seed_patterns)
        return d

    def add_pattern(self, pattern: str, category: str = "mined") -> str:
        """Add a new pattern to the dictionary. Returns the assigned macro token."""
        if pattern in self.pattern_to_macro:
            return self.pattern_to_macro[pattern]
        self._pattern_index += 1
        index = self._pattern_index
        if index > MAX_MACROS:
            raise ValueError(f"Cannot add more than {MAX_MACROS} exact macros (5-digit ID limit)")
        macro = _make_macro_token(index)
        self.pattern_to_macro[pattern] = macro
        self.macro_to_pattern[macro] = pattern
        self.pattern_categories[pattern] = category
        return macro

    def add_template(self, template: str, slot_count: int, category: str = "template") -> str:
        """Add a template pattern. Returns the assigned template macro token."""
        if template in self.template_to_macro:
            return self.template_to_macro[template]
        self._template_index += 1
        if self._template_index > MAX_TEMPLATES:
            raise ValueError(f"Cannot add more than {MAX_TEMPLATES} templates (5-digit ID limit)")
        macro = _make_template_token(self._template_index)
        self.template_to_macro[template] = macro
        self.macro_to_template[macro] = template
        self.template_slots[template] = slot_count
        self.template_categories[template] = category
        return macro

    def remove_pattern(self, pattern: str) -> None:
        """Remove a pattern from the dictionary."""
        if pattern in self.pattern_to_macro:
            macro = self.pattern_to_macro.pop(pattern)
            self.macro_to_pattern.pop(macro, None)
            self.pattern_categories.pop(pattern, None)

    @property
    def size(self) -> int:
        return len(self.pattern_to_macro)

    @property
    def macro_tokens(self) -> list[str]:
        """All macro token strings, sorted by ID."""
        return sorted(self.macro_to_pattern.keys())

    @property
    def patterns_by_length(self) -> list[str]:
        """All patterns sorted by length descending (for longest-match-first compression)."""
        return sorted(self.pattern_to_macro.keys(), key=len, reverse=True)

    @property
    def template_count(self) -> int:
        return len(self.template_to_macro)

    @property
    def templates_by_length(self) -> list[str]:
        """All templates sorted by length descending (for longest-match-first matching)."""
        return sorted(self.template_to_macro.keys(), key=len, reverse=True)

    @property
    def all_macro_tokens(self) -> list[str]:
        """All macro tokens (M + T), sorted, for tokenizer vocabulary."""
        return sorted(list(self.macro_to_pattern.keys()) + list(self.macro_to_template.keys()))

    def save(self, path: str | Path) -> None:
        """Save dictionary to JSON."""
        path = Path(path)
        data = {
            "patterns": [
                {
                    "pattern": pattern,
                    "macro": macro,
                    "category": self.pattern_categories.get(pattern, "unknown"),
                }
                for pattern, macro in self.pattern_to_macro.items()
            ],
            "templates": [
                {
                    "template": template,
                    "macro": macro,
                    "slots": self.template_slots[template],
                    "category": self.template_categories.get(template, "template"),
                }
                for template, macro in self.template_to_macro.items()
            ],
        }
        path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "CompressionDictionary":
        """Load dictionary from JSON."""
        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))
        d = cls()
        for entry in data["patterns"]:
            d.pattern_to_macro[entry["pattern"]] = entry["macro"]
            d.macro_to_pattern[entry["macro"]] = entry["pattern"]
            d.pattern_categories[entry["pattern"]] = entry.get("category", "unknown")
            idx = int(entry["macro"][3:-2])
            d._pattern_index = max(d._pattern_index, idx)
        for entry in data.get("templates", []):
            d.template_to_macro[entry["template"]] = entry["macro"]
            d.macro_to_template[entry["macro"]] = entry["template"]
            d.template_slots[entry["template"]] = entry["slots"]
            d.template_categories[entry["template"]] = entry.get("category", "template")
            idx = int(entry["macro"][3:-2])
            d._template_index = max(d._template_index, idx)
        return d

    @classmethod
    def merge(cls, *dicts: "CompressionDictionary") -> "CompressionDictionary":
        """Merge multiple dictionaries into one with fresh sequential IDs.

        Patterns are added in order. If the same pattern or template exists in
        multiple dictionaries, only the first occurrence is kept.
        """
        merged = cls()
        for d in dicts:
            for pattern in d.patterns_by_length:
                if pattern not in merged.pattern_to_macro:
                    category = d.pattern_categories.get(pattern, "mined")
                    merged.add_pattern(pattern, category=category)
            for template in d.templates_by_length:
                if template not in merged.template_to_macro:
                    slots = d.template_slots[template]
                    category = d.template_categories.get(template, "template")
                    merged.add_template(template, slots, category=category)
        return merged

    def stats(self) -> dict:
        """Summary statistics about the dictionary."""
        categories = {}
        for pattern, cat in self.pattern_categories.items():
            categories[cat] = categories.get(cat, 0) + 1
        for template, cat in self.template_categories.items():
            categories[cat] = categories.get(cat, 0) + 1
        return {
            "total_patterns": self.size,
            "total_templates": self.template_count,
            "categories": categories,
            "avg_pattern_length_chars": (
                sum(len(p) for p in self.pattern_to_macro) / self.size
                if self.size > 0
                else 0
            ),
        }

    def __repr__(self) -> str:
        return f"CompressionDictionary({self.size} patterns, {self.template_count} templates)"
