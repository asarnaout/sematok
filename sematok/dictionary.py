"""
Compression dictionary mapping C# boilerplate patterns to single macro tokens.

Each pattern is a multi-token string that gets replaced by a single macro token
like <|M001|>. The dictionary is bidirectional and supports JSON serialization.
"""

import json
from pathlib import Path


# Seed patterns: hand-picked C# boilerplate sorted by estimated frequency and token savings.
# Format: (pattern_string, category)
# These will be refined by frequency mining (mining.py) once we have a corpus.
SEED_PATTERNS = [
    # Indentation (compresses 6-10% of tokens that are pure whitespace)
    # Ordered longest-first so greedy cascade decomposes correctly
    ("                        ", "indent"),  # 24 spaces (6 levels)
    ("                    ", "indent"),      # 20 spaces (5 levels)
    ("                ", "indent"),          # 16 spaces (4 levels)
    ("            ", "indent"),              # 12 spaces (3 levels)
    ("        ", "indent"),                  # 8 spaces (2 levels)
    ("    ", "indent"),                      # 4 spaces (1 level)
    ("\t\t\t\t\t", "indent"),                # 5 tabs
    ("\t\t\t\t", "indent"),                  # 4 tabs
    ("\t\t\t", "indent"),                    # 3 tabs
    ("\t\t", "indent"),                      # 2 tabs
    ("\t", "indent"),                        # 1 tab

    # Using directives (very high frequency)
    ("using System;", "using"),
    ("using System.Collections.Generic;", "using"),
    ("using System.Linq;", "using"),
    ("using System.Text;", "using"),
    ("using System.Threading.Tasks;", "using"),
    ("using System.IO;", "using"),
    ("using Microsoft.Extensions.DependencyInjection;", "using"),
    ("using Microsoft.AspNetCore.Mvc;", "using"),
    ("using System.Collections;", "using"),
    ("using Xunit;", "using"),

    # Property patterns (extremely high frequency)
    ("{ get; set; }", "property"),
    ("{ get; private set; }", "property"),
    ("{ get; init; }", "property"),
    ("{ get; internal set; }", "property"),
    ("{ get; protected set; }", "property"),

    # Access modifier + keyword combos (high frequency)
    ("public static void", "modifier"),
    ("public static async Task", "modifier"),
    ("public override string ToString()", "modifier"),
    ("public override bool Equals(object", "modifier"),
    ("public override int GetHashCode()", "modifier"),
    ("private readonly", "modifier"),
    ("public abstract class", "modifier"),
    ("public sealed class", "modifier"),
    ("internal static class", "modifier"),
    ("public static class", "modifier"),

    # Common method signatures
    ("public static void Main(string[] args)", "signature"),
    ("static void Main(string[] args)", "signature"),
    ("public void Dispose()", "signature"),
    ("protected virtual void Dispose(bool disposing)", "signature"),

    # Exception patterns
    ("throw new ArgumentNullException(nameof(", "exception"),
    ("throw new NotImplementedException();", "exception"),
    ("throw new InvalidOperationException(", "exception"),
    ("throw new ArgumentException(", "exception"),
    ("throw new NotSupportedException();", "exception"),

    # Common expressions
    ("Console.WriteLine(", "expression"),
    ("Console.ReadLine();", "expression"),
    ("return Task.CompletedTask;", "expression"),
    ("= string.Empty;", "expression"),
    ("= new();", "expression"),
    ("nameof(", "expression"),

    # Attribute patterns
    ("[ApiController]", "attribute"),
    ("[HttpGet]", "attribute"),
    ("[HttpPost]", "attribute"),
    ("[Serializable]", "attribute"),
    ("[Obsolete]", "attribute"),
    ("[TestMethod]", "attribute"),
    ("[Fact]", "attribute"),

    # Generic type patterns
    ("IEnumerable<", "generic"),
    ("IList<", "generic"),
    ("Dictionary<string, ", "generic"),
    ("ILogger<", "generic"),
    ("IOptions<", "generic"),
    ("Task<IActionResult>", "generic"),

    # XML doc patterns
    ("/// <summary>", "xmldoc"),
    ("/// </summary>", "xmldoc"),
    ("/// <param name=\"", "xmldoc"),
    ("/// <returns>", "xmldoc"),
    ("/// <exception cref=\"", "xmldoc"),
]


def _make_macro_token(index: int) -> str:
    """Generate a macro token string like <|M001|>, <|M002|>, etc."""
    return f"<|M{index:03d}|>"


def _make_template_token(index: int) -> str:
    """Generate a template token string like <|T001|>, <|T002|>, etc."""
    return f"<|T{index:03d}|>"


class CompressionDictionary:
    """Bidirectional mapping between C# patterns and macro tokens."""

    def __init__(self):
        self.pattern_to_macro: dict[str, str] = {}
        self.macro_to_pattern: dict[str, str] = {}
        self.pattern_categories: dict[str, str] = {}
        # Template macro fields
        self.template_to_macro: dict[str, str] = {}
        self.macro_to_template: dict[str, str] = {}
        self.template_slots: dict[str, int] = {}
        self.template_categories: dict[str, str] = {}
        self._template_index: int = 0

    @classmethod
    def from_seed(cls) -> "CompressionDictionary":
        """Create a dictionary from the hand-picked seed patterns."""
        d = cls()
        for i, (pattern, category) in enumerate(SEED_PATTERNS, start=1):
            macro = _make_macro_token(i)
            d.pattern_to_macro[pattern] = macro
            d.macro_to_pattern[macro] = pattern
            d.pattern_categories[pattern] = category
        return d

    def add_pattern(self, pattern: str, category: str = "mined") -> str:
        """Add a new pattern to the dictionary. Returns the assigned macro token."""
        if pattern in self.pattern_to_macro:
            return self.pattern_to_macro[pattern]
        index = len(self.pattern_to_macro) + 1
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
        for entry in data.get("templates", []):
            d.template_to_macro[entry["template"]] = entry["macro"]
            d.macro_to_template[entry["macro"]] = entry["template"]
            d.template_slots[entry["template"]] = entry["slots"]
            d.template_categories[entry["template"]] = entry.get("category", "template")
            idx = int(entry["macro"][3:6])
            d._template_index = max(d._template_index, idx)
        return d

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
