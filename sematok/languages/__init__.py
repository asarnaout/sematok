"""
Language configuration registry for sematok.

Each supported language provides a config with tree-sitter grammar,
AST node types, candidate patterns, seed patterns, and structural names.
The compression engine is language-agnostic; all language-specific
knowledge lives in these configs.
"""

import importlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LanguageConfig:
    """All language-specific constants needed by the sematok pipeline."""

    # --- Identity ---
    name: str                          # e.g. "csharp", "python"
    file_extension: str                # e.g. ".cs", ".py"
    tree_sitter_language: object = None  # tree_sitter.Language instance

    # --- Lexer: safe zone detection ---
    unsafe_node_types: set[str] = field(default_factory=set)
    # Optional callback: (node, source_bytes) -> bool.
    # Called on nodes that matched unsafe_node_types. Return True to override
    # and treat the node as safe (e.g. C# "///" doc comments, Python docstrings).
    is_safe_override: Callable[[Any, bytes], bool] | None = None

    # --- Mining: regex candidate extraction ---
    candidate_patterns: list[re.Pattern] = field(default_factory=list)
    seed_patterns: list[tuple[str, str]] = field(default_factory=list)

    # --- AST mining: subtree extraction ---
    subtree_root_types: set[str] = field(default_factory=set)

    # --- Template mining: identifier normalization ---
    fixed_parent_types: set[str] = field(default_factory=set)
    normalize_parent_types: set[str] = field(default_factory=set)
    structural_names: set[str] = field(default_factory=set)

    # --- Compressor: identifier capture ---
    ident_pattern: str = r"([a-zA-Z_]\w*)"

    # --- Download: paths to skip when extracting source files ---
    # Substrings matched against lowercased file paths during corpus extraction.
    skip_path_patterns: list[str] = field(default_factory=list)

    # --- Data: repos and eval splits ---
    repos: list[tuple[str, str]] = field(default_factory=list)
    eval_repos: list[str] = field(default_factory=list)


_REGISTRY: dict[str, str] = {
    "csharp": "sematok.languages.csharp",  # sematok/languages/csharp/__init__.py
    "python": "sematok.languages.python",  # sematok/languages/python/__init__.py
    "java": "sematok.languages.java",      # sematok/languages/java/__init__.py
}


def get_language(name: str) -> LanguageConfig:
    """Load a language config by name."""
    module_path = _REGISTRY.get(name)
    if module_path is None:
        available = ", ".join(sorted(_REGISTRY.keys()))
        raise ValueError(f"Unknown language {name!r}. Available: {available}")
    mod = importlib.import_module(module_path)
    return mod.get_config()


def available_languages() -> list[str]:
    """List registered language names."""
    return sorted(_REGISTRY.keys())


def get_dictionary_path(language: str) -> Path | None:
    """Return path to the shipped dictionary for a language, or None if absent."""
    pkg_dir = Path(__file__).parent / language
    dict_path = pkg_dir / "dictionary.json"
    return dict_path if dict_path.exists() else None
