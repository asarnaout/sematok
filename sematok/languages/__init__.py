"""
Language configuration registry for sematok.

Each supported language provides a config with tree-sitter grammar,
AST node types, candidate patterns, seed patterns, and structural names.
The compression engine is language-agnostic; all language-specific
knowledge lives in these configs.
"""

import importlib
import re
from dataclasses import dataclass, field


@dataclass
class LanguageConfig:
    """All language-specific constants needed by the sematok pipeline."""

    # --- Identity ---
    name: str                          # e.g. "csharp", "python"
    file_extension: str                # e.g. ".cs", ".py"
    tree_sitter_language: object = None  # tree_sitter.Language instance

    # --- Lexer: safe zone detection ---
    unsafe_node_types: set[str] = field(default_factory=set)
    # For XML doc-style comments that should be treated as safe
    safe_comment_prefix: bytes | None = None  # e.g. b"///" for C#

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

    # --- Data: repos and eval splits ---
    repos: list[tuple[str, str]] = field(default_factory=list)
    eval_repos: list[str] = field(default_factory=list)


_REGISTRY: dict[str, str] = {
    "csharp": "sematok.languages.csharp",
}


def get_language(name: str = "csharp") -> LanguageConfig:
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
