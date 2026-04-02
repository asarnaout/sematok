"""
Language configuration template for sematok.

Copy this file to sematok/languages/<your_language>/__init__.py and fill in
each section.  See sematok/languages/csharp/__init__.py for a complete example.

After filling it in, register the language in sematok/languages/__init__.py:

    _REGISTRY["mylang"] = "sematok.languages.mylang"

Required tree-sitter dependency: pip install tree-sitter-<grammar>
"""

import re

# from tree_sitter import Language
# import tree_sitter_<grammar> as ts_grammar

from sematok.languages import LanguageConfig


def get_config() -> LanguageConfig:
    """Return the language configuration."""
    return LanguageConfig(
        # --- Identity ---
        name="mylang",          # short name, used as --language CLI value
        file_extension=".ext",  # e.g. ".py", ".js", ".rs"
        # tree_sitter_language=Language(ts_grammar.language()),

        # --- Lexer: safe zone detection ---
        # Nodes whose byte ranges should be excluded from compression.
        # Typically: comments, string literals, character literals.
        unsafe_node_types={
            # "comment",
            # "string_literal",
        },
        # Optional callback: (node, source_bytes) -> bool.
        # Called on nodes matching unsafe_node_types. Return True to override
        # and treat the node as safe. See csharp/ for an example that treats
        # "///" doc comments as safe.
        is_safe_override=None,

        # --- Mining: regex candidate extraction ---
        # Patterns that match boilerplate snippets in the language.
        # Each pattern should capture a self-contained, frequently-repeated fragment.
        candidate_patterns=[
            # re.compile(r"import\s+[\w.]+"),
        ],

        # --- Dictionary: seed patterns ---
        # (pattern_text, category) pairs for high-frequency boilerplate
        # that should always be considered during mining.
        seed_patterns=[
            # ("import os", "import"),
        ],

        # --- AST mining: subtree root types ---
        # tree-sitter node types to use as roots when extracting AST subtrees.
        subtree_root_types={
            # "expression_statement",
            # "return_statement",
        },

        # --- Template mining: identifier normalization ---
        # Parent types where the child identifier defines the pattern (keep as-is).
        fixed_parent_types={
            # "class_definition",
            # "function_definition",
        },
        # Parent types where the child identifier is a user-chosen name (normalize).
        normalize_parent_types={
            # "assignment",
            # "argument",
        },
        # Well-known names that should never be normalized.
        structural_names={
            # "print", "self", "None", "True", "False",
        },

        # --- Compressor: identifier capture ---
        # Regex for capturing identifiers in template macros.
        ident_pattern=r"([a-zA-Z_]\w*)",

        # --- Download: paths to skip when extracting source files ---
        # Substrings matched against lowercased file paths during corpus extraction.
        skip_path_patterns=[
            # "obj/", "bin/",  # C# example
            # "__pycache__/", ".pyc",  # Python example
        ],

        # --- Data: repos and eval splits ---
        # (owner, repo) pairs for mining corpora.
        repos=[
            # ("owner", "repo"),
        ],
        # "owner--repo" strings for held-out evaluation set.
        eval_repos=[
            # "owner--repo",
        ],
    )
