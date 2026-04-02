"""Python language configuration for sematok."""

import re

import tree_sitter_python as tspython
from tree_sitter import Language

from sematok.languages import LanguageConfig


def _is_docstring(node, source_bytes: bytes) -> bool:
    """Treat Python docstrings as safe (compressible).

    A docstring is a triple-quoted string that is the first statement in
    a module, class, or function body.  These contain highly repetitive
    boilerplate (Args:, Returns:, :param, etc.) worth compressing.
    """
    if node.type != "string":
        return False
    text = source_bytes[node.start_byte:node.end_byte]
    if not (text.startswith(b'"""') or text.startswith(b"'''")):
        return False
    parent = node.parent
    if parent is None or parent.type != "expression_statement":
        return False
    # expression_statement must sit directly inside a module or block
    container = parent.parent
    if container is None or container.type not in ("module", "block"):
        return False
    # Must be the first named child (the actual first statement)
    for child in container.children:
        if child.is_named:
            return child.id == parent.id
    return False


def get_config() -> LanguageConfig:
    """Return the Python language configuration."""
    return LanguageConfig(
        name="python",
        file_extension=".py",
        tree_sitter_language=Language(tspython.language()),
        unsafe_node_types=UNSAFE_NODE_TYPES,
        is_safe_override=_is_docstring,
        candidate_patterns=CANDIDATE_PATTERNS,
        seed_patterns=SEED_PATTERNS,
        subtree_root_types=SUBTREE_ROOT_TYPES,
        fixed_parent_types=FIXED_PARENT_TYPES,
        normalize_parent_types=NORMALIZE_PARENT_TYPES,
        structural_names=STRUCTURAL_NAMES,
        ident_pattern=r"([a-zA-Z_]\w*)",
        skip_path_patterns=SKIP_PATH_PATTERNS,
        repos=REPOS,
        eval_repos=EVAL_REPOS,
    )


# ---------------------------------------------------------------------------
# Lexer: safe zone detection
# ---------------------------------------------------------------------------

UNSAFE_NODE_TYPES = {
    "comment",                # # line comments
    "string",                 # all string literals (incl. triple-quoted, raw, byte, f-strings)
    "concatenated_string",    # implicit "a" "b" concatenation
}

# ---------------------------------------------------------------------------
# Mining: regex candidate extraction
# ---------------------------------------------------------------------------

CANDIDATE_PATTERNS = [
    # Import patterns
    re.compile(r"from\s+[\w.]+\s+import\s+[\w, ]+"),
    re.compile(r"import\s+[\w.]+(?:\s+as\s+\w+)?"),

    # Decorator patterns
    re.compile(r"@(?:property|staticmethod|classmethod|abstractmethod)"),
    re.compile(r"@pytest\.mark\.\w+(?:\([^)\n]*\))?"),
    re.compile(r"@app\.(?:route|get|post|put|delete|patch)\([^)\n]*\)"),
    re.compile(r"@(?:override|deprecated|cache|cached_property|lru_cache)(?:\([^)\n]*\))?"),
    re.compile(r"@(?:dataclass|dataclasses\.dataclass)(?:\([^)\n]*\))?"),

    # Dunder method signatures
    re.compile(
        r"def\s+__(?:init|repr|str|eq|ne|lt|le|gt|ge|hash|len|iter|next"
        r"|getitem|setitem|delitem|contains|call|enter|exit"
        r"|new|del|bool|int|float|bytes|format"
        r"|add|sub|mul|truediv|floordiv|mod|pow"
        r"|radd|rsub|rmul|rtruediv|rfloordiv|rmod|rpow"
        r"|iadd|isub|imul|itruediv|ifloordiv|imod|ipow"
        r"|getattr|setattr|delattr|get|set|delete"
        r"|copy|deepcopy|reduce|reduce_ex)__\s*\([^)\n]{0,80}\)"
    ),

    # Common method signatures with type hints
    re.compile(
        r"def\s+\w+\(self(?:,\s*[^)\n]{0,100})?\)\s*->\s*"
        r"(?:None|str|int|bool|float|list|dict|tuple|set|bytes|Any|Self"
        r"|Optional\[\w+\]|Iterator\[\w+\])"
    ),

    # Type hint patterns
    re.compile(r":\s*(?:Optional|Union|List|Dict|Tuple|Set|FrozenSet|Type|Callable)\["),
    re.compile(r"->\s*(?:None|str|int|bool|float|list|dict|tuple|set|bytes|Any)"),

    # Raise / exception patterns
    re.compile(
        r"raise\s+(?:ValueError|TypeError|KeyError|AttributeError|RuntimeError"
        r"|NotImplementedError|StopIteration|FileNotFoundError|ImportError"
        r"|OSError|IOError|IndexError|AssertionError)\([^)\n]*\)"
    ),
    re.compile(r"except\s+(?:\w+(?:\s+as\s+\w+)?|\([^)\n]+\))\s*:"),

    # Common expressions
    re.compile(r"if\s+__name__\s*==\s*[\"']__main__[\"']\s*:"),
    re.compile(r"super\(\)\.__init__\([^)\n]*\)"),
    re.compile(r"self\.\w+\s*=\s*\w+"),

    # Logging patterns
    re.compile(
        r"(?:logger|logging|log|self\.logger|cls\.logger)\."
        r"(?:debug|info|warning|error|critical|exception)\("
    ),

    # Assertion patterns
    re.compile(r"assert\s+isinstance\([^)\n]+\)"),
    re.compile(r"assert\s+\w+\s+is\s+not\s+None"),

    # Context manager patterns
    re.compile(r"with\s+open\([^)\n]+\)\s+as\s+\w+\s*:"),

    # Common method calls
    re.compile(
        r"\.(?:append|extend|update|pop|remove|insert|clear|copy"
        r"|items|keys|values|get|setdefault"
        r"|join|split|strip|lstrip|rstrip|replace|startswith|endswith"
        r"|format|encode|decode)\("
    ),

    # Docstring section headers (compressible because docstrings are safe)
    re.compile(
        r"(?:Args|Returns|Raises|Yields|Attributes|Examples|Note|Notes"
        r"|References|See Also|Warnings|Todo)\s*:"
    ),
    re.compile(r":param\s+\w+:"),
    re.compile(r":type\s+\w+:"),
    re.compile(r":returns?:"),
    re.compile(r":raises?\s+\w+:"),

    # Pytest patterns
    re.compile(r"pytest\.(?:raises|warns|mark\.\w+|fixture|param|skip|xfail)\("),

    # Class definitions with bases
    re.compile(r"class\s+\w+\([^)\n]{0,60}\)\s*:"),

    # Return patterns
    re.compile(r"return\s+(?:None|True|False|NotImplemented|self|\{\}|\[\]|\(\))"),

    # Typed variable annotations
    re.compile(r"\w+\s*:\s*(?:str|int|float|bool|list|dict|Any|Optional)\s*="),
]

# ---------------------------------------------------------------------------
# Dictionary: seed patterns
# ---------------------------------------------------------------------------

SEED_PATTERNS = [
    # Import patterns (extremely high frequency)
    ("import os", "import"),
    ("import sys", "import"),
    ("import json", "import"),
    ("import logging", "import"),
    ("import re", "import"),
    ("import time", "import"),
    ("import argparse", "import"),
    ("import unittest", "import"),
    ("from typing import", "import"),
    ("from pathlib import Path", "import"),
    ("from collections import", "import"),
    ("from abc import ABC, abstractmethod", "import"),

    # Dunder method signatures (extremely high frequency)
    ("def __init__(self):", "dunder"),
    ("def __repr__(self):", "dunder"),
    ("def __str__(self):", "dunder"),
    ("def __eq__(self, other):", "dunder"),
    ("def __ne__(self, other):", "dunder"),
    ("def __hash__(self):", "dunder"),
    ("def __len__(self):", "dunder"),
    ("def __iter__(self):", "dunder"),
    ("def __next__(self):", "dunder"),
    ("def __enter__(self):", "dunder"),
    ("def __exit__(self, exc_type, exc_val, exc_tb):", "dunder"),
    ("def __bool__(self):", "dunder"),
    ("def __getitem__(self, key):", "dunder"),
    ("def __setitem__(self, key, value):", "dunder"),
    ("def __delitem__(self, key):", "dunder"),
    ("def __contains__(self, item):", "dunder"),
    ("def __call__(self", "dunder"),
    ("def __del__(self):", "dunder"),
    ("def __getattr__(self, name):", "dunder"),
    ("def __setattr__(self, name, value):", "dunder"),

    # Common decorators
    ("@property", "decorator"),
    ("@staticmethod", "decorator"),
    ("@classmethod", "decorator"),
    ("@abstractmethod", "decorator"),
    ("@pytest.fixture", "decorator"),
    ("@pytest.mark.parametrize(", "decorator"),
    ("@pytest.mark.skipif(", "decorator"),
    ("@dataclass", "decorator"),
    ("@override", "decorator"),
    ("@cached_property", "decorator"),

    # Type hint return annotations
    ("-> None:", "typehint"),
    ("-> str:", "typehint"),
    ("-> int:", "typehint"),
    ("-> bool:", "typehint"),
    ("-> float:", "typehint"),
    ("-> list:", "typehint"),
    ("-> dict:", "typehint"),
    ("-> None", "typehint"),
    ("Optional[", "typehint"),

    # Common expressions
    ("if __name__ == \"__main__\":", "expression"),
    ("if __name__ == '__main__':", "expression"),
    ("super().__init__(", "expression"),
    ("raise NotImplementedError(", "expression"),
    ("raise NotImplementedError", "expression"),
    ("raise ValueError(", "expression"),
    ("raise TypeError(", "expression"),
    ("raise KeyError(", "expression"),
    ("raise RuntimeError(", "expression"),
    ("return None", "expression"),
    ("return self", "expression"),
    ("return True", "expression"),
    ("return False", "expression"),
    ("pass", "expression"),

    # Logging boilerplate
    ("logger = logging.getLogger(__name__)", "logging"),
    ("logger.info(", "logging"),
    ("logger.debug(", "logging"),
    ("logger.warning(", "logging"),
    ("logger.error(", "logging"),
    ("logger.exception(", "logging"),

    # Docstring section headers
    ("Args:", "docstring"),
    ("Returns:", "docstring"),
    ("Raises:", "docstring"),
    ("Yields:", "docstring"),
    ("Attributes:", "docstring"),
    ("Examples:", "docstring"),
    (":param ", "docstring"),
    (":returns:", "docstring"),
    (":raises ", "docstring"),
    (":type ", "docstring"),

    # Test patterns (unittest + pytest)
    ("self.assertEqual(", "test"),
    ("self.assertTrue(", "test"),
    ("self.assertFalse(", "test"),
    ("self.assertRaises(", "test"),
    ("self.assertIsNone(", "test"),
    ("self.assertIsNotNone(", "test"),
    ("self.assertIn(", "test"),
    ("self.assertNotIn(", "test"),
    ("self.assertIsInstance(", "test"),
    ("pytest.raises(", "test"),
    ("pytest.mark.parametrize(", "test"),
]

# ---------------------------------------------------------------------------
# AST mining: subtree root types
# ---------------------------------------------------------------------------

SUBTREE_ROOT_TYPES = {
    # Statements
    "expression_statement",
    "return_statement",
    "raise_statement",
    "assert_statement",
    "delete_statement",
    "pass_statement",
    "import_statement",
    "import_from_statement",
    "if_statement",
    "for_statement",
    "while_statement",
    "with_statement",
    # Declarations
    "function_definition",
    "class_definition",
    "decorated_definition",
    # Assignments
    "assignment",
    "augmented_assignment",
    # Expressions (as subtree roots for normalization)
    "call",
    "attribute",
}

# ---------------------------------------------------------------------------
# Template mining: identifier normalization
# ---------------------------------------------------------------------------

# Parent node types where an identifier is structural (defines the pattern)
FIXED_PARENT_TYPES = {
    # Definitions
    "function_definition",
    "class_definition",
    "decorated_definition",
    # Imports
    "import_statement",
    "import_from_statement",
    "dotted_name",
    "aliased_import",
    # Calls & attribute access
    "call",
    "attribute",
    # Type annotations
    "type",
    # Exception clause
    "except_clause",
    # Decorators
    "decorator",
    # Typed parameters
    "typed_parameter",
    "typed_default_parameter",
    # Global/nonlocal
    "global_statement",
    "nonlocal_statement",
}

# Parent node types where an identifier is a user-chosen name (normalizable)
NORMALIZE_PARENT_TYPES = {
    # Assignments
    "assignment",
    "augmented_assignment",
    "pattern_list",
    # Arguments
    "argument_list",
    "keyword_argument",
    # Expressions
    "return_statement",
    "binary_operator",
    "unary_operator",
    "comparison_operator",
    "boolean_operator",
    "not_operator",
    "conditional_expression",
    # Subscript
    "subscript",
    # Literals / containers
    "tuple",
    "list",
    "dictionary",
    "pair",
    "set",
    # Comprehensions
    "list_comprehension",
    "set_comprehension",
    "dictionary_comprehension",
    "generator_expression",
    # Loop / with targets
    "for_statement",
    "as_pattern",
}

# Well-known names that should never be normalized even if parent says so
STRUCTURAL_NAMES = {
    # Python builtins / keywords that tree-sitter may label as identifier
    "self", "cls", "super", "type", "object", "property",
    "None", "True", "False", "NotImplemented", "Ellipsis",

    # Built-in functions
    "print", "len", "range", "enumerate", "zip", "map", "filter",
    "sorted", "reversed", "list", "dict", "set", "tuple", "str",
    "int", "float", "bool", "bytes", "bytearray", "memoryview",
    "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr",
    "callable", "iter", "next", "repr", "hash", "id", "vars", "dir",
    "any", "all", "min", "max", "sum", "abs", "round", "pow", "divmod",
    "input", "open", "format", "chr", "ord", "hex", "oct", "bin",
    "staticmethod", "classmethod", "complex", "frozenset", "slice",

    # Exception types
    "Exception", "BaseException", "ValueError", "TypeError", "KeyError",
    "AttributeError", "RuntimeError", "NotImplementedError", "StopIteration",
    "FileNotFoundError", "ImportError", "ModuleNotFoundError",
    "OSError", "IOError", "IndexError", "NameError", "AssertionError",
    "PermissionError", "TimeoutError", "ConnectionError",
    "OverflowError", "ZeroDivisionError", "RecursionError",
    "SystemExit", "KeyboardInterrupt", "GeneratorExit",
    "UnicodeDecodeError", "UnicodeEncodeError", "UnicodeError",
    "StopAsyncIteration", "BlockingIOError", "BrokenPipeError",

    # Standard library modules (frequently appear in code)
    "os", "sys", "re", "json", "math", "time", "datetime",
    "pathlib", "Path", "logging", "collections", "functools",
    "itertools", "typing", "abc", "io", "copy", "shutil",
    "argparse", "unittest", "pytest", "subprocess", "threading",
    "multiprocessing", "socket", "http", "urllib", "asyncio",
    "warnings", "contextlib", "dataclasses", "enum", "inspect",
    "textwrap", "operator", "struct", "csv", "hashlib", "hmac",
    "base64", "pickle", "tempfile", "glob", "fnmatch",

    # Common framework / library names
    "logger", "app", "db", "request", "response", "session",
    "django", "flask", "fastapi", "sqlalchemy",

    # Common structural method names
    "append", "extend", "update", "pop", "remove", "insert",
    "items", "keys", "values", "get", "setdefault",
    "join", "split", "strip", "replace", "startswith", "endswith",
    "format", "encode", "decode",
    "close", "read", "write", "flush", "seek",
    "setUp", "tearDown", "setUpClass", "tearDownClass",

    # Dunder names
    "__init__", "__repr__", "__str__", "__eq__", "__hash__",
    "__len__", "__iter__", "__next__", "__enter__", "__exit__",
    "__getitem__", "__setitem__", "__delitem__", "__contains__", "__call__",
    "__name__", "__main__", "__file__", "__all__", "__doc__",
    "__class__", "__dict__", "__module__", "__slots__",
    "__new__", "__del__", "__bool__", "__bytes__", "__format__",
    "__getattr__", "__setattr__", "__delattr__",
    "__add__", "__sub__", "__mul__", "__truediv__", "__floordiv__",
    "__mod__", "__pow__", "__neg__", "__pos__", "__abs__",
    "__lt__", "__le__", "__gt__", "__ge__", "__ne__",
}

# ---------------------------------------------------------------------------
# Download: paths to skip when extracting source files
# ---------------------------------------------------------------------------

SKIP_PATH_PATTERNS = [
    "__pycache__/",
    ".pyc",
    "/migrations/",
    "_pb2.py",
    "_pb2_grpc.py",
    "/vendored/",
    "/vendor/",
    "setup.py",
    "conftest.py",
]

# ---------------------------------------------------------------------------
# Data: repos and eval splits
# ---------------------------------------------------------------------------

REPOS = [
    # --- Web frameworks & web apps (8) ---
    ("django", "django"),                     # BSD-3-Clause
    ("pallets", "flask"),                     # BSD-3-Clause
    ("tiangolo", "fastapi"),                  # MIT
    ("encode", "django-rest-framework"),      # BSD-3-Clause
    ("sanic-org", "sanic"),                   # MIT
    ("tornadoweb", "tornado"),               # Apache-2.0
    ("encode", "starlette"),                  # BSD-3-Clause
    ("wagtail", "wagtail"),                   # BSD-3-Clause
    # --- Data science / ML / deep learning (7) ---
    ("pandas-dev", "pandas"),                 # BSD-3-Clause
    ("scikit-learn", "scikit-learn"),          # BSD-3-Clause
    ("huggingface", "transformers"),           # Apache-2.0
    ("keras-team", "keras"),                  # Apache-2.0
    ("google", "jax"),                        # Apache-2.0
    ("dask", "dask"),                         # BSD-3-Clause
    ("mlflow", "mlflow"),                     # Apache-2.0
    # --- CLI tools & system utilities (6) ---
    ("Textualize", "rich"),                   # MIT
    ("pallets", "click"),                     # BSD-3-Clause
    ("tiangolo", "typer"),                    # MIT
    ("Textualize", "textual"),               # MIT
    ("httpie", "cli"),                        # BSD-3-Clause
    ("tqdm", "tqdm"),                         # MIT
    # --- DevOps / infrastructure / cloud (6) ---
    ("apache", "airflow"),                    # Apache-2.0
    ("docker", "docker-py"),                  # Apache-2.0
    ("kubernetes-client", "python"),          # Apache-2.0
    ("boto", "boto3"),                        # Apache-2.0
    ("saltstack", "salt"),                    # Apache-2.0
    ("spotify", "luigi"),                     # Apache-2.0
    # --- Networking / async / protocols (4) ---
    ("aio-libs", "aiohttp"),                  # Apache-2.0
    ("encode", "httpx"),                      # BSD-3-Clause
    ("encode", "uvicorn"),                    # BSD-3-Clause
    ("mitmproxy", "mitmproxy"),              # MIT
    # --- Testing frameworks & tools (3) ---
    ("pytest-dev", "pytest"),                 # MIT
    ("locustio", "locust"),                   # MIT
    ("robotframework", "robotframework"),     # Apache-2.0
    # --- Database / ORM / data storage (3) ---
    ("sqlalchemy", "sqlalchemy"),             # MIT
    ("sqlalchemy", "alembic"),               # MIT
    ("coleifer", "peewee"),                   # MIT
    # --- Security / cryptography (3) ---
    ("pyca", "cryptography"),                # Apache-2.0 + BSD
    ("PyCQA", "bandit"),                     # Apache-2.0
    ("Yelp", "detect-secrets"),              # Apache-2.0
    # --- NLP / text processing (2) ---
    ("explosion", "spaCy"),                   # MIT
    ("nltk", "nltk"),                         # Apache-2.0
    # --- Image / video / audio processing (4) ---
    ("python-pillow", "Pillow"),             # MIT-CMU
    ("scikit-image", "scikit-image"),         # BSD-3-Clause
    ("Zulko", "moviepy"),                    # MIT
    ("librosa", "librosa"),                  # ISC
    # --- Automation / scripting / bots (3) ---
    ("scrapy", "scrapy"),                     # BSD-3-Clause
    ("home-assistant", "core"),              # Apache-2.0
    ("Rapptz", "discord.py"),               # MIT
    # --- Package management / build tools (2) ---
    ("python-poetry", "poetry"),             # MIT
    ("pypa", "pip"),                          # MIT
]

EVAL_REPOS = [
    "pydantic--pydantic",                     # MIT         | Data validation
    "celery--celery",                         # BSD-3-Clause| Task queue
    "gradio-app--gradio",                     # Apache-2.0  | ML UI
    "matplotlib--matplotlib",                 # PSF-like    | Visualization
    "numpy--numpy",                           # BSD-3-Clause| Numerical computing
    "dagster-io--dagster",                    # Apache-2.0  | Data orchestration
    "bokeh--bokeh",                           # BSD-3-Clause| Interactive viz
    "mkdocs--mkdocs",                         # BSD-2-Clause| Documentation
    "kivy--kivy",                             # MIT         | GUI framework
]
