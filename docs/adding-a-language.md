# Adding a Language

1. Copy the template to a new package:
   ```bash
   cp sematok/languages/TEMPLATE.py sematok/languages/yourlang/__init__.py
   ```

2. Fill in the `LanguageConfig` fields. See `sematok/languages/csharp/__init__.py` for a complete example. You need:
   - A tree-sitter grammar (`pip install tree-sitter-<grammar>`)
   - Unsafe node types (comments, strings) for safe zone detection
   - Candidate patterns (regexes matching boilerplate in your language)
   - AST node types for subtree mining
   - Identifier normalization rules for template discovery
   - A list of repos to mine from

3. Register the language in `sematok/languages/__init__.py`:
   ```python
   _REGISTRY["yourlang"] = "sematok.languages.yourlang"
   ```

4. Mine a dictionary, prepare data, and train -- all pipeline commands accept `--language yourlang`.
