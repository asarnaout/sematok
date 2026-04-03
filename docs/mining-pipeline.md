# Mining Pipeline

This document explains how sematok discovers boilerplate patterns from a corpus of source files and builds a compression dictionary.

## Overview

The mining pipeline has two phases:

1. **Candidate extraction** -- four independent channels generate candidate patterns from the corpus
2. **Corpus scoring and filtering** -- every candidate is scored by actual BPE token savings, then filtered by quality thresholds

The four extraction channels are:

| Channel | Discovers | Output |
|---------|-----------|--------|
| Seed patterns | Hand-picked, high-confidence patterns | Exact macros |
| Regex candidates | Patterns matching language-specific regexes | Exact macros |
| N-gram frequency | Recurring substrings with no regex needed | Exact macros |
| AST subtree mining | Structural patterns with identifier normalization | Template macros |

There is also a regex-guided template channel, but in practice the AST channel dominates template discovery. Both channels use the same identifier normalization logic.

## Safe Zones

Before any mining happens, every source file is parsed with tree-sitter. The parser identifies **unsafe regions** -- string literals, comments, and character literals -- where compression would be destructive. Everything outside those regions is a **safe zone**.

All four mining channels operate exclusively within safe zones. This means no pattern will ever match inside a string literal or comment (unless the language config explicitly marks certain comment types as safe, like C# XML doc comments `///`).

## Channel 1: Seed Patterns

Each language config defines a list of hand-picked `(pattern, category)` pairs. These are high-confidence boilerplate strings that the language author knows will be frequent and valuable. For example:

```
("{ get; set; }", "property")
("/// <summary>", "xmldoc")
("[Fact]", "attribute")
```

Seeds are added to the dictionary before any automated mining begins. They provide a baseline of known-good patterns and ensure important patterns aren't missed if they fall just below frequency thresholds.

## Channel 2: Regex Candidates

Each language config defines a list of compiled regex patterns (`candidate_patterns`) that match structural boilerplate in that language. Examples include patterns for `using` directives, access modifiers, attribute annotations, generic type declarations, and common expressions.

The regex channel works by:

1. Extracting safe zone text from each file
2. Running every candidate regex against the safe text
3. Collecting matches and counting frequency (deduplicated per file)
4. Filtering: minimum frequency, minimum repo diversity, minimum BPE token span
5. Scoring: `frequency * (token_count - 1)` -- how many tokens this pattern saves across the corpus

The regex channel is fast and precise but limited to what the language author anticipated. It cannot discover patterns that no regex was written for.

## Channel 3: N-gram Frequency

The n-gram channel discovers recurring substrings without any hand-crafted regexes. It finds patterns that nobody anticipated by brute-force counting.

It uses a two-pass Apriori-pruned approach to stay memory-efficient:

**Pass 1: 8-gram census.** Extract every 8-character substring at word boundaries from safe zones across the entire corpus. Count how many files each 8-gram appears in. Prune periodically to control memory. Keep only 8-grams appearing in at least 15 files.

**Pass 2: Extension.** Revisit the corpus. At every position where a surviving 8-gram starts, try extending it to longer lengths (up to 120 characters), stopping at word boundaries. Count the resulting full-length patterns.

After pass 2, quality filters are applied:
- Must not be a single identifier or pure whitespace
- Must contain at least one punctuation/operator character
- No more than 50% whitespace
- Minimum frequency, repo diversity, and BPE token span

The regex and n-gram results are merged (union, deduplicated) before being added to the dictionary.

## Channel 4: AST Subtree Mining (Templates)

This is the most powerful channel. It discovers **parameterized patterns** -- boilerplate where the structure is fixed but identifiers vary.

For example, these three lines are all instances of the same pattern:

```csharp
this._logger = logger;
this._options = options;
this._context = context;
```

The AST channel discovers the template `this.{0} = {0};` that captures all of them.

### How it works

1. **Parse every file** with tree-sitter to get a full AST
2. **Walk the tree** looking for nodes at configured root types (e.g., `expression_statement`, `return_statement`, `local_declaration_statement`). These are the kinds of AST nodes that tend to contain boilerplate.
3. **For each candidate subtree**, check:
   - Is it within a safe zone?
   - Is the source text between 8 and 200 characters?
   - Is the tree depth between 2 and 6 levels? (Too shallow = trivial, too deep = overly specific)
4. **Normalize identifiers** within the subtree. This is the core insight: tree-sitter tells us the role of each identifier via its parent node type.

### Identifier normalization

Not all identifiers should become slots. The language config defines three categories:

- **Fixed parent types** -- the identifier defines the pattern's structure (e.g., class names, method names, type names). These stay as literal text.
- **Normalize parent types** -- the identifier is a user-chosen name (e.g., variable assignments, arguments, field accesses). These become `{0}`, `{1}`, etc.
- **Structural names** -- well-known identifiers that should never be normalized regardless of context (e.g., `string`, `int`, `var`, `null`, `Task`).

When the same identifier appears multiple times in a subtree, it gets the same slot index. This enables patterns like `this.{0} = {0};` where the field name and the parameter name must match.

5. **Count frequency** across the corpus (deduplicated per file, tracked per repo)
6. **Filter and score** -- same thresholds as other channels: minimum frequency, repo diversity, token span

### Why AST over regex for templates?

Regex-guided template mining also exists (it runs candidate regexes and then normalizes the matches). But it can only find templates within patterns that a regex already matches. The AST channel needs no regexes -- it walks the tree structurally, so it discovers patterns that no human anticipated.

In practice, the AST channel produces the vast majority of templates. For C#, the regex template channel found 7 templates from 1,702 candidates while the AST channel found 172 in the final dictionary.

## Scoring Phase

After all four channels have contributed candidates, the pipeline enters the scoring phase. This is the most computationally expensive step.

For every file in the corpus:

1. **Compress the file** using the full candidate dictionary
2. **For each macro that appears** in the compressed output, calculate the BPE token savings:
   - Exact macros: `tokens_in_original_pattern - 1` (the macro itself is 1 token)
   - Template macros: `tokens_in_expanded_text - (1 + tokens_in_args)` (the macro plus its arguments cost tokens too)
3. **Track per-repo savings** so that a single large repo doesn't dominate the scores

The final score for each entry is **repo-weighted**: each repo contributes its per-file-average savings equally, regardless of how many files it has. This prevents a massive repo from inflating the score of repo-specific patterns.

Per-entry metadata is also tracked:
- **file_count** -- number of files where the pattern appeared
- **repo_count** -- number of distinct repos where the pattern appeared

## Filtering

After scoring, entries are filtered by:

1. **`--min-repos`** (default 2 in `--auto` mode) -- patterns must appear in at least N distinct repos. This eliminates project-specific boilerplate.
2. **`--min-files`** (auto-selected in `--auto` mode) -- patterns must appear in at least N files. Auto-selection picks the highest threshold from [0, 10, 50, 100, 200, 500, 1000, 2000, 5000] that retains at least 90% of total compression impact.
3. **`--max-entries`** (default 0 = no cap) -- if set, keeps only the top N entries by score.

Surviving entries are assigned fresh sequential macro IDs and written to the dictionary.

## Scores Sidecar

When using `--auto`, the scoring data is saved to a sidecar JSON file alongside the dictionary (`dictionary_scores.json`). This file contains the score, file count, repo count, content, and category for every scored entry -- not just the survivors.

This enables `--refilter`: you can adjust `--min-files` or `--max-entries` and rebuild the dictionary instantly from the saved scores without re-running the expensive mining and scoring phases.
