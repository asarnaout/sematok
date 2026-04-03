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

This matters because boilerplate patterns can appear inside strings without being boilerplate:

```csharp
var example = "{ get; set; }";   // without safe zones, this becomes: var example = "<|M004|>";
```

All four mining channels operate exclusively within safe zones. No pattern will ever be discovered in or applied to a string literal or comment.

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

Example -- given this line in a source file:

```
using System.Collections.Generic;
```

Pass 1 extracts 8-grams at word boundaries:

```
using Sy  ← position 0 (start of line = word boundary)
System.C  ← position 6 (prev char is space)
Collecti  ← position 13 (prev char is '.')
Generic;  ← position 25 (prev char is '.')
```

Suppose `using Sy` appears in 60,000+ files and survives the threshold. In pass 2, the pipeline revisits every position where `using Sy` starts and tries extending:

```
using Sy          (8 chars)
using Sys         (9)
...
using System      (12, word boundary ✓)
using System.     (13, word boundary ✓)
...
using System.Collections.Generic;   (33, word boundary ✓)
```

Each extension that lands on a word boundary is counted as a candidate. After pass 2, `using System.Collections.Generic;` has a file count and repo count, and enters the scoring pipeline like any other candidate.

After pass 2, quality filters are applied:
- Must not be a single identifier or pure whitespace
- Must contain at least one punctuation/operator character
- No more than 50% whitespace
- Minimum frequency, repo diversity, and BPE token span

The regex and n-gram results are merged (union, deduplicated) before being added to the dictionary.

## Channel 4: AST Subtree Mining (Templates)

This channel discovers **parameterized patterns** -- boilerplate where the structure is fixed but identifiers vary. For example, these three lines are all instances of the same pattern:

```csharp
this.bar = bar;
this.logger = logger;
this.name = name;
```

Here's how the AST channel discovers the template that captures all of them.

### Step 1: Parse and walk the tree

Tree-sitter parses `this.bar = bar;` into an AST:

```
expression_statement                  ← configured as a subtree root type
└── assignment_expression
    ├── member_access_expression
    │   ├── this
    │   └── identifier: "bar"         ← parent is member_access
    └── identifier: "bar"             ← parent is assignment
```

The pipeline walks the full AST looking for nodes at configured root types (e.g., `expression_statement`, `return_statement`). When it finds one, it extracts the subtree.

### Step 2: Decide which identifiers to normalize

Not every identifier should become a slot. Tree-sitter tells us the **role** of each identifier through its parent node type. The language config classifies parent types into two categories:

- **Fixed** -- the identifier defines the pattern (type names, method names). Keep as literal text.
- **Normalize** -- the identifier is a user-chosen name (variable assignments, arguments, field accesses). Replace with `{0}`, `{1}`, etc.

Well-known names like `string`, `int`, `var`, `Task` are never normalized regardless of context.

Applied to our example:

```
identifier: "bar"   parent: member_access   → normalize  → {0}
identifier: "bar"   parent: assignment       → normalize  → {0} (same text = same slot)
```

### Step 3: Build the template

Replace normalized identifiers with their slots:

```
this.bar = bar;      → this.{0} = {0};
this.logger = logger;  → this.{0} = {0};
this.name = name;      → this.{0} = {0};
```

All three lines collapse to the same template: `this.{0} = {0};`. The dictionary stores this template once, assigned a macro ID like `<|T00001|>`.

### What the model sees

At compression time, identifiers are captured as arguments:

```
this.bar = bar;       → <|T00001:bar|>
this.logger = logger; → <|T00001:logger|>
this.name = name;     → <|T00001:name|>
```

The tokenizer expansion step (`expand_tokenizer.py`) registers two kinds of tokens for templates: a **prefix** (`<|T00001:`) and a shared **closer** (`|>`). The argument between them is tokenized as regular BPE. So the model sees:

```
tokens:  <|T00001:  logger  |>
         ─────────  ──────  ──
         1 token    1 token  1 token
```

Instead of tokenizing `this.logger = logger;` as 5 BPE tokens, the compressed form costs 3 tokens -- a saving of 2 tokens per occurrence. At decompression, the argument is plugged back into the `{0}` slots to recover the original line exactly.

### Step 4: Count and filter

Templates are counted across the corpus (deduplicated per file, tracked per repo) and filtered by the same thresholds as other channels: minimum frequency, repo diversity, and BPE token span.

## Scoring Phase

After all four channels have contributed candidates, the pipeline enters the scoring phase. This is the most computationally expensive step.

For every file in the corpus:

1. **Compress the file** using the full candidate dictionary
2. **For each macro that appears** in the compressed output, calculate the BPE token savings:
   - Exact macros: `tokens_in_original_pattern - 1` (the macro itself is 1 token)
   - Template macros: `tokens_in_expanded_text - (2 + tokens_in_args)` (prefix + args + closer cost tokens)
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
