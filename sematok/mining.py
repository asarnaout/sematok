"""
Pattern mining: discover frequent multi-token C# boilerplate patterns from a corpus.

Scans C# files, extracts candidate patterns, scores them by
frequency * token_savings, and produces a refined compression dictionary.

Quality filters:
- Must appear in at least 2 repos (no repo-specific patterns)
- Must be at least 8 characters (no short junk like [0], [i])
- Must span at least 3 BPE tokens (not worth compressing otherwise)
- Must appear at least 50 times across the corpus
- No embedded newlines (regex artifacts)
- No pure-logic patterns (only boilerplate/scaffolding)

Usage:
    python -m sematok.mining --corpus data/raw_cs --output sematok/dictionary.json
"""

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import tiktoken
from tqdm import tqdm

from sematok.dictionary import CompressionDictionary, SEED_PATTERNS
from sematok.lexer import get_safe_ranges


# --- Candidate pattern extraction regexes (applied to safe zones only) ---

CANDIDATE_PATTERNS = [
    # Using directives
    re.compile(r"using\s+[\w.]+;"),

    # Attribute patterns -- require 3+ char name to reject [0], [i], [1] etc.
    # Negative lookbehind: reject if preceded by a word char or ')' to avoid
    # matching array indexers like buffer[length] or func()[result].
    re.compile(r"(?<![)\w])\[\w{3,}(?:\([^)\n]*\))?\]"),

    # Property accessors
    re.compile(r"\{\s*get;\s*(?:(?:private|protected|internal)\s+)?set;\s*\}"),
    re.compile(r"\{\s*get;\s*(?:init|internal\s+set|protected\s+set);\s*\}"),
    re.compile(r"\{\s*get;\s*\}"),

    # Access modifier combos (2-4 keywords before a type/name)
    re.compile(
        r"(?:public|private|protected|internal)\s+"
        r"(?:static\s+)?(?:readonly\s+)?"
        r"(?:virtual\s+|override\s+|abstract\s+|sealed\s+|async\s+)?"
        r"(?:partial\s+)?"
        r"(?:class|struct|interface|enum|record|void|string|int|bool|long|double|float|decimal|byte|char|object"
        r"|Task|Task<\w+>|IActionResult|ActionResult)"
    ),

    # Common method signatures (with parameter lists)
    re.compile(
        r"(?:public|private|protected|internal)\s+"
        r"(?:static\s+)?(?:override\s+)?(?:async\s+)?"
        r"(?:void|string|int|bool|Task|Task<\w+>)\s+"
        r"\w+\([^)\n]{0,80}\)"
    ),

    # Throw patterns
    re.compile(r"throw\s+new\s+\w+(?:Exception|Error)\([^)\n]*\);"),

    # Common framework expressions
    re.compile(r"Console\.(?:Write|WriteLine|ReadLine|Read)\("),
    re.compile(r"return\s+Task\.(?:CompletedTask|FromResult|Delay|Run)\b"),
    re.compile(r"=\s*(?:string\.Empty|new\(\)|default!?|Array\.Empty<\w+>\(\));"),
    re.compile(r"Debug\.(?:Assert|WriteLine|Write)\("),
    re.compile(r"ArgumentNullException\.ThrowIfNull\("),

    # XML doc
    re.compile(
        r"///\s*<(?:summary|/summary|param\s+name=\"[^\"\n]*\"|returns|/returns"
        r"|exception\s+cref=\"[^\"\n]*\"|remarks|/remarks|value|/value"
        r"|inheritdoc\s*/|see\s+cref=\"[^\"\n]*\"\s*/?)>"
    ),

    # Generic type patterns
    re.compile(
        r"(?:IEnumerable|IList|ICollection|IDictionary|IReadOnlyList|IReadOnlyCollection"
        r"|IReadOnlyDictionary|Dictionary|List|HashSet|SortedSet|Queue|Stack"
        r"|ConcurrentDictionary|Task|ValueTask|Func|Action|Lazy"
        r"|ILogger|IOptions|IServiceProvider|IConfiguration"
        r"|ReadOnlySpan|Span|Memory|ReadOnlyMemory"
        r"|Nullable|KeyValuePair)<"
    ),

    # Namespace/class scaffolding
    re.compile(r"namespace\s+[\w.]+"),

    # Common attributes (multi-word, framework-specific)
    re.compile(r"\[(?:MethodImpl|DllImport|MarshalAs|StructLayout|FieldOffset)\([^)\n]+\)\]"),
    re.compile(r"\[(?:Conditional|Obsolete|Description|Category|DefaultValue)\([^)\n]*\)\]"),
    re.compile(
        r"\[(?:Theory|Fact|InlineData|MemberData|ClassData"
        r"|TestMethod|TestClass|TestCategory"
        r"|ConditionalFact|ConditionalTheory"
        r"|ApiController|Route|HttpGet|HttpPost|HttpPut|HttpDelete|HttpPatch"
        r"|Authorize|AllowAnonymous"
        r"|Required|StringLength|Range|MaxLength|MinLength"
        r"|JsonProperty|JsonPropertyName|JsonIgnore"
        r"|Serializable|Flags|Browsable)\b[^]\n]*\]"
    ),

    # Interface implementation declarations
    re.compile(r":\s*(?:IDisposable|IAsyncDisposable|IEquatable<\w+>|IComparable<\w+>|ICloneable|IEnumerable<\w+>)"),

    # Common parameter patterns (boilerplate, not logic)
    re.compile(r"CancellationToken\s+cancellationToken"),
    re.compile(r"IServiceProvider\s+serviceProvider"),
    re.compile(r"ILogger<\w+>\s+logger"),

    # Async/dispose boilerplate
    re.compile(r"ConfigureAwait\(false\)"),
    re.compile(r"\.GetAwaiter\(\)\.GetResult\(\)"),
    re.compile(r"async\s+ValueTask"),
    re.compile(r"async\s+Task<\w+>"),

    # Assertion patterns (test boilerplate)
    re.compile(r"Assert\.(?:Equal|NotEqual|True|False|Null|NotNull|Throws|Contains|Empty|Same|NotSame"
               r"|IsType|IsAssignableFrom|InRange|Collection|Single)\b"),

    # Common pragma/preprocessor
    re.compile(r"#pragma\s+warning\s+(?:disable|restore)\s+[\w,\s]+"),
    re.compile(r"#if\s+!?(?:NET\w*|NETCOREAPP|NETSTANDARD|DEBUG|RELEASE|WINDOWS)"),

    # Null-checking boilerplate
    re.compile(r"\?\?\s*throw\s+new\s+\w+Exception\("),
    re.compile(r"is\s+(?:not\s+)?null"),

    # Generic constraints
    re.compile(r"where\s+\w+\s*:\s*(?:class|struct|notnull|new\(\)|unmanaged)"),
    re.compile(r"where\s+\w+\s*:\s*(?:IComparable|IEquatable|IEnumerable|IDisposable|ICloneable)<\w+>"),

    # String validation methods
    re.compile(r"string\.IsNullOr(?:Empty|WhiteSpace)\("),

    # LINQ terminal methods
    re.compile(r"\.(?:ToList|ToArray|ToDictionary|ToHashSet|FirstOrDefault|SingleOrDefault|LastOrDefault|First|Single|Last|Count|Any|All)\("),

    # Common method calls (zero-arg, pure boilerplate)
    re.compile(r"\.(?:ToString|GetType|GetHashCode|Equals|Dispose|GetAwaiter)\(\)"),
]


# --- Quality filter thresholds ---

# Minimum BPE tokens a pattern must span to be worth compressing
MIN_TOKEN_SPAN = 3
# Minimum frequency across corpus to consider a pattern
MIN_FREQUENCY = 50
# Minimum character length for a pattern
MIN_CHAR_LENGTH = 8
# Minimum number of distinct repos a pattern must appear in
MIN_REPOS = 2


def _get_bpe_token_count(text: str, enc: tiktoken.Encoding) -> int:
    """Count how many BPE tokens a text requires."""
    return len(enc.encode(text))


def _load_file_repo_map(corpus_dir: Path) -> dict[str, str]:
    """Load metadata.jsonl to map filename -> repo source."""
    meta_path = corpus_dir / "metadata.jsonl"
    file_to_repo = {}
    if meta_path.exists():
        with open(meta_path, encoding="utf-8") as f:
            for line in f:
                entry = json.loads(line)
                file_to_repo[entry["filename"]] = entry["source"]
    return file_to_repo


def _is_valid_pattern(pattern: str) -> bool:
    """Quick quality check before counting a candidate."""
    # No embedded newlines
    if "\n" in pattern or "\r" in pattern:
        return False
    # Minimum length
    if len(pattern) < MIN_CHAR_LENGTH:
        return False
    # Not pure whitespace
    if not pattern.strip():
        return False
    # Not a single word (too generic)
    if re.fullmatch(r"\w+", pattern):
        return False
    return True


def extract_candidates_from_file(source: str) -> list[str]:
    """Extract candidate boilerplate patterns from a C# source file."""
    try:
        safe_ranges = get_safe_ranges(source)
    except Exception:
        safe_ranges = [(0, len(source.encode("utf-8")))]

    source_bytes = source.encode("utf-8")
    safe_text_parts = []
    for start, end in safe_ranges:
        safe_text_parts.append(source_bytes[start:end].decode("utf-8", errors="replace"))
    safe_text = "\n".join(safe_text_parts)

    candidates = []
    for pattern_re in CANDIDATE_PATTERNS:
        for match in pattern_re.finditer(safe_text):
            candidate = match.group(0).strip()
            if candidate and _is_valid_pattern(candidate):
                candidates.append(candidate)

    return candidates


def mine_patterns(
    corpus_dir: Path,
    top_n: int = 1000,
    min_frequency: int = MIN_FREQUENCY,
    min_token_span: int = MIN_TOKEN_SPAN,
    min_repos: int = MIN_REPOS,
    max_files: int | None = None,
    exclude_repos: list[str] | None = None,
) -> list[tuple[str, int, int, float, int]]:
    """
    Mine frequent boilerplate patterns from a corpus of C# files.

    Args:
        exclude_repos: Repo names to skip (e.g. held-out eval repos).
            This ensures the dictionary is built only from training data.

    Returns:
        List of (pattern, frequency, token_count, score, repo_count) tuples
        sorted by score descending.
        Score = frequency * (token_count - 1), representing total tokens saved.
    """
    enc = tiktoken.get_encoding("gpt2")

    # Load repo mapping for cross-repo validation
    file_to_repo = _load_file_repo_map(corpus_dir)
    exclude_set = set(exclude_repos) if exclude_repos else set()

    # Track global frequency and per-repo presence
    pattern_counter: Counter = Counter()
    pattern_repos: dict[str, set[str]] = defaultdict(set)

    cs_files = sorted(corpus_dir.glob("*.cs"))

    # Exclude eval repo files from mining
    if exclude_set and file_to_repo:
        before = len(cs_files)
        cs_files = [f for f in cs_files if file_to_repo.get(f.name, "unknown") not in exclude_set]
        print(f"Excluded {before - len(cs_files)} files from {len(exclude_set)} eval repos")

    if max_files:
        cs_files = cs_files[:max_files]

    print(f"Mining patterns from {len(cs_files)} files...")
    for f in tqdm(cs_files, desc="Scanning"):
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        repo = file_to_repo.get(f.name, "unknown")
        candidates = extract_candidates_from_file(source)

        # Deduplicate per file: count each pattern at most once per file
        unique_candidates = set(candidates)
        pattern_counter.update(unique_candidates)
        for c in unique_candidates:
            pattern_repos[c].add(repo)

    # Score and filter patterns
    scored = []
    rejected = {"low_freq": 0, "few_repos": 0, "short_span": 0}

    for pattern, freq in pattern_counter.items():
        if freq < min_frequency:
            rejected["low_freq"] += 1
            continue
        repo_count = len(pattern_repos[pattern])
        if repo_count < min_repos:
            rejected["few_repos"] += 1
            continue
        token_count = _get_bpe_token_count(pattern, enc)
        if token_count < min_token_span:
            rejected["short_span"] += 1
            continue
        score = freq * (token_count - 1)
        scored.append((pattern, freq, token_count, score, repo_count))

    scored.sort(key=lambda x: x[3], reverse=True)

    print(f"\nCandidate patterns found: {len(pattern_counter)}")
    print(f"After filtering: {len(scored)} passed, rejected: {rejected}")

    return scored[:top_n]


def build_mined_dictionary(
    corpus_dir: Path,
    top_n: int = 9999,
    include_seeds: bool = True,
    min_repos: int = MIN_REPOS,
    max_files: int | None = None,
    exclude_repos: list[str] | None = None,
) -> tuple[CompressionDictionary, list[tuple[str, int, int, float, int]]]:
    """
    Build a compression dictionary by combining seed patterns with mined patterns.

    Seed patterns are always included first. Mined patterns fill the remaining
    slots up to top_n total patterns.

    Returns:
        (dictionary, mined_patterns) -- the dictionary and the raw mined results
        for downstream display/analysis.
    """
    if include_seeds:
        d = CompressionDictionary.from_seed()
        remaining = top_n - d.size
    else:
        d = CompressionDictionary()
        remaining = top_n

    mined = []
    if remaining > 0:
        mined = mine_patterns(
            corpus_dir, top_n=remaining * 2,
            min_repos=min_repos, max_files=max_files,
            exclude_repos=exclude_repos,
        )
        added = 0
        for pattern, freq, tok_count, score, repo_count in mined:
            if pattern in d.pattern_to_macro:
                continue
            d.add_pattern(pattern, category="mined")
            added += 1
            if added >= remaining:
                break
        print(f"Added {added} mined patterns (total: {d.size})")

    return d, mined


def main():
    parser = argparse.ArgumentParser(description="Mine C# boilerplate patterns")
    parser.add_argument("--corpus", type=str, required=True, help="Directory with .cs files")
    parser.add_argument("--output", type=str, default="sematok/dictionary.json")
    parser.add_argument("--top", type=int, default=9999, help="Max patterns in dictionary (actual count depends on quality filters)")
    parser.add_argument("--min-repos", type=int, default=MIN_REPOS, help="Min repos a pattern must appear in")
    parser.add_argument("--min-freq", type=int, default=MIN_FREQUENCY, help="Min frequency across corpus")
    parser.add_argument("--max-files", type=int, default=None)
    parser.add_argument("--no-seeds", action="store_true", help="Don't include seed patterns")
    parser.add_argument(
        "--exclude-repos", type=str, nargs="+", default=None,
        help="Repos to exclude from mining (e.g. held-out eval repos)",
    )
    parser.add_argument("--verbose", action="store_true", help="Print all accepted patterns")
    args = parser.parse_args()

    d, mined = build_mined_dictionary(
        Path(args.corpus),
        top_n=args.top,
        include_seeds=not args.no_seeds,
        min_repos=args.min_repos,
        max_files=args.max_files,
        exclude_repos=args.exclude_repos,
    )

    d.save(args.output)
    print(f"\nDictionary saved to {args.output}")
    print(f"Stats: {d.stats()}")

    # Print top patterns (reuse results from build, no re-scan)
    print(f"\nTop 30 mined patterns by score:")
    for pattern, freq, tok_count, score, repo_count in mined[:30]:
        in_dict = "+" if pattern in d.pattern_to_macro else " "
        print(
            f"  [{in_dict}] freq={freq:5d} toks={tok_count:2d} "
            f"score={score:7d} repos={repo_count} | {pattern!r}"
        )


if __name__ == "__main__":
    main()
