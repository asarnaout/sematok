"""Analyze per-repo distribution of every macro in the current dictionary.

For each macro, counts how many distinct repos it appears in and which ones.
Outputs a threshold analysis showing what would be lost at each --min-repos value.

Usage:
    python -m sematok.repo_distribution --language csharp

This helps choose the right --min-repos threshold for mining.py.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from tqdm import tqdm

from sematok.compressor import Compressor
from sematok.dictionary import CompressionDictionary
from sematok.languages import get_language
from sematok.lexer import get_safe_ranges, set_language


def main():
    parser = argparse.ArgumentParser(description="Analyze per-repo macro distribution")
    parser.add_argument("--language", type=str, default="csharp", help="Language config to use")
    parser.add_argument("--corpus", type=str, default=None, help="Corpus directory (default: data/raw_<lang>)")
    parser.add_argument("--dictionary", type=str, default=None, help="Dictionary JSON path")
    args = parser.parse_args()

    lang = get_language(args.language)
    set_language(lang)
    corpus_dir = Path(args.corpus) if args.corpus else Path(f"data/raw_{lang.name}")
    from sematok.languages import get_dictionary_path
    if args.dictionary:
        dict_path = Path(args.dictionary)
    else:
        resolved = get_dictionary_path(args.language)
        dict_path = resolved if resolved else Path(f"sematok/languages/{args.language}/dictionary.json")

    eval_repos = set(lang.eval_repos)

    # Load dictionary
    d = CompressionDictionary.load(dict_path)
    compressor = Compressor(d, language=lang)

    # Load file->repo map
    meta_path = corpus_dir / "metadata.jsonl"
    file_to_repo = {}
    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            entry = json.loads(line)
            file_to_repo[entry["filename"]] = entry["source"]

    # Only training files (exclude eval repos)
    files = sorted(corpus_dir.glob(f"*{lang.file_extension}"))
    files = [f for f in files if file_to_repo.get(f.name, "unknown") not in eval_repos]
    print(f"Training files: {len(files)}")

    macro_re = re.compile(r"<\|M\d{3}\|>")
    template_re = re.compile(r"<\|T(\d{3}):[^|]*\|>")

    # macro -> set of repos
    macro_repos: dict[str, set[str]] = defaultdict(set)

    for f in tqdm(files, desc="Scanning"):
        repo = file_to_repo.get(f.name, "unknown")
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
            safe_ranges = get_safe_ranges(source, allow_xmldoc=True)
            compressed = compressor.compress(source, safe_ranges=safe_ranges)
        except Exception:
            continue

        seen = set()
        for macro in macro_re.findall(compressed):
            seen.add(macro)
        for match in template_re.finditer(compressed):
            seen.add(f"<|T{match.group(1)}|>")

        for macro in seen:
            macro_repos[macro].add(repo)

    # Build results
    results = []
    for macro, pattern in d.macro_to_pattern.items():
        repos = macro_repos.get(macro, set())
        results.append((macro, pattern, "pattern", len(repos), sorted(repos)))

    for macro, template in d.macro_to_template.items():
        repos = macro_repos.get(macro, set())
        results.append((macro, template, "template", len(repos), sorted(repos)))

    # Sort by repo count ascending (least diverse first)
    results.sort(key=lambda x: (x[3], x[0]))

    # Print threshold analysis
    print("\n" + "=" * 80)
    print("REPO DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Histogram
    repo_count_hist = Counter(r[3] for r in results)
    print(f"\nTotal entries: {len(results)}")
    print(f"\nRepo count distribution:")
    for count in sorted(repo_count_hist.keys()):
        print(f"  {count:3d} repos: {repo_count_hist[count]:4d} entries")

    # Cumulative: what survives at each threshold
    print(f"\nThreshold analysis:")
    print(f"  {'Threshold':<12} {'Surviving':<12} {'Killed':<10} {'% surviving'}")
    print(f"  {'-'*12} {'-'*12} {'-'*10} {'-'*12}")
    total = len(results)
    for threshold in [1, 2, 3, 4, 5, 7, 10, 15, 20]:
        surviving = sum(1 for r in results if r[3] >= threshold)
        killed = total - surviving
        print(f"  >= {threshold:<8} {surviving:<12} {killed:<10} {surviving/total*100:.1f}%")

    # Detail: entries at exactly 1, 2, 3 repos
    for n in [1, 2, 3]:
        entries_at_n = [r for r in results if r[3] == n]
        if entries_at_n:
            print(f"\n{'='*80}")
            print(f"ENTRIES IN EXACTLY {n} REPO{'S' if n > 1 else ''} ({len(entries_at_n)} entries)")
            print(f"{'='*80}")
            for macro, pattern, typ, count, repos in entries_at_n:
                pat_display = pattern[:70] + "..." if len(pattern) > 70 else pattern
                print(f"  {macro:<10} {pat_display:<75} repos: {', '.join(repos)}")


if __name__ == "__main__":
    main()
