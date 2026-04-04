"""
Download source files from permissively-licensed GitHub repositories.

Shallow-clones repos into data/repos/, then extracts source files
into data/raw_<lang>/ for training.

Usage:
    python -m data.download --language csharp
    python -m data.download --language python --max-files 20000
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from tqdm import tqdm

from sematok.languages import get_language


def clone_repo(org: str, name: str, repos_dir: Path) -> Path:
    """Shallow-clone a GitHub repo. Returns the repo directory path."""
    repo_dir = repos_dir / f"{org}--{name}"
    if repo_dir.exists():
        print(f"  {org}/{name}: already cloned, skipping")
        return repo_dir

    url = f"https://github.com/{org}/{name}.git"
    print(f"  {org}/{name}: cloning (shallow)...")
    try:
        subprocess.run(
            ["git", "clone", "--depth", "1", url, str(repo_dir)],
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        # Remove partial clone so retries actually retry
        if repo_dir.exists():
            shutil.rmtree(repo_dir, ignore_errors=True)
        raise
    return repo_dir


def extract_source_files(
    repos_dir: Path,
    output_dir: Path,
    file_extension: str = ".py",
    source_extensions: list[str] | None = None,
    min_length: int = 100,
    max_length: int = 50000,
    max_files: int | None = None,
    skip_path_patterns: list[str] | None = None,
    designated_repos: set[str] | None = None,
) -> int:
    """
    Walk cloned repos and copy source files to the output directory.

    Only processes repos in *designated_repos* (``"org--name"`` format).
    Filters by length, skips auto-generated files, and renames to sequential numbering.
    """
    # Clear any previous extraction to prevent stale files
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.jsonl"

    count = 0
    skipped = 0

    with open(meta_path, "w", encoding="utf-8") as meta_f:
        for repo_dir in sorted(repos_dir.iterdir()):
            if not repo_dir.is_dir():
                continue

            # Only extract from repos designated for this language
            if designated_repos is not None and repo_dir.name not in designated_repos:
                continue

            repo_name = repo_dir.name
            extensions = source_extensions or [file_extension]
            src_files = []
            for ext in extensions:
                src_files.extend(repo_dir.rglob(f"*{ext}"))
            src_files.sort()  # deterministic order across extensions
            ext_label = "/".join(extensions)
            print(f"  {repo_name}: {len(src_files)} {ext_label} files found")

            skip_list = skip_path_patterns or []
            for src_file in tqdm(src_files, desc=f"  {repo_name}", leave=False):
                # Skip auto-generated / junk files (language-specific)
                if skip_list and any(
                    skip in str(src_file).lower()
                    for skip in skip_list
                ):
                    skipped += 1
                    continue

                try:
                    content = src_file.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    skipped += 1
                    continue

                # Length filter
                if len(content) < min_length or len(content) > max_length:
                    skipped += 1
                    continue

                # Save
                out_path = output_dir / f"{count:06d}{file_extension}"
                out_path.write_text(content, encoding="utf-8")

                meta = {
                    "index": count,
                    "filename": out_path.name,
                    "original_path": str(src_file.relative_to(repos_dir)),
                    "original_size": len(content),
                    "source": repo_name,
                    "license": "permissive",  # MIT, Apache-2.0, or BSD-3-Clause
                }
                meta_f.write(json.dumps(meta) + "\n")

                count += 1
                if max_files and count >= max_files:
                    print(f"\n  Reached max_files limit ({max_files})")
                    return count

    print(f"\nExtracted {count} files ({skipped} skipped)")
    return count


def main():
    parser = argparse.ArgumentParser(description="Download training data from MIT repos")
    parser.add_argument("--output", type=str, default=None, help="Output directory (default: data/raw_<lang>)")
    parser.add_argument("--repos-dir", type=str, default="data/repos", help="Where to clone repos")
    parser.add_argument("--max-files", type=int, default=None, help="Max files to extract")
    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=50000)
    parser.add_argument("--language", type=str, required=True, help="Language config to use (e.g. csharp, python)")
    args = parser.parse_args()

    lang = get_language(args.language)
    output_path = args.output or f"data/raw_{lang.name}"
    repos_dir = Path(args.repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Clone repos
    print("Cloning repositories...")
    failed_repos = []
    for org, name in lang.repos:
        try:
            clone_repo(org, name, repos_dir)
        except subprocess.CalledProcessError as e:
            print(f"  {org}/{name}: FAILED to clone ({e})")
            failed_repos.append(f"{org}/{name}")
            continue

    # Step 1.5: Verify all designated repos were cloned
    designated = {f"{org}--{name}" for org, name in lang.repos}
    missing = [d for d in sorted(designated) if not (repos_dir / d).exists()]
    if missing:
        print(f"\nERROR: {len(missing)} designated repos are missing from {repos_dir}:")
        for m in missing:
            print(f"  - {m}")
        print("Aborting extraction. Fix clone failures and re-run.")
        sys.exit(1)

    if failed_repos:
        # All repos exist (maybe from a previous run) but some failed this time
        print(f"\nWARNING: {len(failed_repos)} repos failed to clone this run "
              f"(using previously cloned copies): {', '.join(failed_repos)}")

    # Step 2: Extract source files (only from this language's repos)
    extensions = lang.source_extensions or [lang.file_extension]
    ext_label = "/".join(extensions)
    print(f"\nExtracting {ext_label} files from {len(designated)} designated repos...")
    count = extract_source_files(
        repos_dir,
        Path(output_path),
        file_extension=lang.file_extension,
        source_extensions=lang.source_extensions or None,
        min_length=args.min_length,
        max_length=args.max_length,
        max_files=args.max_files,
        skip_path_patterns=lang.skip_path_patterns,
        designated_repos=designated,
    )

    # Step 3: Verify extraction integrity
    output_dir = Path(output_path)
    meta_path = output_dir / "metadata.jsonl"
    file_count = len(list(output_dir.glob(f"*{lang.file_extension}")))
    meta_count = sum(1 for _ in open(meta_path, encoding="utf-8")) if meta_path.exists() else 0
    if file_count != meta_count:
        print(f"\nERROR: Integrity mismatch! {file_count} source files but {meta_count} metadata entries.")
        sys.exit(1)
    if file_count != count:
        print(f"\nERROR: Integrity mismatch! extract returned {count} but {file_count} files on disk.")
        sys.exit(1)

    # Step 4: Summary
    total_size = sum(f.stat().st_size for f in output_dir.glob(f"*{lang.file_extension}"))
    print(f"\nDone. {count} {lang.name} files ({total_size / 1024 / 1024:.1f} MB) saved to {output_dir}")
    print("License: permissive (MIT, Apache-2.0, or BSD-3-Clause)")


if __name__ == "__main__":
    main()
