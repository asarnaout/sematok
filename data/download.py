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
    subprocess.run(
        ["git", "clone", "--depth", "1", url, str(repo_dir)],
        check=True,
        capture_output=True,
    )
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
) -> int:
    """
    Walk all cloned repos and copy source files to the output directory.

    Filters by length, skips auto-generated files, and renames to sequential numbering.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    meta_path = output_dir / "metadata.jsonl"

    count = 0
    skipped = 0

    with open(meta_path, "w", encoding="utf-8") as meta_f:
        for repo_dir in sorted(repos_dir.iterdir()):
            if not repo_dir.is_dir():
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
    for org, name in lang.repos:
        try:
            clone_repo(org, name, repos_dir)
        except subprocess.CalledProcessError as e:
            print(f"  {org}/{name}: FAILED to clone ({e})")
            continue

    # Step 2: Extract source files
    extensions = lang.source_extensions or [lang.file_extension]
    ext_label = "/".join(extensions)
    print(f"\nExtracting {ext_label} files...")
    count = extract_source_files(
        repos_dir,
        Path(output_path),
        file_extension=lang.file_extension,
        source_extensions=lang.source_extensions or None,
        min_length=args.min_length,
        max_length=args.max_length,
        max_files=args.max_files,
        skip_path_patterns=lang.skip_path_patterns,
    )

    # Step 3: Summary
    output_dir = Path(output_path)
    total_size = sum(f.stat().st_size for f in output_dir.glob(f"*{lang.file_extension}"))
    print(f"\nDone. {count} {lang.name} files ({total_size / 1024 / 1024:.1f} MB) saved to {output_dir}")
    print("License: permissive (MIT, Apache-2.0, or BSD-3-Clause)")


if __name__ == "__main__":
    main()
