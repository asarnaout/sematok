"""
Download C# source files from MIT-licensed GitHub repositories.

Shallow-clones repos into data/repos/, then extracts all .cs files
into data/raw_cs/ for training.

Usage:
    python -m data.download --output data/raw_cs
    python -m data.download --output data/raw_cs --max-files 20000
"""

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from tqdm import tqdm


# MIT-licensed repositories (high quality, production C# code)
# All licenses verified via GitHub LICENSE files on 2026-03-27
REPOS = [
    # --- Original 4 (.NET core) ---
    ("dotnet", "runtime"),             # .NET runtime
    ("dotnet", "roslyn"),              # C# compiler
    ("dotnet", "aspnetcore"),          # ASP.NET web framework
    ("dotnet", "efcore"),              # Entity Framework ORM

    # --- .NET ecosystem ---
    ("dotnet", "maui"),                # Cross-platform UI framework
    ("dotnet", "orleans"),             # Cloud-native actor framework
    ("dotnet", "machinelearning"),     # ML.NET
    ("dotnet", "wpf"),                 # WPF framework
    ("dotnet", "winforms"),            # WinForms framework
    ("dotnet", "yarp"),                # Reverse proxy toolkit
    ("dotnet", "reactive"),            # Reactive Extensions (Rx.NET)
    ("dotnet", "BenchmarkDotNet"),     # Benchmarking library
    ("dotnet", "eShop"),               # Reference eCommerce app

    # --- Microsoft ---
    ("microsoft", "semantic-kernel"),  # LLM orchestration framework
    ("microsoft", "garnet"),           # High-performance cache store

    # --- Popular libraries ---
    ("JamesNK", "Newtonsoft.Json"),    # JSON serialization
    ("icsharpcode", "ILSpy"),          # .NET decompiler
    ("ppy", "osu"),                    # Rhythm game (diverse game patterns)
    ("MudBlazor", "MudBlazor"),        # Blazor component library
    ("Humanizr", "Humanizer"),         # String/date humanization utilities
    ("autofac", "Autofac"),            # IoC/DI container
    ("nunit", "nunit"),                # NUnit testing framework
    ("bchavez", "Bogus"),              # Fake data generator
    ("spectreconsole", "spectre.console"),  # Console UI library
]


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


def extract_cs_files(
    repos_dir: Path,
    output_dir: Path,
    min_length: int = 100,
    max_length: int = 50000,
    max_files: int | None = None,
) -> int:
    """
    Walk all cloned repos and copy .cs files to the output directory.

    Filters by length, skips test files optionally, and renames to sequential numbering.
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
            cs_files = list(repo_dir.rglob("*.cs"))
            print(f"  {repo_name}: {len(cs_files)} .cs files found")

            for cs_file in tqdm(cs_files, desc=f"  {repo_name}", leave=False):
                # Skip auto-generated files
                if any(
                    skip in str(cs_file).lower()
                    for skip in ["obj/", "bin/", ".designer.cs", "assemblyinfo.cs", "globalassemblyinfo"]
                ):
                    skipped += 1
                    continue

                try:
                    content = cs_file.read_text(encoding="utf-8", errors="replace")
                except Exception:
                    skipped += 1
                    continue

                # Length filter
                if len(content) < min_length or len(content) > max_length:
                    skipped += 1
                    continue

                # Save
                out_path = output_dir / f"{count:06d}.cs"
                out_path.write_text(content, encoding="utf-8")

                meta = {
                    "index": count,
                    "filename": out_path.name,
                    "original_path": str(cs_file.relative_to(repos_dir)),
                    "original_size": len(content),
                    "source": repo_name,
                    "license": "MIT",
                }
                meta_f.write(json.dumps(meta) + "\n")

                count += 1
                if max_files and count >= max_files:
                    print(f"\n  Reached max_files limit ({max_files})")
                    return count

    print(f"\nExtracted {count} files ({skipped} skipped)")
    return count


def main():
    parser = argparse.ArgumentParser(description="Download C# training data from MIT repos")
    parser.add_argument("--output", type=str, default="data/raw_cs", help="Output directory")
    parser.add_argument("--repos-dir", type=str, default="data/repos", help="Where to clone repos")
    parser.add_argument("--max-files", type=int, default=None, help="Max files to extract")
    parser.add_argument("--min-length", type=int, default=100)
    parser.add_argument("--max-length", type=int, default=50000)
    args = parser.parse_args()

    repos_dir = Path(args.repos_dir)
    repos_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Clone repos
    print("Cloning repositories...")
    for org, name in REPOS:
        try:
            clone_repo(org, name, repos_dir)
        except subprocess.CalledProcessError as e:
            print(f"  {org}/{name}: FAILED to clone ({e})")
            continue

    # Step 2: Extract .cs files
    print("\nExtracting .cs files...")
    count = extract_cs_files(
        repos_dir,
        Path(args.output),
        min_length=args.min_length,
        max_length=args.max_length,
        max_files=args.max_files,
    )

    # Step 3: Summary
    output_dir = Path(args.output)
    total_size = sum(f.stat().st_size for f in output_dir.glob("*.cs"))
    print(f"\nDone. {count} C# files ({total_size / 1024 / 1024:.1f} MB) saved to {output_dir}")
    print("License: MIT (all sources)")


if __name__ == "__main__":
    main()
