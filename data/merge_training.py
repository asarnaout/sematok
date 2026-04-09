"""
Merge multiple training JSONL files into one, with shuffling.

Usage:
    python -m data.merge_training \
        --inputs data/finetune/csharp/train.jsonl data/finetune/csharp/train_starcoder.jsonl \
        --output data/finetune/csharp/train_merged.jsonl
"""

import argparse
import random
from pathlib import Path


def merge_and_shuffle(inputs: list[str], output: str, seed: int = 42):
    lines = []
    for path in inputs:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Input not found: {path}")
        with open(p, "r", encoding="utf-8") as f:
            file_lines = [line for line in f if line.strip()]
            print(f"  {p.name}: {len(file_lines):,} records")
            lines.extend(file_lines)

    print(f"\nTotal: {len(lines):,} records")
    rng = random.Random(seed)
    rng.shuffle(lines)

    out = Path(output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for line in lines:
            f.write(line if line.endswith("\n") else line + "\n")

    print(f"Written to {out}")


def main():
    parser = argparse.ArgumentParser(description="Merge and shuffle training JSONL files")
    parser.add_argument("--inputs", type=str, nargs="+", required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    merge_and_shuffle(args.inputs, args.output, args.seed)


if __name__ == "__main__":
    main()
