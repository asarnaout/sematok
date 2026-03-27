"""
Side-by-side comparison of compressed vs baseline models.

Usage:
    python -m evaluation.compare \
        --compressed-ckpt out/compressed/best.pt \
        --baseline-ckpt out/baseline/best.pt \
        --compressed-data data/compressed \
        --baseline-data data/baseline \
        --corpus data/raw_cs
"""

import argparse
import json
from pathlib import Path

import torch

from model.config import GPTConfig
from model.gpt import GPT
from sematok.dictionary import CompressionDictionary
from evaluation.metrics import (
    compression_ratio,
    perplexity,
    syntactic_validity,
    context_utilization,
)
from inference.generate import InferencePipeline


def load_model(checkpoint_path: str, device: torch.device) -> tuple[GPT, GPTConfig]:
    """Load a model from checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config)
    model.load_state_dict(ckpt["model"])
    model = model.to(device)
    model.eval()
    return model, config


TEST_PROMPTS = [
    "using System;\n\npublic class ",
    "using System;\nusing System.Collections.Generic;\n\nnamespace MyApp\n{\n    public class ",
    "public static void Main(string[] args)\n    {\n        ",
    "public class Calculator\n{\n    public int Add(int a, int b)\n    {\n        ",
    "// A simple linked list implementation\nusing System;\n\npublic class ",
]


def run_comparison(
    compressed_ckpt: str | None,
    baseline_ckpt: str | None,
    compressed_data: str | None,
    baseline_data: str | None,
    corpus: str | None,
    dictionary_path: str | None = None,
    device: str = "auto",
):
    """Run full comparison between compressed and baseline models."""
    dev = torch.device("cuda" if torch.cuda.is_available() and device == "auto" else device if device != "auto" else "cpu")

    dictionary = (
        CompressionDictionary.load(dictionary_path)
        if dictionary_path
        else CompressionDictionary.from_seed()
    )

    results = {"compressed": {}, "baseline": {}}

    # 1. Compression ratio
    if corpus:
        print("\n=== Compression Ratio ===")
        cr = compression_ratio(Path(corpus), dictionary, max_files=100)
        print(f"  Original tokens:   {cr['original_tokens']:,}")
        print(f"  Compressed tokens: {cr['compressed_tokens']:,}")
        print(f"  Reduction:         {cr['reduction_pct']:.1f}%")
        results["compression_ratio"] = cr

    # 2. Perplexity
    print("\n=== Perplexity ===")
    for name, ckpt_path, data_path in [
        ("compressed", compressed_ckpt, compressed_data),
        ("baseline", baseline_ckpt, baseline_data),
    ]:
        if not ckpt_path or not data_path:
            continue
        model, config = load_model(ckpt_path, dev)
        val_bin = Path(data_path) / "val.bin"
        if val_bin.exists():
            ppl = perplexity(model, val_bin, config.block_size, device=dev)
            print(f"  {name:12s}: {ppl:.2f}")
            results[name]["perplexity"] = ppl
        del model

    # 3. Generation quality
    print("\n=== Generation Samples ===")
    for name, ckpt_path, is_compressed in [
        ("compressed", compressed_ckpt, True),
        ("baseline", baseline_ckpt, False),
    ]:
        if not ckpt_path:
            continue

        pipe = InferencePipeline(ckpt_path, dictionary, compressed=is_compressed, device=device)
        generated = []

        for prompt in TEST_PROMPTS:
            output = pipe.generate_completion(prompt, max_new_tokens=128, temperature=0.8, top_k=50)
            generated.append(prompt + output)

        # Syntactic validity
        sv = syntactic_validity(generated)
        print(f"\n  {name} validity: {sv['valid']}/{sv['total']} ({sv['validity_rate']:.0f}%)")
        results[name]["syntactic_validity"] = sv

        # Show a sample
        print(f"\n  {name} sample (prompt: {TEST_PROMPTS[0]!r}):")
        print(f"  {generated[0][:300]}")
        del pipe

    # 4. Context utilization
    if corpus:
        print("\n=== Context Utilization ===")
        cs_files = sorted(Path(corpus).glob("*.cs"))[:10]
        improvements = []
        for f in cs_files:
            source = f.read_text(encoding="utf-8", errors="replace")
            cu = context_utilization(source, dictionary, block_size=1024)
            improvements.append(cu["improvement_pct"])
        avg_improvement = sum(improvements) / len(improvements) if improvements else 0
        print(f"  Avg context improvement: {avg_improvement:.1f}%")
        results["context_utilization_pct"] = avg_improvement

    # Summary table
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    if "compression_ratio" in results:
        print(f"  Token reduction:     {results['compression_ratio']['reduction_pct']:.1f}%")
    if "context_utilization_pct" in results:
        print(f"  Context utilization: +{results['context_utilization_pct']:.1f}%")
    for name in ["compressed", "baseline"]:
        if name in results and results[name]:
            print(f"  {name}:")
            if "perplexity" in results[name]:
                print(f"    Perplexity:  {results[name]['perplexity']:.2f}")
            if "syntactic_validity" in results[name]:
                sv = results[name]["syntactic_validity"]
                print(f"    Validity:    {sv['validity_rate']:.0f}%")

    return results


def main():
    parser = argparse.ArgumentParser(description="Compare compressed vs baseline models")
    parser.add_argument("--compressed-ckpt", type=str)
    parser.add_argument("--baseline-ckpt", type=str)
    parser.add_argument("--compressed-data", type=str)
    parser.add_argument("--baseline-data", type=str)
    parser.add_argument("--corpus", type=str, help="Raw .cs corpus for compression ratio")
    parser.add_argument("--dictionary", type=str, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--output", type=str, default=None, help="Save results JSON")
    args = parser.parse_args()

    results = run_comparison(
        compressed_ckpt=args.compressed_ckpt,
        baseline_ckpt=args.baseline_ckpt,
        compressed_data=args.compressed_data,
        baseline_data=args.baseline_data,
        corpus=args.corpus,
        dictionary_path=args.dictionary,
        device=args.device,
    )

    if args.output:
        Path(args.output).write_text(json.dumps(results, indent=2, default=str))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
