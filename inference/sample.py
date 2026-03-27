"""
Interactive sampling CLI for side-by-side comparison.

Usage:
    python -m inference.sample --compressed out/compressed/best.pt --baseline out/baseline/best.pt
    python -m inference.sample --compressed out/compressed/best.pt  # single model
"""

import argparse

from inference.generate import InferencePipeline
from sematok.dictionary import CompressionDictionary


def main():
    parser = argparse.ArgumentParser(description="Interactive C# code generation")
    parser.add_argument("--compressed", type=str, help="Path to compressed model checkpoint")
    parser.add_argument("--baseline", type=str, help="Path to baseline model checkpoint")
    parser.add_argument("--dictionary", type=str, default=None, help="Dictionary JSON path")
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    if not args.compressed and not args.baseline:
        parser.error("Provide at least one of --compressed or --baseline")

    # Load dictionary
    if args.dictionary:
        dictionary = CompressionDictionary.load(args.dictionary)
    else:
        dictionary = CompressionDictionary.from_seed()

    # Load models
    pipes = {}
    if args.compressed:
        print("Loading compressed model...")
        pipes["compressed"] = InferencePipeline(
            args.compressed, dictionary, compressed=True, device=args.device
        )
    if args.baseline:
        print("Loading baseline model...")
        pipes["baseline"] = InferencePipeline(
            args.baseline, dictionary, compressed=False, device=args.device
        )

    print("\n" + "=" * 60)
    print("Interactive C# Code Generation")
    print("Type a C# prompt and press Enter. Type 'quit' to exit.")
    print("=" * 60)

    while True:
        try:
            prompt = input("\nPrompt> ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if prompt.lower() in ("quit", "exit", "q"):
            break
        if not prompt:
            continue

        for name, pipe in pipes.items():
            print(f"\n--- {name.upper()} ---")
            output = pipe.generate_completion(
                prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            print(output)

    print("\nBye!")


if __name__ == "__main__":
    main()
