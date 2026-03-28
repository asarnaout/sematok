"""
Expand Qwen2.5-Coder-1.5B-Instruct's vocabulary with sematok macro tokens.

Adds 1000 new tokens to the tokenizer and initializes their embeddings
using mean-of-expansion (average the embeddings of the tokens that each
macro expands to). Applied to both embed_tokens and lm_head layers.

New tokens:
  - 917 exact macro tokens:    <|M001|> through <|M917|>
  - 82 template prefix tokens: <|T001:  through <|T082:
  - 1 closing delimiter:       |>

Usage:
    python -m training.expand_tokenizer --output models/qwen-sematok-base
    python -m training.expand_tokenizer --output models/qwen-sematok-base --dictionary sematok/dictionary.json
"""

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AddedToken

from sematok.dictionary import CompressionDictionary

QWEN_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
SLOT_RE = re.compile(r"\{\d+\}")


def build_token_list(dictionary: CompressionDictionary) -> list[str]:
    """Build the list of new tokens to add: M macros + T prefixes + |> closer."""
    tokens = []

    # Exact macro tokens (appear verbatim in compressed text)
    for macro in sorted(dictionary.macro_to_pattern.keys()):
        tokens.append(macro)

    # Template prefix tokens (e.g., "<|T001:" matches start of "<|T001:args|>")
    for macro in sorted(dictionary.macro_to_template.keys()):
        prefix = macro[:-2] + ":"  # "<|T001|>" -> "<|T001:"
        tokens.append(prefix)

    # Closing delimiter for template macros
    tokens.append("|>")

    return tokens


def _get_expansion_ids(
    token_str: str,
    dictionary: CompressionDictionary,
    tokenizer: AutoTokenizer,
) -> list[int]:
    """Get token IDs for a macro's expansion text."""
    if token_str in dictionary.macro_to_pattern:
        expansion = dictionary.macro_to_pattern[token_str]
    elif token_str.endswith(":"):
        # Template prefix like "<|T001:" -> look up "<|T001|>"
        base_macro = token_str[:-1] + "|>"
        template = dictionary.macro_to_template.get(base_macro, "")
        # Strip slot placeholders, keep literal text
        expansion = SLOT_RE.sub("", template).strip()
        if not expansion:
            return []
    elif token_str == "|>":
        # Closing delimiter: use its component characters
        return tokenizer.encode("|>", add_special_tokens=False)
    else:
        return []

    return tokenizer.encode(expansion, add_special_tokens=False)


def init_embeddings(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    new_tokens: list[str],
    dictionary: CompressionDictionary,
):
    """Initialize new token embeddings via mean-of-expansion."""
    embed_layer = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()

    initialized = 0
    skipped = 0

    with torch.no_grad():
        for token_str in new_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            expansion_ids = _get_expansion_ids(token_str, dictionary, tokenizer)

            if not expansion_ids:
                skipped += 1
                continue

            # Mean of expansion token embeddings -> input embedding
            expansion_embeds = embed_layer.weight[expansion_ids]
            embed_layer.weight[token_id] = expansion_embeds.mean(dim=0)

            # Same for output head
            if lm_head is not None:
                head_embeds = lm_head.weight[expansion_ids]
                lm_head.weight[token_id] = head_embeds.mean(dim=0)

            initialized += 1

    print(f"  Initialized: {initialized}, skipped (empty expansion): {skipped}")


def verify(tokenizer: AutoTokenizer, model: AutoModelForCausalLM, new_tokens: list[str]):
    """Print verification stats."""
    # Check round-trip for a few tokens
    samples = [new_tokens[0], new_tokens[len(new_tokens) // 2], new_tokens[-1]]
    print("\nVerification:")
    for tok in samples:
        ids = tokenizer.encode(tok, add_special_tokens=False)
        decoded = tokenizer.decode(ids)
        status = "OK" if decoded == tok else f"MISMATCH: got '{decoded}'"
        print(f"  {tok} -> {len(ids)} token(s) -> {status}")

    # Check a compressed text sample
    sample = '<|M001|> <|M002|> <|T001:_logger,logger|>'
    ids = tokenizer.encode(sample, add_special_tokens=False)
    print(f"\n  Sample: '{sample}'")
    print(f"  Tokens: {len(ids)}")
    print(f"  Decoded: {[tokenizer.decode([i]) for i in ids]}")

    # Check embeddings are non-zero
    embed = model.get_input_embeddings()
    token_id = tokenizer.convert_tokens_to_ids(new_tokens[0])
    norm = embed.weight[token_id].norm().item()
    print(f"\n  Embedding norm for {new_tokens[0]}: {norm:.4f} (should be >0)")


def main():
    parser = argparse.ArgumentParser(
        description="Expand Qwen tokenizer with sematok macro tokens"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for expanded model + tokenizer",
    )
    parser.add_argument(
        "--dictionary", type=str, default="sematok/dictionary.json",
        help="Path to dictionary JSON (default: sematok/dictionary.json)",
    )
    parser.add_argument(
        "--model", type=str, default=QWEN_MODEL,
        help=f"Base model ID (default: {QWEN_MODEL})",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load dictionary
    print(f"Loading dictionary: {args.dictionary}")
    dictionary = CompressionDictionary.load(args.dictionary)
    print(f"  Exact macros: {dictionary.size}, Templates: {dictionary.template_count}")

    # Build token list
    new_tokens = build_token_list(dictionary)
    print(f"  New tokens to add: {len(new_tokens)}")

    # Load base model + tokenizer
    print(f"\nLoading base model: {args.model}")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    original_vocab_size = len(tokenizer)
    print(f"  Original vocab size: {original_vocab_size}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.float32,
    )

    # Add tokens (normalized=False prevents tokenizer from stripping special chars)
    added_tokens = [
        AddedToken(t, special=False, normalized=False) for t in new_tokens
    ]
    num_added = tokenizer.add_tokens(added_tokens)
    print(f"\n  Tokens added: {num_added}")
    print(f"  New vocab size: {len(tokenizer)}")

    # Resize model embeddings
    model.resize_token_embeddings(len(tokenizer))

    # Initialize embeddings
    print("\nInitializing embeddings (mean-of-expansion)...")
    init_embeddings(model, tokenizer, new_tokens, dictionary)

    # Verify
    verify(tokenizer, model, new_tokens)

    # Save
    print(f"\nSaving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save token mapping for reference
    token_map = {}
    for t in new_tokens:
        tid = tokenizer.convert_tokens_to_ids(t)
        token_map[t] = tid
    map_path = output_dir / "macro_token_map.json"
    map_path.write_text(json.dumps(token_map, indent=2), encoding="utf-8")
    print(f"  Token map: {map_path}")

    print(f"\nDone. Expanded model saved to {output_dir}")
    print(f"  Vocab: {original_vocab_size} -> {len(tokenizer)} (+{num_added})")


if __name__ == "__main__":
    main()
