"""
Expand a model's vocabulary with sematok macro tokens.

Adds new tokens to the tokenizer and initializes their embeddings using
either Token Distillation (default) or mean-of-expansion (fallback).

Token Distillation (arXiv:2505.20133) optimizes each exact macro's embedding
so the model's early-layer hidden states match what they would be with the
original sub-tokens. Mean-of-expansion is used as fallback for template
prefixes, the closing delimiter, and any exact macro lacking corpus contexts.

New tokens (count depends on dictionary):
  - Exact macro tokens:        <|M00001|> through <|MNNNNN|>
  - Template prefix tokens:    <|T00001:  through <|TNNNNN:
  - 1 closing delimiter:       |>

Supported model architectures: LLaMA-family (Qwen, LLaMA, Mistral, CodeLlama,
DeepSeek-Coder). The distillation step accesses model.model.embed_tokens and
model.model.layers which are standard in these architectures.

Usage:
    # Token distillation (default, requires corpus):
    python -m training.expand_tokenizer --output models/sematok-base --corpus data/raw_cs

    # Mean-of-expansion fallback:
    python -m training.expand_tokenizer --output models/sematok-base --no-distill

    # Different base model:
    python -m training.expand_tokenizer --output models/sematok-base --model meta-llama/CodeLlama-7b-hf
"""

import argparse
import json
import random
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, AddedToken

from sematok.dictionary import CompressionDictionary

DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-Coder-1.5B-Instruct"
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


# ---------------------------------------------------------------------------
# Mean-of-expansion initialization (original method, used as fallback)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Token Distillation
# ---------------------------------------------------------------------------

def build_context_index(
    corpus_dir: Path,
    patterns: list[str],
    num_contexts: int = 25,
    context_chars: int = 300,
    seed: int = 42,
    file_extension: str = ".cs",
) -> dict[str, list[str]]:
    """Scan corpus files once, collect text windows around each pattern.

    Shuffles the file list (seeded) and reads each file, searching for all
    patterns via str.find(). Stops once every pattern has enough contexts.

    Returns {pattern: [text_window, ...]}.
    """
    index: dict[str, list[str]] = {p: [] for p in patterns}
    remaining = set(patterns)

    src_files = sorted(corpus_dir.glob(f"*{file_extension}"))
    rng = random.Random(seed)
    rng.shuffle(src_files)

    print(f"  Scanning corpus ({len(src_files)} files) for {len(patterns)} patterns...")
    for filepath in tqdm(src_files, desc="  Building context index", leave=False):
        if not remaining:
            break

        try:
            text = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        # Check each remaining pattern against this file
        done_in_file = []
        for pattern in remaining:
            pos = 0
            while len(index[pattern]) < num_contexts:
                pos = text.find(pattern, pos)
                if pos == -1:
                    break

                # Extract context window around the match
                start = max(0, pos - context_chars)
                end = min(len(text), pos + len(pattern) + context_chars)
                window = text[start:end]
                index[pattern].append(window)
                pos += len(pattern)

            if len(index[pattern]) >= num_contexts:
                done_in_file.append(pattern)

        for p in done_in_file:
            remaining.discard(p)

    found = sum(1 for v in index.values() if v)
    total = len(patterns)
    print(f"  Found contexts for {found}/{total} patterns "
          f"({total - found} will fall back to mean-of-expansion)")
    return index


def _partial_forward(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    num_layers: int,
) -> torch.Tensor:
    """Forward through embed_tokens + first num_layers transformer layers.

    Returns hidden states [batch, seq_len, hidden_size].
    Uses the model's own forward method to avoid internal API dependencies.
    """
    outputs = model(input_ids, output_hidden_states=True, use_cache=False)
    # hidden_states[0] = embedding output, [i] = after layer i-1
    return outputs.hidden_states[num_layers]


def _find_expansion_span(
    input_ids: list[int],
    expansion_ids: list[int],
) -> int | None:
    """Find the start index of expansion_ids within input_ids.

    Returns the start index, or None if not found.
    """
    exp_len = len(expansion_ids)
    for i in range(len(input_ids) - exp_len + 1):
        if input_ids[i:i + exp_len] == expansion_ids:
            return i
    return None


def distill_embedding(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    token_id: int,
    contexts: list[str],
    expansion_ids: list[int],
    target_layer: int = 4,
    num_steps: int = 100,
    lr: float = 1e-3,
) -> torch.Tensor:
    """Optimize one token's embedding via hidden-state matching.

    For each usable context:
      1. Tokenize context (contains expansion text naturally).
      2. Forward through first target_layer layers -> reference hidden states.
      3. Replace expansion sub-tokens with single new token.
      4. Forward again -> candidate hidden states.
      5. MSE loss on aligned positions after the replacement point.

    Returns the optimized embedding vector [hidden_size].
    """
    device = next(model.parameters()).device
    embed_layer = model.get_input_embeddings()

    # Prepare usable contexts: tokenize and find the expansion span
    prepared = []
    for ctx_text in contexts:
        ids = tokenizer.encode(ctx_text, add_special_tokens=False)
        span_start = _find_expansion_span(ids, expansion_ids)
        if span_start is None:
            continue

        # Build the replacement sequence: prefix + [token_id] + suffix
        prefix = ids[:span_start]
        suffix = ids[span_start + len(expansion_ids):]

        # Need at least a few suffix tokens to compare hidden states
        if len(suffix) < 2:
            continue

        prepared.append((ids, prefix, suffix, span_start))

        if len(prepared) >= 25:
            break

    if not prepared:
        return None

    # Pre-compute reference hidden states and candidate tensors.
    # These are constant across all optimization steps because:
    #   - ref uses orig_ids (original sub-tokens, never contains token_id)
    #   - new_tensor is built from fixed prefix/suffix + constant token_id
    ref_cache = []
    with torch.no_grad():
        for orig_ids, prefix, suffix, span_start in prepared:
            orig_tensor = torch.tensor([orig_ids], device=device)
            ref_hidden = _partial_forward(model, orig_tensor, target_layer)
            new_ids = prefix + [token_id] + suffix
            new_tensor = torch.tensor([new_ids], device=device)
            ref_cache.append((ref_hidden, new_tensor, prefix, suffix, span_start))

    # Initialize from mean-of-expansion as starting point
    with torch.no_grad():
        init_embed = embed_layer.weight[expansion_ids].mean(dim=0).clone()

    # Optimize in float32 to avoid overflow in AdamW's second moment
    # estimates (float16 max ~65504, squared gradients easily exceed this).
    # Cast back to model dtype when injecting into the model.
    model_dtype = embed_layer.weight.dtype
    new_embed = torch.nn.Parameter(init_embed.float())
    optimizer = torch.optim.AdamW([new_embed], lr=lr)

    num_compare = 10  # compare this many positions after the expansion boundary

    def _make_hook(embed_param, pos, dtype):
        def hook(module, input, output):
            out = output.clone()
            out[0, pos] = embed_param.to(dtype)
            return out
        return hook

    for step in range(num_steps):
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        for ref_hidden, new_tensor, prefix, suffix, span_start in ref_cache:
            # Write our optimized embedding into the model temporarily
            embed_layer.weight.data[token_id] = new_embed.data.to(model_dtype)

            # Candidate forward -- use a hook to inject gradient-connected embedding
            new_token_pos = len(prefix)

            hook = embed_layer.register_forward_hook(_make_hook(new_embed, new_token_pos, model_dtype))
            try:
                outputs = model(new_tensor, output_hidden_states=True, use_cache=False)
                hidden = outputs.hidden_states[target_layer]
            finally:
                hook.remove()

            # Compare suffix positions: same content, shifted indices
            # Original: suffix starts at span_start + len(expansion_ids)
            # New: suffix starts at len(prefix) + 1
            orig_suffix_start = span_start + len(expansion_ids)
            new_suffix_start = len(prefix) + 1

            k = min(num_compare, len(suffix), ref_hidden.shape[1] - orig_suffix_start)
            if k < 1:
                continue

            ref_slice = ref_hidden[0, orig_suffix_start:orig_suffix_start + k].detach()
            new_slice = hidden[0, new_suffix_start:new_suffix_start + k]

            total_loss = total_loss + torch.nn.functional.mse_loss(new_slice, ref_slice)
            count += 1

        if count == 0:
            break

        loss = total_loss / count
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    result = new_embed.detach().to(model_dtype)
    if torch.isnan(result).any():
        return None  # fall back to mean-of-expansion
    return result


def init_embeddings_distilled(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    new_tokens: list[str],
    dictionary: CompressionDictionary,
    corpus_dir: Path,
    num_steps: int = 100,
    lr: float = 1e-3,
    target_layer: int = 4,
    num_contexts: int = 25,
    context_window: int = 50,
    file_extension: str = ".cs",
):
    """Token Distillation for exact macros, mean-of-expansion fallback for rest.

    1. Separate tokens into exact macros vs templates/closer.
    2. Build context index for exact macros (single corpus scan).
    3. For each exact macro with contexts: distill_embedding().
    4. For exact macros with no contexts: fall back to mean-of-expansion.
    5. For all templates and closer: mean-of-expansion.
    6. For lm_head on all tokens: mean-of-expansion (safe default).
    """
    embed_layer = model.get_input_embeddings()
    lm_head = model.get_output_embeddings()

    # Separate exact macros from templates/closer
    exact_tokens = [t for t in new_tokens if t in dictionary.macro_to_pattern]
    other_tokens = [t for t in new_tokens if t not in dictionary.macro_to_pattern]

    # Collect the expansion patterns for exact macros
    patterns = [dictionary.macro_to_pattern[t] for t in exact_tokens]

    # Build context index (single scan of the corpus)
    context_chars = context_window * 6  # ~6 chars per token is a rough estimate
    context_index = build_context_index(
        corpus_dir, patterns,
        num_contexts=num_contexts,
        context_chars=context_chars,
        file_extension=file_extension,
    )

    # Freeze model body for distillation
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    # Distill exact macros
    distilled = 0
    fell_back = 0

    print(f"  Distilling {len(exact_tokens)} exact macros "
          f"({num_steps} steps each, layer {target_layer})...")

    for token_str in tqdm(exact_tokens, desc="  Distilling"):
        token_id = tokenizer.convert_tokens_to_ids(token_str)
        pattern = dictionary.macro_to_pattern[token_str]
        expansion_ids = _get_expansion_ids(token_str, dictionary, tokenizer)

        if not expansion_ids:
            fell_back += 1
            continue

        contexts = context_index.get(pattern, [])
        if not contexts:
            # Fall back to mean-of-expansion
            with torch.no_grad():
                expansion_embeds = embed_layer.weight[expansion_ids]
                embed_layer.weight[token_id] = expansion_embeds.mean(dim=0)
            fell_back += 1
            continue

        result = distill_embedding(
            model, tokenizer, token_id, contexts, expansion_ids,
            target_layer=target_layer, num_steps=num_steps, lr=lr,
        )

        if result is not None:
            with torch.no_grad():
                embed_layer.weight[token_id] = result
            distilled += 1
        else:
            # Fall back to mean-of-expansion
            with torch.no_grad():
                expansion_embeds = embed_layer.weight[expansion_ids]
                embed_layer.weight[token_id] = expansion_embeds.mean(dim=0)
            fell_back += 1

    print(f"  Distilled: {distilled}, fell back to mean: {fell_back}")

    # Templates and closer: mean-of-expansion
    other_init = 0
    other_skipped = 0
    with torch.no_grad():
        for token_str in other_tokens:
            token_id = tokenizer.convert_tokens_to_ids(token_str)
            expansion_ids = _get_expansion_ids(token_str, dictionary, tokenizer)
            if not expansion_ids:
                other_skipped += 1
                continue
            expansion_embeds = embed_layer.weight[expansion_ids]
            embed_layer.weight[token_id] = expansion_embeds.mean(dim=0)
            other_init += 1

    print(f"  Templates/closer (mean-of-expansion): {other_init}, "
          f"skipped: {other_skipped}")

    # lm_head: mean-of-expansion for ALL tokens (safe default)
    if lm_head is not None and lm_head.weight.data_ptr() != embed_layer.weight.data_ptr():
        # Weights are untied -- initialize lm_head separately
        head_init = 0
        with torch.no_grad():
            for token_str in new_tokens:
                token_id = tokenizer.convert_tokens_to_ids(token_str)
                expansion_ids = _get_expansion_ids(token_str, dictionary, tokenizer)
                if not expansion_ids:
                    continue
                head_embeds = lm_head.weight[expansion_ids]
                lm_head.weight[token_id] = head_embeds.mean(dim=0)
                head_init += 1
        print(f"  lm_head (mean-of-expansion, untied): {head_init}")
    else:
        print("  lm_head: tied with embed_tokens (shared weights)")


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Expand model tokenizer with sematok macro tokens"
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for expanded model + tokenizer",
    )
    parser.add_argument(
        "--dictionary", type=str, default=None,
        help="Path to dictionary JSON (default: auto-detect from language)",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_BASE_MODEL,
        help=f"Base model ID (default: {DEFAULT_BASE_MODEL})",
    )
    parser.add_argument(
        "--language", type=str, required=True,
        help="Language config to use (e.g. csharp, python)",
    )

    # Token distillation arguments
    parser.add_argument(
        "--corpus", type=str, default=None,
        help="Path to raw corpus (required for --distill)",
    )
    distill_group = parser.add_mutually_exclusive_group()
    distill_group.add_argument(
        "--distill", action="store_true", default=True,
        help="Use token distillation for exact macros (default)",
    )
    distill_group.add_argument(
        "--no-distill", action="store_false", dest="distill",
        help="Use mean-of-expansion only (no distillation)",
    )
    parser.add_argument(
        "--distill-steps", type=int, default=100,
        help="Optimization steps per token (default: 100)",
    )
    parser.add_argument(
        "--distill-lr", type=float, default=1e-3,
        help="AdamW learning rate for distillation (default: 1e-3)",
    )
    parser.add_argument(
        "--distill-layer", type=int, default=4,
        help="Target hidden layer for state matching, 0-indexed (default: 4)",
    )
    parser.add_argument(
        "--distill-contexts", type=int, default=25,
        help="Number of corpus contexts per token (default: 25)",
    )
    parser.add_argument(
        "--context-window", type=int, default=50,
        help="Tokens of surrounding context (default: 50)",
    )

    args = parser.parse_args()

    # Resolve dictionary path
    if args.dictionary is None:
        from sematok.languages import get_dictionary_path
        resolved = get_dictionary_path(args.language)
        if resolved is None:
            parser.error(f"No dictionary found for language '{args.language}'. "
                         "Provide --dictionary explicitly.")
        args.dictionary = str(resolved)

    if args.distill and not args.corpus:
        parser.error("--distill requires --corpus (path to raw corpus). "
                      "Use --no-distill for mean-of-expansion only.")

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
        args.model, dtype=torch.float16, device_map="auto",
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
    if args.distill:
        corpus_dir = Path(args.corpus)
        from sematok.languages import get_language
        lang = get_language(args.language)
        print("\nInitializing embeddings (token distillation)...")
        init_embeddings_distilled(
            model, tokenizer, new_tokens, dictionary,
            corpus_dir=corpus_dir,
            num_steps=args.distill_steps,
            lr=args.distill_lr,
            target_layer=args.distill_layer,
            num_contexts=args.distill_contexts,
            context_window=args.context_window,
            file_extension=lang.file_extension,
        )
    else:
        print("\nInitializing embeddings (mean-of-expansion)...")
        init_embeddings(model, tokenizer, new_tokens, dictionary)

    # Verify
    verify(tokenizer, model, new_tokens)

    # Save
    print(f"\nSaving to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Save token mapping for reference (includes original vocab size)
    token_map = {"_original_vocab_size": original_vocab_size}
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
