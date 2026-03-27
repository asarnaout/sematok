"""
Evaluation metrics for comparing compressed vs baseline models.

Measures:
- Compression ratio (token savings)
- Perplexity on held-out data
- Syntactic validity of generated code
- Context utilization (how much code fits in a fixed window)
"""

import math
from pathlib import Path

import numpy as np
import torch

import tree_sitter_c_sharp as tscsharp
from tree_sitter import Language, Parser

from model.gpt import GPT
from model.config import GPTConfig
from sematok.dictionary import CompressionDictionary
from sematok.compressor import Compressor
from sematok.lexer import get_safe_ranges
from tokenizer.extended_tokenizer import ExtendedTokenizer


CS_LANGUAGE = Language(tscsharp.language())


def compression_ratio(
    corpus_dir: Path, dictionary: CompressionDictionary, max_files: int | None = None
) -> dict:
    """
    Measure token savings from compression across a corpus.

    Returns dict with original/compressed token counts and reduction percentage.
    """
    tokenizer = ExtendedTokenizer(dictionary)
    compressor = Compressor(dictionary)

    cs_files = sorted(corpus_dir.glob("*.cs"))
    if max_files:
        cs_files = cs_files[:max_files]

    total_original = 0
    total_compressed = 0

    for f in cs_files:
        try:
            source = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue

        original_tokens = len(tokenizer.encode(source))
        try:
            safe_ranges = get_safe_ranges(source)
            compressed_text = compressor.compress(source, safe_ranges=safe_ranges)
        except Exception:
            compressed_text = compressor.compress(source)

        compressed_tokens = len(tokenizer.encode(compressed_text))
        total_original += original_tokens
        total_compressed += compressed_tokens

    reduction = (1 - total_compressed / total_original) * 100 if total_original > 0 else 0

    return {
        "original_tokens": total_original,
        "compressed_tokens": total_compressed,
        "reduction_pct": reduction,
        "files_analyzed": len(cs_files),
    }


@torch.no_grad()
def perplexity(
    model: GPT,
    data_path: Path,
    block_size: int,
    batch_size: int = 8,
    max_batches: int = 100,
    device: torch.device | None = None,
) -> float:
    """
    Compute perplexity on a .bin data file.

    Perplexity = exp(average cross-entropy loss).
    """
    if device is None:
        device = next(model.parameters()).device

    data = np.memmap(str(data_path), dtype=np.uint16, mode="r")
    model.eval()

    total_loss = 0.0
    n_batches = 0

    for _ in range(max_batches):
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
        x, y = x.to(device), y.to(device)

        _, loss = model(x, targets=y)
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / n_batches
    return math.exp(avg_loss)


def syntactic_validity(generated_codes: list[str]) -> dict:
    """
    Check if generated code samples are syntactically valid C#.

    Uses tree-sitter to parse each sample. A sample is "valid" if
    the parse tree has no ERROR nodes.
    """
    parser = Parser(CS_LANGUAGE)

    valid = 0
    total = len(generated_codes)

    for code in generated_codes:
        try:
            tree = parser.parse(code.encode("utf-8"))
            has_error = _has_error_node(tree.root_node)
            if not has_error:
                valid += 1
        except Exception:
            pass  # Parse failure = invalid

    return {
        "total": total,
        "valid": valid,
        "invalid": total - valid,
        "validity_rate": valid / total * 100 if total > 0 else 0,
    }


def _has_error_node(node) -> bool:
    """Recursively check if a parse tree has any ERROR nodes."""
    if node.type == "ERROR":
        return True
    for child in node.children:
        if _has_error_node(child):
            return True
    return False


def context_utilization(
    source: str,
    dictionary: CompressionDictionary,
    block_size: int = 1024,
) -> dict:
    """
    Measure how much raw source code fits in a fixed token window.

    For the same block_size, compressed encoding fits more source characters.
    """
    tokenizer = ExtendedTokenizer(dictionary)
    compressor = Compressor(dictionary)

    # Baseline: how many chars fit in block_size tokens
    baseline_tokens = tokenizer.encode(source)
    baseline_chars = 0
    for i, tid in enumerate(baseline_tokens):
        if i >= block_size:
            break
        baseline_chars += len(tokenizer.decode([tid]))

    # Compressed: how many chars fit in block_size tokens
    try:
        safe_ranges = get_safe_ranges(source)
        compressed_text = compressor.compress(source, safe_ranges=safe_ranges)
    except Exception:
        compressed_text = compressor.compress(source)

    compressed_tokens = tokenizer.encode(compressed_text)
    compressed_chars = 0
    for i, tid in enumerate(compressed_tokens):
        if i >= block_size:
            break
        decoded = tokenizer.decode([tid])
        # If it's a macro token, count the decompressed length
        if tokenizer.is_macro_id(tid) and decoded in dictionary.macro_to_pattern:
            compressed_chars += len(dictionary.macro_to_pattern[decoded])
        else:
            compressed_chars += len(decoded)

    improvement = (
        (compressed_chars - baseline_chars) / baseline_chars * 100
        if baseline_chars > 0 else 0
    )

    return {
        "block_size": block_size,
        "baseline_chars": baseline_chars,
        "compressed_chars": compressed_chars,
        "improvement_pct": improvement,
    }
