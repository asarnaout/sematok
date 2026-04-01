"""Shared vocabulary validation for sematok training scripts."""

import json
from pathlib import Path


def validate_expanded_vocab(model_path: str) -> int:
    """Check that a model directory contains a properly vocabulary-expanded model.

    Validates:
      1. config.json exists
      2. macro_token_map.json exists with _original_vocab_size
      3. Current vocab_size > original vocab size (tokens were added)

    Returns the vocab_size on success.
    """
    model_dir = Path(model_path)

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"No config.json found in {model_path}. "
            "Is this the right model directory?"
        )

    token_map_path = model_dir / "macro_token_map.json"
    if not token_map_path.exists():
        raise FileNotFoundError(
            f"No macro_token_map.json in {model_path}. "
            "Run expand_tokenizer.py first (Step 5)."
        )

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    vocab_size = config.get("vocab_size", 0)

    with open(token_map_path, encoding="utf-8") as f:
        token_map = json.load(f)
    original_vocab_size = token_map.get("_original_vocab_size", 0)

    if original_vocab_size > 0 and vocab_size <= original_vocab_size:
        raise ValueError(
            f"Model has vocab_size={vocab_size}, expected > {original_vocab_size}. "
            "Run expand_tokenizer.py first (Step 5)."
        )

    return vocab_size


def get_new_token_ids(model_path: str) -> tuple[int, list[int]]:
    """Read macro_token_map.json and return (original_vocab_size, new_token_ids).

    If the map contains '_original_vocab_size', uses that directly.
    Otherwise computes it as vocab_size - num_new_tokens.
    """
    model_dir = Path(model_path)
    token_map_path = model_dir / "macro_token_map.json"

    with open(token_map_path, encoding="utf-8") as f:
        token_map = json.load(f)

    # Separate metadata keys (prefixed with _) from token entries
    new_ids = [v for k, v in token_map.items() if not k.startswith("_")]
    original_vocab_size = token_map.get("_original_vocab_size")

    if original_vocab_size is None:
        config_path = model_dir / "config.json"
        with open(config_path, encoding="utf-8") as f:
            config = json.load(f)
        original_vocab_size = config["vocab_size"] - len(new_ids)

    return original_vocab_size, new_ids
