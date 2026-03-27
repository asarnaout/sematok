"""Tests for the GPT model."""

import torch

from model.config import GPTConfig
from model.gpt import GPT


def _small_config() -> GPTConfig:
    """Tiny config for fast tests."""
    return GPTConfig(
        block_size=64,
        vocab_size=100,
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )


def test_forward_shape():
    """Output logits should have shape (B, T, vocab_size)."""
    config = _small_config()
    model = GPT(config)
    model.eval()

    B, T = 2, 16
    idx = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(idx)

    assert logits.shape == (B, T, config.vocab_size)
    assert loss is None  # No targets provided


def test_forward_with_targets():
    """When targets are provided, loss should be a scalar."""
    config = _small_config()
    model = GPT(config)
    model.train()

    B, T = 2, 16
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))
    logits, loss = model(idx, targets=targets)

    assert logits.shape == (B, T, config.vocab_size)
    assert loss is not None
    assert loss.dim() == 0  # Scalar
    assert loss.item() > 0  # Cross-entropy should be positive


def test_generate():
    """Generation should extend the input sequence."""
    config = _small_config()
    model = GPT(config)
    model.eval()

    B = 1
    start_tokens = torch.randint(0, config.vocab_size, (B, 4))
    max_new = 10
    output = model.generate(start_tokens, max_new_tokens=max_new)

    assert output.shape == (B, 4 + max_new)


def test_generate_with_topk():
    """Generation with top-k sampling should work."""
    config = _small_config()
    model = GPT(config)
    model.eval()

    start = torch.randint(0, config.vocab_size, (1, 2))
    output = model.generate(start, max_new_tokens=5, temperature=0.8, top_k=10)
    assert output.shape == (1, 7)


def test_parameter_count():
    config = _small_config()
    model = GPT(config)
    n_params = model.count_parameters()
    assert n_params > 0
    # Tiny model should be well under 1M params
    assert n_params < 1_000_000


def test_weight_tying():
    """Embedding and LM head should share the same weight tensor."""
    config = _small_config()
    model = GPT(config)
    assert model.transformer.wte.weight is model.lm_head.weight


def test_extended_vocab():
    """Model should work with a larger vocabulary (simulating macro tokens)."""
    config = GPTConfig(
        block_size=64,
        vocab_size=50357,  # 50257 base + 100 macros
        n_layer=2,
        n_head=2,
        n_embd=32,
        dropout=0.0,
        bias=False,
    )
    model = GPT(config)
    model.eval()

    # Use token IDs that include macro range
    idx = torch.tensor([[50257, 50258, 50300, 0, 1, 2]])  # Mix of macro and regular IDs
    logits, _ = model(idx)
    assert logits.shape == (1, 6, 50357)
