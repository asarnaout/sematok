"""GPT model configuration."""

from dataclasses import dataclass


@dataclass
class GPTConfig:
    block_size: int = 1024       # max sequence length
    vocab_size: int = 60256      # 50257 base GPT-2 + up to 9999 macro tokens (actual count set by dictionary)
    n_layer: int = 8             # number of transformer blocks
    n_head: int = 8              # number of attention heads
    n_embd: int = 512            # embedding dimension
    dropout: float = 0.1
    bias: bool = False           # bias in Linear layers and LayerNorm
