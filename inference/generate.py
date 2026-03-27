"""
Inference pipeline: compress -> generate -> decompress.

Handles both compressed and baseline models transparently.
"""

import torch

from model.config import GPTConfig
from model.gpt import GPT
from sematok.dictionary import CompressionDictionary
from sematok.compressor import Compressor
from sematok.decompressor import Decompressor
from sematok.lexer import get_safe_ranges
from tokenizer.extended_tokenizer import ExtendedTokenizer


class InferencePipeline:
    """End-to-end inference: prompt -> compress -> generate -> decompress -> output."""

    def __init__(
        self,
        checkpoint_path: str,
        dictionary: CompressionDictionary | None = None,
        compressed: bool = True,
        device: str = "auto",
    ):
        self.compressed = compressed

        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Dictionary and tools
        self.dictionary = dictionary or CompressionDictionary.from_seed()
        self.tokenizer = ExtendedTokenizer(self.dictionary)
        self.compressor = Compressor(self.dictionary) if compressed else None
        self.decompressor = Decompressor(self.dictionary) if compressed else None

        # Load model from checkpoint
        ckpt = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        config = GPTConfig(**ckpt["config"])
        self.model = GPT(config)
        self.model.load_state_dict(ckpt["model"])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {checkpoint_path}")
        print(f"  Parameters: {self.model.count_parameters():,}")
        print(f"  Device: {self.device}")
        print(f"  Mode: {'compressed' if compressed else 'baseline'}")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """
        Generate C# code from a prompt.

        For compressed mode: compress prompt -> generate -> decompress output.
        For baseline mode: generate directly.
        """
        # Step 1: Compress the prompt (if compressed mode)
        if self.compressor:
            try:
                safe_ranges = get_safe_ranges(prompt)
                text = self.compressor.compress(prompt, safe_ranges=safe_ranges)
            except Exception:
                text = self.compressor.compress(prompt)
        else:
            text = prompt

        # Step 2: Tokenize
        tokens = self.tokenizer.encode(text)
        idx = torch.tensor([tokens], dtype=torch.long, device=self.device)

        # Step 3: Generate
        with torch.no_grad():
            output_ids = self.model.generate(
                idx,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
            )

        # Step 4: Decode
        generated_ids = output_ids[0].tolist()
        output_text = self.tokenizer.decode(generated_ids)

        # Step 5: Decompress (if compressed mode)
        if self.decompressor:
            output_text = self.decompressor.decompress(output_text)

        return output_text

    def generate_completion(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.8,
        top_k: int = 50,
    ) -> str:
        """Generate and return only the new tokens (not the prompt)."""
        full = self.generate(prompt, max_new_tokens, temperature, top_k)
        # Remove the prompt from the output
        if full.startswith(prompt):
            return full[len(prompt):]
        return full
