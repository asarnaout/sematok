"""
Extended tokenizer: GPT-2 BPE (via tiktoken) + macro special tokens.

Macro tokens like <|M001|> are registered as additional special tokens
with IDs starting at 50257 (right after the base GPT-2 vocabulary).
"""

import tiktoken

from sematok.dictionary import CompressionDictionary, _make_macro_token

# GPT-2 base vocabulary size
BASE_VOCAB_SIZE = 50257
# Maximum number of macro tokens we support
MAX_MACROS = 200


def create_extended_encoding(
    dictionary: CompressionDictionary,
) -> tiktoken.Encoding:
    """
    Create a tiktoken encoding that includes macro tokens as special tokens.

    The base GPT-2 encoding has 50257 tokens (IDs 0-50256).
    Macro tokens are assigned IDs 50257, 50258, etc.
    """
    base = tiktoken.get_encoding("gpt2")

    # Build special tokens dict: macro_string -> token_id
    # We always register slots for all macros in the dictionary,
    # maintaining consistent IDs based on the macro number.
    special_tokens = {}
    for macro_str in dictionary.macro_tokens:
        # Extract the index from <|M001|> -> 1, <|M002|> -> 2, etc.
        idx = int(macro_str[3:6])
        token_id = BASE_VOCAB_SIZE + idx - 1  # M001 -> 50257, M002 -> 50258, etc.
        special_tokens[macro_str] = token_id

    # Also keep the base GPT-2 special tokens
    special_tokens["<|endoftext|>"] = 50256

    extended = tiktoken.Encoding(
        name="gpt2-extended-macros",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=special_tokens,
    )
    return extended


class ExtendedTokenizer:
    """Wrapper around tiktoken with macro token support."""

    def __init__(self, dictionary: CompressionDictionary):
        self.dictionary = dictionary
        self.encoding = create_extended_encoding(dictionary)
        self._vocab_size = BASE_VOCAB_SIZE + dictionary.size

    @property
    def vocab_size(self) -> int:
        """Total vocabulary size including macro tokens."""
        return self._vocab_size

    def encode(self, text: str) -> list[int]:
        """
        Encode text to token IDs.

        The text should already be compressed (macro tokens present as literal strings).
        Regular text is encoded with GPT-2 BPE, macro tokens map to their special IDs.
        """
        return self.encoding.encode(text, allowed_special="all")

    def decode(self, ids: list[int]) -> str:
        """Decode token IDs back to text (including macro token strings)."""
        return self.encoding.decode(ids)

    def encode_ordinary(self, text: str) -> list[int]:
        """Encode without special token handling (treats macro strings as regular text)."""
        return self.encoding.encode_ordinary(text)

    def token_to_id(self, token: str) -> int | None:
        """Get the ID for a specific token string."""
        try:
            ids = self.encoding.encode(token, allowed_special="all")
            if len(ids) == 1:
                return ids[0]
        except Exception:
            pass
        return None

    def id_to_token(self, token_id: int) -> str:
        """Get the string for a specific token ID."""
        return self.encoding.decode([token_id])

    def is_macro_id(self, token_id: int) -> bool:
        """Check if a token ID corresponds to a macro token."""
        return token_id >= BASE_VOCAB_SIZE

    def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text."""
        return len(self.encode(text))
