"""
Memory-mapped dataset for training from .bin token files.

Reads pre-tokenized data from numpy .bin files (uint16) and yields
(x, y) pairs where y is x shifted by one position.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class TokenDataset(Dataset):
    """
    Memory-mapped dataset over a .bin file of uint16 token IDs.

    Returns (x, y) pairs of shape (block_size,) where:
        x = tokens[i : i + block_size]
        y = tokens[i+1 : i + block_size + 1]
    """

    def __init__(self, bin_path: str | Path, block_size: int):
        self.block_size = block_size
        self.data = np.memmap(str(bin_path), dtype=np.uint16, mode="r")
        self.n_tokens = len(self.data)

        if self.n_tokens <= block_size:
            raise ValueError(
                f"Data file has {self.n_tokens} tokens, need > {block_size} (block_size)"
            )

    def __len__(self) -> int:
        return self.n_tokens - self.block_size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx : idx + self.block_size + 1].astype(np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y

    @property
    def size_mb(self) -> float:
        return self.n_tokens * 2 / (1024 * 1024)  # uint16 = 2 bytes


def create_dataloaders(
    data_dir: str | Path,
    block_size: int,
    batch_size: int,
    num_workers: int = 0,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Create train and val dataloaders from a prepared data directory.

    Expects data_dir to contain train.bin and val.bin files.
    """
    data_dir = Path(data_dir)
    train_ds = TokenDataset(data_dir / "train.bin", block_size)
    val_ds = TokenDataset(data_dir / "val.bin", block_size)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    print(f"Train: {train_ds.n_tokens:,} tokens ({train_ds.size_mb:.1f} MB)")
    print(f"Val:   {val_ds.n_tokens:,} tokens ({val_ds.size_mb:.1f} MB)")

    return train_loader, val_loader
