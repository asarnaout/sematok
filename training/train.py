"""
Training loop for the GPT model.

Supports both compressed and baseline training modes with identical hyperparameters.
Designed for RTX 4060 8GB: mixed precision, gradient accumulation, checkpointing.

Usage:
    python -m training.train --data data/compressed --out out/compressed
    python -m training.train --data data/baseline --out out/baseline
"""

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import yaml

from model.config import GPTConfig
from model.gpt import GPT
from training.dataset import TokenDataset


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    progress = (step - warmup_steps) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(math.pi * progress))


@torch.no_grad()
def estimate_loss(
    model: GPT,
    train_data: np.ndarray,
    val_data: np.ndarray,
    block_size: int,
    batch_size: int,
    eval_steps: int,
    device: torch.device,
) -> dict[str, float]:
    """Estimate train and val loss over eval_steps random batches."""
    model.eval()
    results = {}

    for split_name, data in [("train", train_data), ("val", val_data)]:
        losses = []
        for _ in range(eval_steps):
            ix = torch.randint(len(data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy(data[i:i+block_size].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
            x, y = x.to(device), y.to(device)
            _, loss = model(x, targets=y)
            losses.append(loss.item())
        results[split_name] = np.mean(losses)

    model.train()
    return results


def train(
    data_dir: str,
    out_dir: str,
    model_config_path: str = "configs/model_small.yaml",
    train_config_path: str = "configs/train.yaml",
    resume_from: str | None = None,
):
    """Main training function."""
    data_dir = Path(data_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load configs
    with open(model_config_path) as f:
        model_cfg = yaml.safe_load(f)
    with open(train_config_path) as f:
        train_cfg = yaml.safe_load(f)

    # Load data metadata to get vocab size
    meta_path = data_dir / "meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            data_meta = json.load(f)
        vocab_size = data_meta["vocab_size"]
        print(f"Vocab size from data: {vocab_size}")
    else:
        vocab_size = model_cfg.get("vocab_size", 50314)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Model
    gpt_config = GPTConfig(
        block_size=model_cfg["block_size"],
        vocab_size=vocab_size,
        n_layer=model_cfg["n_layer"],
        n_head=model_cfg["n_head"],
        n_embd=model_cfg["n_embd"],
        dropout=model_cfg.get("dropout", 0.1),
        bias=model_cfg.get("bias", False),
    )
    model = GPT(gpt_config)
    model = model.to(device)
    print(f"Model: {model.count_parameters():,} parameters ({model.estimate_size_mb():.1f} MB)")

    # Training hyperparams
    batch_size = train_cfg["batch_size"]
    block_size = gpt_config.block_size
    grad_accum = train_cfg["gradient_accumulation_steps"]
    max_lr = train_cfg["learning_rate"]
    min_lr = max_lr / 10
    warmup_steps = train_cfg["warmup_steps"]
    max_steps = train_cfg["max_steps"]
    eval_interval = train_cfg["eval_interval"]
    eval_steps = train_cfg["eval_steps"]
    log_interval = train_cfg["log_interval"]
    use_amp = train_cfg.get("dtype", "float16") == "float16" and device.type == "cuda"

    print(f"Batch size: {batch_size} x {grad_accum} accumulation = {batch_size * grad_accum} effective")
    print(f"Max steps: {max_steps}, LR: {max_lr}")
    print(f"AMP: {use_amp}")

    # Load data as memory-mapped arrays
    train_data = np.memmap(str(data_dir / "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(str(data_dir / "val.bin"), dtype=np.uint16, mode="r")
    print(f"Train data: {len(train_data):,} tokens")
    print(f"Val data: {len(val_data):,} tokens")

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr,
        weight_decay=train_cfg["weight_decay"],
        betas=(0.9, 0.95),
        fused=device.type == "cuda",
    )

    # AMP scaler
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)

    # Resume from checkpoint
    start_step = 0
    best_val_loss = float("inf")
    if resume_from:
        ckpt = torch.load(resume_from, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        print(f"Resumed from step {start_step}")

    # Training loop
    model.train()
    t0 = time.time()
    losses_log = []

    for step in range(start_step, max_steps):
        # Update learning rate
        lr = get_lr(step, warmup_steps, max_steps, max_lr, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        # Gradient accumulation loop
        optimizer.zero_grad(set_to_none=True)
        accum_loss = 0.0

        for micro_step in range(grad_accum):
            # Random batch from training data
            ix = torch.randint(len(train_data) - block_size, (batch_size,))
            x = torch.stack([torch.from_numpy(train_data[i:i+block_size].astype(np.int64)) for i in ix])
            y = torch.stack([torch.from_numpy(train_data[i+1:i+1+block_size].astype(np.int64)) for i in ix])
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                _, loss = model(x, targets=y)
                loss = loss / grad_accum

            scaler.scale(loss).backward()
            accum_loss += loss.item()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        scaler.step(optimizer)
        scaler.update()

        # Logging
        if step % log_interval == 0:
            t1 = time.time()
            dt = t1 - t0
            tokens_per_sec = batch_size * grad_accum * block_size / dt if dt > 0 else 0
            print(
                f"step {step:6d} | loss {accum_loss:.4f} | lr {lr:.2e} | "
                f"{tokens_per_sec:.0f} tok/s | {dt*1000:.0f}ms"
            )
            losses_log.append({"step": step, "train_loss": accum_loss, "lr": lr})
            t0 = time.time()

        # Evaluation
        if step > 0 and step % eval_interval == 0:
            eval_results = estimate_loss(
                model, train_data, val_data, block_size, batch_size, eval_steps, device
            )
            print(
                f"  EVAL step {step} | "
                f"train_loss {eval_results['train']:.4f} | "
                f"val_loss {eval_results['val']:.4f}"
            )

            # Save best model
            if eval_results["val"] < best_val_loss:
                best_val_loss = eval_results["val"]
                ckpt = {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "best_val_loss": best_val_loss,
                    "config": gpt_config.__dict__,
                }
                torch.save(ckpt, out_dir / "best.pt")
                print(f"  Saved best model (val_loss={best_val_loss:.4f})")

    # Save final checkpoint
    ckpt = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "step": max_steps,
        "best_val_loss": best_val_loss,
        "config": gpt_config.__dict__,
    }
    torch.save(ckpt, out_dir / "final.pt")

    # Save loss log
    log_path = out_dir / "losses.json"
    log_path.write_text(json.dumps(losses_log, indent=2), encoding="utf-8")

    print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to {out_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train GPT model")
    parser.add_argument("--data", type=str, required=True, help="Data directory (with train.bin/val.bin)")
    parser.add_argument("--out", type=str, required=True, help="Output directory for checkpoints")
    parser.add_argument("--model-config", type=str, default="configs/model_small.yaml")
    parser.add_argument("--train-config", type=str, default="configs/train.yaml")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    args = parser.parse_args()

    train(
        data_dir=args.data,
        out_dir=args.out,
        model_config_path=args.model_config,
        train_config_path=args.train_config,
        resume_from=args.resume,
    )


if __name__ == "__main__":
    main()
