"""
Phase-1 embedding warmup for sematok macro tokens.

Freezes the entire transformer body and trains ONLY the new token embedding
rows (embed_tokens + lm_head) for a few epochs on compressed training data.
This bridges the gap between expand_tokenizer initialization and full LoRA
fine-tuning, letting the new embeddings migrate to where the model actually
needs them before LoRA adapters start learning around them.

Pipeline position: Step 5 -> Step 5.5 (this) -> Step 6.

Usage:
    python -m training.warmup_embeddings \\
        --model models/qwen-sematok-base \\
        --train data/finetune/train.jsonl

    # With eval monitoring and custom output:
    python -m training.warmup_embeddings \\
        --model models/qwen-sematok-base \\
        --train data/finetune/train.jsonl \\
        --eval data/finetune/eval.jsonl \\
        --output models/qwen-sematok-warmed \\
        --epochs 3
"""

import argparse
import json
import shutil
import time
from pathlib import Path

from training.vocab_utils import validate_expanded_vocab, get_new_token_ids

try:
    from unsloth import FastLanguageModel
    from unsloth import UnslothTrainer, UnslothTrainingArguments
except ImportError:
    raise ImportError(
        "Unsloth is required for warmup training. Install with:\n"
        "  pip install unsloth peft bitsandbytes accelerate"
    )

try:
    from datasets import Dataset
    # Python 3.14 breaks dill's pickle internals (Pickler._batch_setitems
    # signature changed). Patch datasets fingerprinting to avoid the crash.
    import datasets.arrow_dataset as _ds_ad
    _ds_ad.generate_fingerprint = lambda dataset: "0" * 64
except ImportError:
    raise ImportError(
        "The datasets library is required. Install with:\n"
        "  pip install datasets"
    )

import torch

# Hyperparameter defaults
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_LR = 1e-3
DEFAULT_BATCH_SIZE = 4
DEFAULT_GRAD_ACCUM = 4
DEFAULT_EPOCHS = 2
DEFAULT_WARMUP_STEPS = 100


def _mask_original_rows(grad: torch.Tensor, original_vocab_size: int) -> torch.Tensor:
    """Zero out gradient rows for original vocabulary tokens."""
    grad[:original_vocab_size] = 0
    return grad


def load_model(model_path: str, max_seq_length: int):
    """Load the expanded model in 4-bit quantization via Unsloth."""
    validate_expanded_vocab(model_path)

    print(f"Loading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
        local_files_only=True,
    )
    print(f"  Vocab size: {len(tokenizer)}")
    print(f"  4-bit quantized, max_seq_length={max_seq_length}")
    return model, tokenizer


def freeze_for_warmup(model, model_path: str) -> int:
    """Freeze all params, unfreeze only new token embedding rows.

    Registers gradient hooks to zero out gradients for original rows,
    ensuring only the new macro token embeddings are updated.

    Returns the original_vocab_size.
    """
    original_vocab_size, new_ids = get_new_token_ids(model_path)

    # Freeze everything
    for param in model.parameters():
        param.requires_grad_(False)

    # Unfreeze embedding weights (full tensor -- hooks handle row masking)
    embed_weight = model.get_input_embeddings().weight
    embed_weight.requires_grad_(True)

    # Register hook to zero out original rows' gradients
    embed_weight.register_hook(
        lambda grad, ovs=original_vocab_size: _mask_original_rows(grad, ovs)
    )

    # Handle lm_head if weights are untied
    lm_head = model.get_output_embeddings()
    if lm_head is not None and lm_head.weight.data_ptr() != embed_weight.data_ptr():
        lm_head.weight.requires_grad_(True)
        lm_head.weight.register_hook(
            lambda grad, ovs=original_vocab_size: _mask_original_rows(grad, ovs)
        )
        print(f"  Weights untied: embed_tokens and lm_head trained independently")
    else:
        print(f"  Weights tied: embed_tokens and lm_head share gradients")

    num_new = len(new_ids)
    hidden_size = embed_weight.shape[1]
    trainable_params = num_new * hidden_size
    if lm_head is not None and lm_head.weight.data_ptr() != embed_weight.data_ptr():
        trainable_params *= 2
    total_params = sum(p.numel() for p in model.parameters())

    print(f"  New token rows: {num_new} (IDs >= {original_vocab_size})")
    print(f"  Effective trainable params: {trainable_params:,} / {total_params:,} "
          f"({100 * trainable_params / total_params:.4f}%)")

    return original_vocab_size


def _load_jsonl(path: str):
    """Load a JSONL file into a Dataset, bypassing datasets caching.

    Python 3.14 breaks dill's pickle internals which datasets uses for
    fingerprinting. Loading via Dataset.from_list avoids that code path.
    """
    import json as _json
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(_json.loads(line))
    return Dataset.from_list(records)


def load_datasets(train_path: str, eval_path: str | None):
    """Load JSONL datasets for training and optional evaluation."""
    if not Path(train_path).exists():
        raise FileNotFoundError(f"Training data not found: {train_path}")

    print(f"\nLoading training data: {train_path}")
    train_dataset = _load_jsonl(train_path)
    print(f"  Train samples: {len(train_dataset):,}")

    eval_dataset = None
    if eval_path and Path(eval_path).exists():
        print(f"Loading eval data: {eval_path}")
        eval_dataset = _load_jsonl(eval_path)
        print(f"  Eval samples: {len(eval_dataset):,}")

    return train_dataset, eval_dataset


def create_trainer(model, tokenizer, train_dataset, eval_dataset, args):
    """Configure trainer for embedding warmup.

    Higher LR than LoRA fine-tuning (1e-3 vs 2e-4) since we're only
    training embeddings. No LoRA, no embedding_learning_rate separation.
    """
    effective_batch = args.batch_size * args.grad_accum
    total_steps = len(train_dataset) // effective_batch * args.epochs

    print(f"\nTraining configuration:")
    print(f"  LR: {args.lr}")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} = {effective_batch} effective")
    print(f"  Epochs: {args.epochs}, ~{total_steps:,} steps")
    print(f"  Warmup: {args.warmup_steps} steps")

    training_args = UnslothTrainingArguments(
        output_dir=args.output or args.model,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=500 if eval_dataset else None,
        save_strategy="no",
        seed=42,
        report_to="none",
        optim="adamw_8bit",
    )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        packing=False,
    )
    return trainer


def save_model(model, tokenizer, output_dir: str, source_dir: str):
    """Save the warmed-up model weights.

    Copies macro_token_map.json from source if output differs from source.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nSaving warmed-up model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    # Preserve macro_token_map.json
    source_map = Path(source_dir) / "macro_token_map.json"
    dest_map = output_path / "macro_token_map.json"
    if source_map.exists() and str(source_map.resolve()) != str(dest_map.resolve()):
        shutil.copy2(source_map, dest_map)
        print(f"  Copied macro_token_map.json from {source_dir}")

    print("  Model saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Phase-1 embedding warmup for sematok macro tokens"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to expanded model (from expand_tokenizer.py)",
    )
    parser.add_argument(
        "--train", type=str, required=True,
        help="Path to training JSONL (compressed data)",
    )
    parser.add_argument(
        "--eval", type=str, default=None,
        help="Path to eval JSONL (optional)",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: overwrite --model in place)",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS)

    args = parser.parse_args()

    if args.output is None:
        args.output = args.model

    start = time.time()

    model, tokenizer = load_model(args.model, args.max_seq_length)
    original_vocab_size = freeze_for_warmup(model, args.model)
    train_dataset, eval_dataset = load_datasets(args.train, args.eval)
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, args)

    print("\nStarting embedding warmup...")
    trainer.train()

    # Print final metrics
    metrics = trainer.state.log_history
    if metrics:
        last = [m for m in metrics if "loss" in m]
        if last:
            print(f"\n  Final train loss: {last[-1]['loss']:.4f}")
        eval_metrics = [m for m in metrics if "eval_loss" in m]
        if eval_metrics:
            print(f"  Final eval loss: {eval_metrics[-1]['eval_loss']:.4f}")

    save_model(model, tokenizer, args.output, args.model)

    elapsed = time.time() - start
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
