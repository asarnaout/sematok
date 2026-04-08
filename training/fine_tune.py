"""
LoRA fine-tune a model with sematok macro tokens.

Continued pre-training (CLM) via 4-bit QLoRA using Unsloth. The model at
--model must already have its vocabulary expanded (run expand_tokenizer.py
first). This script does NOT resize embeddings -- it trains the already-
initialized new token embeddings alongside LoRA adapters on all attention,
MLP, and embedding layers.

Usage:
    python -m training.fine_tune \\
        --model models/sematok-base \\
        --train data/finetune/train.jsonl \\
        --eval data/finetune/eval.jsonl \\
        --output models/sematok-finetuned
"""

import argparse
import json
import time
from pathlib import Path

from training.vocab_utils import validate_expanded_vocab

try:
    from unsloth import FastLanguageModel
    from unsloth import UnslothTrainer, UnslothTrainingArguments
except ImportError:
    raise ImportError(
        "Unsloth is required for fine-tuning. Install with:\n"
        "  pip install unsloth peft bitsandbytes accelerate"
    )

try:
    from datasets import Dataset
    # Python 3.14 breaks dill's pickle internals (Pickler._batch_setitems
    # signature changed). Patch datasets fingerprinting to avoid the crash.
    # Must patch in arrow_dataset where it's actually called, not just in
    # fingerprint module (the import already cached the reference).
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
DEFAULT_LR = 2e-4
DEFAULT_EMBEDDING_LR_FACTOR = 0.1  # embedding_lr = lr * this
DEFAULT_LORA_R = 16
DEFAULT_LORA_ALPHA_FACTOR = 2  # alpha = r * this
DEFAULT_LORA_DROPOUT = 0.05
DEFAULT_BATCH_SIZE = 2
DEFAULT_GRAD_ACCUM = 8
DEFAULT_EPOCHS = 1
DEFAULT_WARMUP_STEPS = 500
DEFAULT_SAVE_STEPS = 500
DEFAULT_EVAL_STEPS = 500
DEFAULT_SAVE_TOTAL_LIMIT = 2

# Standard for LLaMA-family architectures (Qwen, LLaMA, Mistral, CodeLlama).
# Override with --target-modules for models using different module names
# (e.g., Falcon uses self_attn, Phi uses fc1/fc2).
DEFAULT_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
    "embed_tokens", "lm_head",
]


def load_model(model_path: str, max_seq_length: int):
    """Load the expanded model in 4-bit quantization."""
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


def apply_lora(
    model, lora_r: int, lora_dropout: float,
    target_modules: list[str] | None = None,
):
    """Apply LoRA adapters to all attention, MLP, and embedding layers."""
    if target_modules is None:
        target_modules = DEFAULT_TARGET_MODULES
    lora_alpha = lora_r * DEFAULT_LORA_ALPHA_FACTOR
    print(f"\nApplying LoRA: r={lora_r}, alpha={lora_alpha}, dropout={lora_dropout}")
    print(f"  Target modules: {target_modules}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        use_rslora=True,
        use_gradient_checkpointing="unsloth",
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


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
    """Load JSONL datasets for training and evaluation."""
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
    """Configure and return the UnslothTrainer."""
    embedding_lr = args.lr * DEFAULT_EMBEDDING_LR_FACTOR
    effective_batch = args.batch_size * args.grad_accum
    total_steps = len(train_dataset) // effective_batch * args.epochs

    print(f"\nTraining configuration:")
    print(f"  LR: {args.lr}, embedding LR: {embedding_lr}")
    print(f"  Batch: {args.batch_size} x {args.grad_accum} = {effective_batch} effective")
    print(f"  Epochs: {args.epochs}, ~{total_steps:,} steps")
    print(f"  Warmup: {DEFAULT_WARMUP_STEPS} steps")

    training_args = UnslothTrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        embedding_learning_rate=embedding_lr,
        lr_scheduler_type="cosine",
        warmup_steps=DEFAULT_WARMUP_STEPS,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=not torch.cuda.is_bf16_supported(),
        bf16=torch.cuda.is_bf16_supported(),
        logging_steps=10,
        eval_strategy="steps" if eval_dataset else "no",
        eval_steps=args.eval_steps if eval_dataset else None,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=DEFAULT_SAVE_TOTAL_LIMIT,
        seed=42,
        report_to="none",
        optim="adamw_8bit",
    )

    # Pre-tokenize: Unsloth 2026.4+ expects input_ids in the dataset
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
        )

    train_dataset = train_dataset.map(
        tokenize, batched=True, remove_columns=["text"],
        desc="Tokenizing train",
    )
    if eval_dataset is not None:
        eval_dataset = eval_dataset.map(
            tokenize, batched=True, remove_columns=["text"],
            desc="Tokenizing eval",
        )

    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        args=training_args,
        max_seq_length=args.max_seq_length,
        packing=False,
    )
    return trainer


def save_model(model, tokenizer, output_dir: str):
    """Save merged 16-bit model and LoRA adapters.

    Saves LoRA adapters first (lightweight, always works), then attempts
    the full merged save. If the merge fails (known issue with tied
    embeddings in PEFT), the LoRA adapters are still available.
    """
    lora_dir = output_dir + "-lora"
    merged_dir = output_dir + "-merged"

    # Save LoRA adapters first -- this is small and reliable
    print(f"\nSaving LoRA adapters to {lora_dir}...")
    model.save_pretrained_merged(lora_dir, tokenizer, save_method="lora")
    print("  LoRA adapters saved.")

    # Attempt full merged save -- may fail with tied embeddings
    print(f"Saving merged 16-bit model to {merged_dir}...")
    try:
        model.save_pretrained_merged(merged_dir, tokenizer, save_method="merged_16bit")
        print("  Merged model saved.")
    except Exception as e:
        print(f"\n  WARNING: Merged save failed: {e}")
        print(f"  LoRA adapters are safe at {lora_dir}")
        print(f"  Checkpoints are in {output_dir}/")
        print(f"  You can merge manually later -- ask Claude for help.")


def main():
    parser = argparse.ArgumentParser(
        description="LoRA fine-tune Qwen with sematok macro tokens"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Path to expanded model (from expand_tokenizer.py)",
    )
    parser.add_argument(
        "--train", type=str, required=True,
        help="Path to training JSONL",
    )
    parser.add_argument(
        "--eval", type=str, default=None,
        help="Path to eval JSONL (optional)",
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Output directory for fine-tuned model",
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM)
    parser.add_argument("--max-seq-length", type=int, default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--lora-r", type=int, default=DEFAULT_LORA_R)
    parser.add_argument(
        "--target-modules", type=str, nargs="+", default=None,
        help=f"LoRA target modules (default: {DEFAULT_TARGET_MODULES})",
    )
    parser.add_argument("--save-steps", type=int, default=DEFAULT_SAVE_STEPS)
    parser.add_argument("--eval-steps", type=int, default=DEFAULT_EVAL_STEPS)
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from latest checkpoint in output directory",
    )
    args = parser.parse_args()

    start = time.time()

    model, tokenizer = load_model(args.model, args.max_seq_length)
    target_modules = args.target_modules if args.target_modules else None
    model = apply_lora(model, args.lora_r, DEFAULT_LORA_DROPOUT, target_modules)
    train_dataset, eval_dataset = load_datasets(args.train, args.eval)
    trainer = create_trainer(model, tokenizer, train_dataset, eval_dataset, args)

    print("\nStarting training...")
    if args.resume:
        print("  Resuming from latest checkpoint")
        trainer.train(resume_from_checkpoint=True)
    else:
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

    save_model(model, tokenizer, args.output)

    elapsed = time.time() - start
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal time: {hours}h {minutes}m {seconds}s")


if __name__ == "__main__":
    main()
