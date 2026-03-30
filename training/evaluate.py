"""
Evaluate sematok fine-tuning: macro comprehension and capability retention.

Computes perplexity across four configurations:
  A = Base model on uncompressed C#        (baseline)
  B = Fine-tuned model on compressed C#    (our goal)
  C = Fine-tuned model on uncompressed C#  (retention check)
  D = Base model on compressed C#          (sanity: should be terrible)

Usage:
    # All four configs:
    python -m training.evaluate --all \\
        --base-model models/qwen-sematok-base \\
        --finetuned-model models/qwen-sematok-finetuned-merged

    # Quick validation with 500 files:
    python -m training.evaluate --all \\
        --base-model models/qwen-sematok-base \\
        --finetuned-model models/qwen-sematok-finetuned-merged \\
        --max-files 500

    # Single config:
    python -m training.evaluate \\
        --model models/qwen-sematok-finetuned-merged \\
        --eval data/finetune/eval.jsonl
"""

import argparse
import gc
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

# Set encoding before Unsloth imports (sloth emoji crashes Windows CP1252)
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

try:
    from unsloth import FastLanguageModel
except ImportError:
    raise ImportError(
        "Unsloth is required for evaluation. Install with:\n"
        "  pip install unsloth peft bitsandbytes accelerate"
    )

# Python 3.14 compat: patch datasets fingerprinting (dill pickle breakage)
try:
    import datasets.arrow_dataset as _ds_ad
    _ds_ad.generate_fingerprint = lambda dataset: "0" * 64
except Exception:
    pass

import torch
from tqdm import tqdm

# Defaults
DEFAULT_MAX_SEQ_LENGTH = 2048
DEFAULT_COMPRESSED_EVAL = "data/finetune/eval.jsonl"
DEFAULT_DICTIONARY = "sematok/dictionary.json"
DEFAULT_OUTPUT = "out/eval_results.json"

EXPECTED_VOCAB_SIZE = 152665


# ---------------------------------------------------------------------------
# Model loading / unloading
# ---------------------------------------------------------------------------

def load_model(model_path: str, max_seq_length: int, load_in_4bit: bool = True):
    """Load model + tokenizer via Unsloth."""
    config_path = Path(model_path) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"No config.json in {model_path}")

    with open(config_path, encoding="utf-8") as f:
        config = json.load(f)
    vocab_size = config.get("vocab_size", 0)
    if vocab_size != EXPECTED_VOCAB_SIZE:
        raise ValueError(
            f"vocab_size={vocab_size}, expected {EXPECTED_VOCAB_SIZE}. "
            "Run expand_tokenizer.py first."
        )

    print(f"\nLoading model: {model_path}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_path,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
        local_files_only=True,
    )
    FastLanguageModel.for_inference(model)
    model.eval()
    quant = "4-bit" if load_in_4bit else "full precision"
    print(f"  Loaded ({quant}, vocab={len(tokenizer)})")
    return model, tokenizer


def unload_model(model, tokenizer):
    """Free GPU memory between model loads."""
    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("  Model unloaded, GPU cache cleared.")


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_texts(path: str) -> list[str]:
    """Load JSONL file, return list of text strings."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                texts.append(record["text"])
    return texts


def generate_uncompressed_eval(
    compressed_path: str,
    dictionary_path: str,
    output_path: str,
) -> str:
    """Decompress eval.jsonl to produce uncompressed version.

    Skips if output already exists with matching line count.
    """
    compressed_path = Path(compressed_path)
    output_path = Path(output_path)

    # Check if we can skip
    if output_path.exists():
        with open(compressed_path, encoding="utf-8") as f:
            compressed_count = sum(1 for line in f if line.strip())
        with open(output_path, encoding="utf-8") as f:
            uncompressed_count = sum(1 for line in f if line.strip())
        if compressed_count == uncompressed_count:
            print(f"  Uncompressed eval exists ({uncompressed_count} files), skipping.")
            return str(output_path)
        print(f"  Line count mismatch ({compressed_count} vs {uncompressed_count}), regenerating.")

    # Generate
    from sematok.dictionary import CompressionDictionary
    from sematok.decompressor import Decompressor

    print(f"  Generating uncompressed eval: {output_path}")
    dictionary = CompressionDictionary.load(str(dictionary_path))
    decompressor = Decompressor(dictionary)

    count = 0
    with open(compressed_path, "r", encoding="utf-8") as fin, \
         open(output_path, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            original_text = decompressor.decompress(record["text"])
            fout.write(json.dumps({"text": original_text}, ensure_ascii=False) + "\n")
            count += 1

    print(f"  Decompressed {count} files.")
    return str(output_path)


def subsample(texts: list[str], max_files: int, seed: int) -> list[str]:
    """Subsample texts if max_files > 0."""
    if max_files <= 0 or max_files >= len(texts):
        return texts
    rng = random.Random(seed)
    indices = rng.sample(range(len(texts)), max_files)
    indices.sort()
    return [texts[i] for i in indices]


# ---------------------------------------------------------------------------
# Perplexity computation
# ---------------------------------------------------------------------------

def compute_perplexity(
    model,
    tokenizer,
    texts: list[str],
    max_seq_length: int,
    label: str = "",
) -> dict:
    """Compute token-weighted perplexity over a list of text strings.

    Returns dict with avg_loss, perplexity, total_tokens, num_files, num_skipped.
    """
    total_loss_sum = 0.0
    total_positions = 0
    num_files = 0
    num_skipped = 0

    desc = f"  {label}" if label else "  Perplexity"
    for text in tqdm(texts, desc=desc, unit="file"):
        encoding = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        )
        input_ids = encoding["input_ids"].to(model.device)
        seq_len = input_ids.shape[1]

        if seq_len < 2:
            num_skipped += 1
            continue

        try:
            with torch.no_grad():
                outputs = model(input_ids=input_ids, labels=input_ids)
                loss = outputs.loss
        except torch.cuda.OutOfMemoryError:
            print(f"\n  WARNING: OOM on file with {seq_len} tokens, skipping.")
            torch.cuda.empty_cache()
            num_skipped += 1
            continue

        if torch.isnan(loss) or torch.isinf(loss):
            num_skipped += 1
            continue

        loss_val = loss.item()
        positions = seq_len - 1  # number of next-token predictions

        total_loss_sum += loss_val * positions
        total_positions += positions
        num_files += 1

    if total_positions == 0:
        return {
            "avg_loss": float("inf"),
            "perplexity": float("inf"),
            "total_tokens": 0,
            "num_files": 0,
            "num_skipped": num_skipped,
        }

    avg_loss = total_loss_sum / total_positions
    ppl = math.exp(avg_loss)

    return {
        "avg_loss": round(avg_loss, 4),
        "perplexity": round(ppl, 4),
        "total_tokens": total_positions + num_files,
        "num_files": num_files,
        "num_skipped": num_skipped,
    }


# ---------------------------------------------------------------------------
# Results display
# ---------------------------------------------------------------------------

CONFIG_DESCRIPTIONS = {
    "A": ("Base", "Uncompressed"),
    "B": ("Finetuned", "Compressed"),
    "C": ("Finetuned", "Uncompressed"),
    "D": ("Base", "Compressed"),
}


def build_analysis(results: dict) -> dict:
    """Compute comparison metrics from the four config results."""
    analysis = {}

    a = results.get("A", {})
    b = results.get("B", {})
    c = results.get("C", {})
    d = results.get("D", {})

    # Macro comprehension: B vs D
    if b.get("perplexity") and d.get("perplexity"):
        b_ppl = b["perplexity"]
        d_ppl = d["perplexity"]
        factor = d_ppl / b_ppl if b_ppl > 0 else 0
        analysis["macro_comprehension"] = {
            "B_ppl": b_ppl,
            "D_ppl": d_ppl,
            "improvement_factor": round(factor, 2),
        }

    # C# retention: C vs A
    if c.get("perplexity") and a.get("perplexity"):
        c_ppl = c["perplexity"]
        a_ppl = a["perplexity"]
        delta = ((c_ppl - a_ppl) / a_ppl * 100) if a_ppl > 0 else 0
        analysis["csharp_retention"] = {
            "C_ppl": c_ppl,
            "A_ppl": a_ppl,
            "delta_pct": round(delta, 2),
        }

    # Effective compression: B vs A
    if b.get("perplexity") and a.get("perplexity"):
        b_ppl = b["perplexity"]
        a_ppl = a["perplexity"]
        delta = ((b_ppl - a_ppl) / a_ppl * 100) if a_ppl > 0 else 0
        analysis["effective_compression"] = {
            "B_ppl": b_ppl,
            "A_ppl": a_ppl,
            "delta_pct": round(delta, 2),
        }

    return analysis


def print_results_table(results: dict, analysis: dict):
    """Print formatted comparison table."""
    print("\n" + "=" * 72)
    print("Sematok Evaluation Results")
    print("=" * 72)

    header = f"{'Config':<8} {'Model':<12} {'Input':<14} {'PPL':>10} {'Loss':>8} {'Tokens':>12} {'Files':>7}"
    print(header)
    print("-" * 72)

    for cfg in ["A", "B", "C", "D"]:
        if cfg not in results:
            continue
        r = results[cfg]
        model_name, input_name = CONFIG_DESCRIPTIONS[cfg]
        ppl = f"{r['perplexity']:.2f}" if r['perplexity'] < 1e6 else ">>>"
        loss = f"{r['avg_loss']:.4f}" if r['avg_loss'] < 1e6 else ">>>"
        tokens = f"{r['total_tokens']:,}"
        files = f"{r['num_files']:,}"
        print(f"{cfg:<8} {model_name:<12} {input_name:<14} {ppl:>10} {loss:>8} {tokens:>12} {files:>7}")

    if analysis:
        print("\n" + "-" * 72)
        print("Analysis")
        print("-" * 72)

        if "macro_comprehension" in analysis:
            mc = analysis["macro_comprehension"]
            print(f"  Macro comprehension (B vs D):  PPL {mc['B_ppl']:.2f} vs {mc['D_ppl']:.2f}"
                  f"  ({mc['improvement_factor']:.1f}x improvement)")

        if "csharp_retention" in analysis:
            cr = analysis["csharp_retention"]
            sign = "+" if cr["delta_pct"] >= 0 else ""
            print(f"  C# retention (C vs A):         PPL {cr['C_ppl']:.2f} vs {cr['A_ppl']:.2f}"
                  f"  ({sign}{cr['delta_pct']:.1f}% delta)")

        if "effective_compression" in analysis:
            ec = analysis["effective_compression"]
            sign = "+" if ec["delta_pct"] >= 0 else ""
            print(f"  Effective compression (B vs A): PPL {ec['B_ppl']:.2f} vs {ec['A_ppl']:.2f}"
                  f"  ({sign}{ec['delta_pct']:.1f}% delta)")

    print("=" * 72)


def save_results(results: dict, analysis: dict, output_path: str, args):
    """Save results to JSON."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "max_seq_length": args.max_seq_length,
        "load_in_4bit": args.load_in_4bit,
        "max_files": args.max_files,
        "configs": results,
        "analysis": analysis,
    }

    with open(output, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate sematok fine-tuning: perplexity across 4 configs"
    )

    # All-configs mode
    parser.add_argument("--all", action="store_true",
                        help="Run all four configs (A, B, C, D)")
    parser.add_argument("--base-model", type=str,
                        help="Base model path (with --all)")
    parser.add_argument("--finetuned-model", type=str,
                        help="Fine-tuned model path (with --all)")

    # Single-config mode
    parser.add_argument("--model", type=str,
                        help="Model path (single-config mode)")
    parser.add_argument("--eval", type=str,
                        help="Eval JSONL path (single-config mode)")

    # Shared options
    parser.add_argument("--compressed-eval", type=str,
                        default=DEFAULT_COMPRESSED_EVAL,
                        help=f"Compressed eval JSONL (default: {DEFAULT_COMPRESSED_EVAL})")
    parser.add_argument("--dictionary", type=str,
                        default=DEFAULT_DICTIONARY,
                        help=f"Dictionary path for decompression (default: {DEFAULT_DICTIONARY})")
    parser.add_argument("--output", type=str, default=DEFAULT_OUTPUT,
                        help=f"JSON results output (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--max-seq-length", type=int,
                        default=DEFAULT_MAX_SEQ_LENGTH)
    parser.add_argument("--max-files", type=int, default=0,
                        help="Limit to N files (0 = all)")
    parser.add_argument("--load-in-4bit", action="store_true", default=True,
                        help="Load model in 4-bit (default)")
    parser.add_argument("--no-4bit", dest="load_in_4bit",
                        action="store_false",
                        help="Load model in full precision")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    start = time.time()

    if args.all:
        _run_all_configs(args)
    elif args.model and args.eval:
        _run_single_config(args)
    else:
        parser.error("Use --all with --base-model/--finetuned-model, "
                      "or --model with --eval")

    elapsed = time.time() - start
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\nTotal time: {hours}h {minutes}m {seconds}s")


def _run_single_config(args):
    """Evaluate one model on one dataset."""
    texts = load_texts(args.eval)
    texts = subsample(texts, args.max_files, args.seed)
    print(f"Evaluating {len(texts)} files")

    model, tokenizer = load_model(
        args.model, args.max_seq_length, args.load_in_4bit
    )
    result = compute_perplexity(
        model, tokenizer, texts, args.max_seq_length, label="Eval"
    )
    unload_model(model, tokenizer)

    print(f"\n  Loss: {result['avg_loss']:.4f}")
    print(f"  Perplexity: {result['perplexity']:.2f}")
    print(f"  Tokens: {result['total_tokens']:,}")
    print(f"  Files: {result['num_files']:,} ({result['num_skipped']} skipped)")


def _run_all_configs(args):
    """Run all four configs: A, B, C, D."""
    if not args.base_model or not args.finetuned_model:
        raise ValueError("--all requires --base-model and --finetuned-model")

    compressed_path = args.compressed_eval
    if not Path(compressed_path).exists():
        raise FileNotFoundError(f"Compressed eval not found: {compressed_path}")

    # Generate uncompressed eval
    uncompressed_path = str(
        Path(compressed_path).parent / "eval_uncompressed.jsonl"
    )
    print("Preparing eval data...")
    generate_uncompressed_eval(compressed_path, args.dictionary, uncompressed_path)

    # Load texts
    print("Loading eval texts...")
    compressed_texts = load_texts(compressed_path)
    uncompressed_texts = load_texts(uncompressed_path)

    # Subsample (same indices for both)
    if args.max_files > 0 and args.max_files < len(compressed_texts):
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(compressed_texts)), args.max_files))
        compressed_texts = [compressed_texts[i] for i in indices]
        uncompressed_texts = [uncompressed_texts[i] for i in indices]

    n = len(compressed_texts)
    print(f"Evaluating {n} files per config\n")

    results = {}

    # --- Base model: configs A and D ---
    model, tokenizer = load_model(
        args.base_model, args.max_seq_length, args.load_in_4bit
    )

    print("\nConfig A: Base model + uncompressed C#")
    results["A"] = compute_perplexity(
        model, tokenizer, uncompressed_texts, args.max_seq_length,
        label="A (base+uncompressed)",
    )
    results["A"]["model"] = args.base_model
    results["A"]["eval_data"] = uncompressed_path
    print(f"  -> PPL: {results['A']['perplexity']:.2f}, Loss: {results['A']['avg_loss']:.4f}")

    print("\nConfig D: Base model + compressed C#")
    results["D"] = compute_perplexity(
        model, tokenizer, compressed_texts, args.max_seq_length,
        label="D (base+compressed)",
    )
    results["D"]["model"] = args.base_model
    results["D"]["eval_data"] = compressed_path
    print(f"  -> PPL: {results['D']['perplexity']:.2f}, Loss: {results['D']['avg_loss']:.4f}")

    unload_model(model, tokenizer)

    # --- Fine-tuned model: configs B and C ---
    model, tokenizer = load_model(
        args.finetuned_model, args.max_seq_length, args.load_in_4bit
    )

    print("\nConfig B: Fine-tuned model + compressed C#")
    results["B"] = compute_perplexity(
        model, tokenizer, compressed_texts, args.max_seq_length,
        label="B (finetuned+compressed)",
    )
    results["B"]["model"] = args.finetuned_model
    results["B"]["eval_data"] = compressed_path
    print(f"  -> PPL: {results['B']['perplexity']:.2f}, Loss: {results['B']['avg_loss']:.4f}")

    print("\nConfig C: Fine-tuned model + uncompressed C#")
    results["C"] = compute_perplexity(
        model, tokenizer, uncompressed_texts, args.max_seq_length,
        label="C (finetuned+uncompressed)",
    )
    results["C"]["model"] = args.finetuned_model
    results["C"]["eval_data"] = uncompressed_path
    print(f"  -> PPL: {results['C']['perplexity']:.2f}, Loss: {results['C']['avg_loss']:.4f}")

    unload_model(model, tokenizer)

    # --- Results ---
    analysis = build_analysis(results)
    print_results_table(results, analysis)
    save_results(results, analysis, args.output, args)


if __name__ == "__main__":
    main()
