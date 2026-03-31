"""
Evaluate functional correctness: can the model generate correct C# from
compressed prompts?

Splits eval files into prefix (prompt) and suffix (ground truth), generates
a continuation, and measures similarity between generation and ground truth.

Two configurations:
  A = Base model on uncompressed prefix      (baseline)
  B = Fine-tuned model on compressed prefix   (our goal)

Usage:
    python -m training.evaluate_correctness \\
        --base-model models/qwen-sematok-base \\
        --finetuned-model models/qwen-sematok-finetuned-merged

    # Quick test:
    python -m training.evaluate_correctness \\
        --base-model models/qwen-sematok-base \\
        --finetuned-model models/qwen-sematok-finetuned-merged \\
        --max-files 50 --gen-tokens 64
"""

import argparse
import collections
import gc
import json
import math
import os
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

os.environ.setdefault("PYTHONIOENCODING", "utf-8")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from training.evaluate import (
    generate_uncompressed_eval,
    load_texts,
    unload_model,
)

# We can't use Unsloth's FastLanguageModel here. Unsloth patches the
# model class with fast_forward_inference which assumes single-token
# autoregressive decoding. This breaks model.generate() on long
# prefixes (rotary embedding shape mismatch). Loading with plain
# transformers + bitsandbytes avoids the issue entirely.

EXPECTED_VOCAB_SIZE = 152665


def load_model_for_generation(model_path: str, max_seq_length: int, load_in_4bit: bool = True):
    """Load model + tokenizer for generation (plain transformers, no Unsloth)."""
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
    kwargs = {}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    else:
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(
        model_path, device_map="auto", local_files_only=True, **kwargs,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    model.eval()
    quant = "4-bit" if load_in_4bit else "full precision"
    print(f"  Loaded ({quant}, vocab={len(tokenizer)}, generation mode)")
    return model, tokenizer

# ---------------------------------------------------------------------------
# Prefix / suffix splitting
# ---------------------------------------------------------------------------


def split_prefix_suffix(text, tokenizer, prefix_ratio, min_tokens, max_prefix_tokens):
    """Split text into prefix and suffix by token count.

    Returns (prefix_text, suffix_text, prefix_ids, suffix_ids) or None if the
    file is too short.
    """
    token_ids = tokenizer(
        text,
        add_special_tokens=False,
        truncation=False,
        return_tensors=None,
    )["input_ids"]

    total = len(token_ids)
    if total < min_tokens:
        return None

    split_point = min(int(total * prefix_ratio), max_prefix_tokens)
    if split_point < 1 or split_point >= total:
        return None

    prefix_ids = token_ids[:split_point]
    suffix_ids = token_ids[split_point:]

    prefix_text = tokenizer.decode(prefix_ids, skip_special_tokens=False)
    suffix_text = tokenizer.decode(suffix_ids, skip_special_tokens=False)

    return prefix_text, suffix_text, prefix_ids, suffix_ids


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_continuation(model, tokenizer, prefix_text, max_new_tokens):
    """Generate a continuation from a prefix.

    Returns (generated_text, generated_token_ids) or None on OOM.
    """
    inputs = tokenizer(
        prefix_text,
        return_tensors="pt",
        add_special_tokens=False,
        truncation=False,
    )
    input_ids = inputs["input_ids"].to(model.device)

    try:
        with torch.no_grad():
            output_ids = model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return None

    generated_ids = output_ids[0, input_ids.shape[1]:].tolist()
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=False)
    return generated_text, generated_ids


# ---------------------------------------------------------------------------
# Metrics (standard code completion: EM, ES, BLEU-4, chrF)
# ---------------------------------------------------------------------------


def compute_exact_match(gen_text, ref_text):
    """Binary exact match (standard EM). 1 if identical, 0 otherwise."""
    return 1.0 if gen_text == ref_text else 0.0


def _levenshtein(s, t):
    """Levenshtein edit distance via standard DP."""
    n, m = len(s), len(t)
    if n == 0:
        return m
    if m == 0:
        return n
    # Use two-row optimization for memory efficiency
    prev = list(range(m + 1))
    curr = [0] * (m + 1)
    for i in range(1, n + 1):
        curr[0] = i
        for j in range(1, m + 1):
            cost = 0 if s[i - 1] == t[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[m]


def compute_edit_similarity(gen_text, ref_text):
    """Edit Similarity = 1 - (levenshtein / max(len(gen), len(ref))).

    Standard metric from RepoBench (ICLR 2024), Meta-Tokens, CodeXGLUE.
    """
    if not gen_text and not ref_text:
        return 1.0
    max_len = max(len(gen_text), len(ref_text))
    if max_len == 0:
        return 1.0
    dist = _levenshtein(gen_text, ref_text)
    return 1.0 - dist / max_len


def compute_bleu4(reference, hypothesis):
    """Sentence-level BLEU-4 with add-1 smoothing (Chen & Cherry, 2014).

    Tokenizes on whitespace. Standard in code generation evaluation.
    """
    ref_tokens = reference.split()
    hyp_tokens = hypothesis.split()

    if not hyp_tokens or not ref_tokens:
        return 0.0

    # Brevity penalty
    bp = min(1.0, math.exp(1.0 - len(ref_tokens) / len(hyp_tokens))) if len(hyp_tokens) > 0 else 0.0

    # N-gram precisions with add-1 smoothing for n > 1
    log_avg = 0.0
    for n in range(1, 5):
        hyp_ngrams = collections.Counter(
            tuple(hyp_tokens[i:i + n]) for i in range(len(hyp_tokens) - n + 1)
        )
        ref_ngrams = collections.Counter(
            tuple(ref_tokens[i:i + n]) for i in range(len(ref_tokens) - n + 1)
        )
        clipped = sum(min(hyp_ngrams[ng], ref_ngrams[ng]) for ng in hyp_ngrams)
        total = sum(hyp_ngrams.values())

        if n == 1:
            precision = clipped / total if total > 0 else 0.0
        else:
            # Add-1 smoothing for higher-order n-grams
            precision = (clipped + 1) / (total + 1) if total > 0 else 0.0

        if precision == 0.0:
            return 0.0

        log_avg += math.log(precision) / 4.0

    return bp * math.exp(log_avg)


def compute_chrf(reference, hypothesis, max_n=6):
    """Character n-gram F-score (chrF).

    Recommended by Evtikhiev et al. (JSS 2023, "Out of the BLEU") as the
    best reference-based metric for code generation evaluation.
    """
    if not reference and not hypothesis:
        return 1.0
    if not reference or not hypothesis:
        return 0.0

    def _char_ngrams(text, n):
        return collections.Counter(text[i:i + n] for i in range(len(text) - n + 1))

    f_scores = []
    for n in range(1, max_n + 1):
        ref_ngrams = _char_ngrams(reference, n)
        hyp_ngrams = _char_ngrams(hypothesis, n)
        if not ref_ngrams or not hyp_ngrams:
            continue
        overlap = sum((ref_ngrams & hyp_ngrams).values())
        precision = overlap / sum(hyp_ngrams.values()) if sum(hyp_ngrams.values()) > 0 else 0.0
        recall = overlap / sum(ref_ngrams.values()) if sum(ref_ngrams.values()) > 0 else 0.0
        if precision + recall > 0:
            f_scores.append(2 * precision * recall / (precision + recall))
        else:
            f_scores.append(0.0)

    return sum(f_scores) / len(f_scores) if f_scores else 0.0


# ---------------------------------------------------------------------------
# Config evaluation loop
# ---------------------------------------------------------------------------


def evaluate_config(
    model, tokenizer, texts, label, decompressor,
    gen_tokens, prefix_ratio, min_tokens, max_prefix_tokens,
):
    """Evaluate a single configuration. Returns aggregated metrics dict."""
    from tqdm import tqdm

    per_file = []
    num_skipped = 0

    for text in tqdm(texts, desc=f"  {label}", unit="file"):
        split = split_prefix_suffix(
            text, tokenizer, prefix_ratio, min_tokens, max_prefix_tokens,
        )
        if split is None:
            num_skipped += 1
            continue

        prefix_text, suffix_text, prefix_ids, suffix_ids = split

        result = generate_continuation(model, tokenizer, prefix_text, gen_tokens)
        if result is None:
            num_skipped += 1
            continue

        gen_text, gen_ids = result

        # Decode reference from same number of tokens as generated
        ref_ids_trimmed = suffix_ids[:len(gen_ids)]
        ref_text_raw = tokenizer.decode(ref_ids_trimmed, skip_special_tokens=False)

        # Decompress both sides for text comparison (Config B has macros)
        ref_text = decompressor.decompress(ref_text_raw) if decompressor else ref_text_raw
        gen_text_dec = decompressor.decompress(gen_text) if decompressor else gen_text

        per_file.append({
            "exact_match": compute_exact_match(gen_text_dec, ref_text),
            "edit_similarity": compute_edit_similarity(gen_text_dec, ref_text),
            "bleu4": compute_bleu4(ref_text, gen_text_dec),
            "chrf": compute_chrf(ref_text, gen_text_dec),
            "gen_tokens": len(gen_ids),
            "ref_tokens": len(ref_ids_trimmed),
        })

    if not per_file:
        return {"num_files": 0, "num_skipped": num_skipped}

    def _agg(key):
        values = [f[key] for f in per_file]
        return {
            "mean": round(statistics.mean(values), 4),
            "median": round(statistics.median(values), 4),
            "std": round(statistics.stdev(values), 4) if len(values) > 1 else 0.0,
        }

    return {
        "exact_match": _agg("exact_match"),
        "edit_similarity": _agg("edit_similarity"),
        "bleu4": _agg("bleu4"),
        "chrf": _agg("chrf"),
        "num_files": len(per_file),
        "num_skipped": num_skipped,
    }


# ---------------------------------------------------------------------------
# Results display and persistence
# ---------------------------------------------------------------------------


METRICS = ["exact_match", "edit_similarity", "bleu4", "chrf"]

METRIC_LABELS = {
    "exact_match": "Exact Match (EM)",
    "edit_similarity": "Edit Similarity (ES)",
    "bleu4": "BLEU-4",
    "chrf": "chrF",
}


def build_comparison(results):
    """Compute B vs A deltas."""
    if "A" not in results or "B" not in results:
        return {}
    if results["A"]["num_files"] == 0 or results["B"]["num_files"] == 0:
        return {}

    comp = {}
    for metric in METRICS:
        a_val = results["A"][metric]["mean"]
        b_val = results["B"][metric]["mean"]
        delta_pct = round((b_val - a_val) / a_val * 100, 1) if a_val != 0 else 0.0
        comp[metric] = {"A": a_val, "B": b_val, "delta_pct": delta_pct}
    return comp


def print_results_table(results, comparison):
    """Print ASCII results table."""
    print()
    print("=" * 76)
    print("Sematok Correctness Evaluation (Step 7b)")
    print("=" * 76)
    header = f"{'Config':<9}{'Model':<13}{'Input':<15}{'EM':>6}{'ES':>8}{'BLEU-4':>8}{'chrF':>8}{'Files':>7}"
    print(header)
    print("-" * 76)

    config_meta = {
        "A": ("Base", "Uncompressed"),
        "B": ("Finetuned", "Compressed"),
    }
    for cfg in ("A", "B"):
        r = results.get(cfg)
        if not r or r["num_files"] == 0:
            continue
        model_name, input_type = config_meta[cfg]
        em = r["exact_match"]["mean"]
        es = r["edit_similarity"]["mean"]
        bleu = r["bleu4"]["mean"]
        chrf = r["chrf"]["mean"]
        print(f"{cfg:<9}{model_name:<13}{input_type:<15}{em:>6.4f}{es:>8.4f}{bleu:>8.4f}{chrf:>8.4f}{r['num_files']:>7}")

    if comparison:
        print()
        print("-" * 76)
        print("Analysis")
        print("-" * 76)
        for metric in METRICS:
            label = METRIC_LABELS[metric]
            c = comparison[metric]
            print(f"  {label:<24s}(B vs A): {c['B']:.4f} vs {c['A']:.4f}  ({c['delta_pct']:+.1f}%)")

    print("=" * 76)


def save_results(results, comparison, output_path, args):
    """Save results to JSON."""
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    data = {
        "timestamp": datetime.now().isoformat(),
        "params": {
            "max_seq_length": args.max_seq_length,
            "gen_tokens": args.gen_tokens,
            "prefix_ratio": args.prefix_ratio,
            "min_tokens": args.min_tokens,
            "max_prefix_tokens": args.max_prefix_tokens,
            "max_files": args.max_files,
            "seed": args.seed,
        },
        "configs": results,
        "comparison": comparison,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved to {output_path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def run_correctness_eval(args):
    """Run configs A and B."""
    compressed_path = args.compressed_eval
    if not Path(compressed_path).exists():
        raise FileNotFoundError(f"Compressed eval not found: {compressed_path}")

    # Prepare data
    uncompressed_path = str(
        Path(compressed_path).parent / "eval_uncompressed.jsonl"
    )
    print("Preparing eval data...")
    generate_uncompressed_eval(compressed_path, args.dictionary, uncompressed_path)

    print("Loading eval texts...")
    compressed_texts = load_texts(compressed_path)
    uncompressed_texts = load_texts(uncompressed_path)

    # Subsample with shared indices
    if args.max_files > 0 and args.max_files < len(compressed_texts):
        rng = random.Random(args.seed)
        indices = sorted(rng.sample(range(len(compressed_texts)), args.max_files))
        compressed_texts = [compressed_texts[i] for i in indices]
        uncompressed_texts = [uncompressed_texts[i] for i in indices]

    n = len(compressed_texts)
    print(f"Evaluating {n} files, generating {args.gen_tokens} tokens each\n")

    # Load decompressor for Config B
    from sematok.decompressor import Decompressor
    from sematok.dictionary import CompressionDictionary

    dictionary = CompressionDictionary.load(args.dictionary)
    decompressor = Decompressor(dictionary)

    results = {}
    gen_kwargs = dict(
        gen_tokens=args.gen_tokens,
        prefix_ratio=args.prefix_ratio,
        min_tokens=args.min_tokens,
        max_prefix_tokens=args.max_prefix_tokens,
    )

    # --- Config A: Base model + uncompressed ---
    model, tokenizer = load_model_for_generation(
        args.base_model, args.max_seq_length, args.load_in_4bit,
    )
    print("\nConfig A: Base model + uncompressed C#")
    results["A"] = evaluate_config(
        model, tokenizer, uncompressed_texts,
        label="A (base+uncompressed)", decompressor=None, **gen_kwargs,
    )
    results["A"]["model"] = args.base_model
    results["A"]["eval_data"] = uncompressed_path
    _print_config_summary("A", results["A"])
    unload_model(model, tokenizer)

    # --- Config B: Finetuned model + compressed ---
    model, tokenizer = load_model_for_generation(
        args.finetuned_model, args.max_seq_length, args.load_in_4bit,
    )
    print("\nConfig B: Finetuned model + compressed C#")
    results["B"] = evaluate_config(
        model, tokenizer, compressed_texts,
        label="B (finetuned+compressed)", decompressor=decompressor, **gen_kwargs,
    )
    results["B"]["model"] = args.finetuned_model
    results["B"]["eval_data"] = compressed_path
    _print_config_summary("B", results["B"])
    unload_model(model, tokenizer)

    # --- Results ---
    comparison = build_comparison(results)
    print_results_table(results, comparison)
    save_results(results, comparison, args.output, args)


def _print_config_summary(label, result):
    if result["num_files"] == 0:
        print(f"  -> Config {label}: no files evaluated")
        return
    em = result["exact_match"]["mean"]
    es = result["edit_similarity"]["mean"]
    bleu = result["bleu4"]["mean"]
    chrf = result["chrf"]["mean"]
    print(f"  -> EM: {em:.4f}, ES: {es:.4f}, BLEU-4: {bleu:.4f}, chrF: {chrf:.4f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate functional correctness of sematok fine-tuning.",
    )
    parser.add_argument("--base-model", required=True, help="Path to base model")
    parser.add_argument("--finetuned-model", required=True, help="Path to finetuned model")
    parser.add_argument(
        "--compressed-eval", default="data/finetune/eval.jsonl",
        help="Compressed eval JSONL (default: data/finetune/eval.jsonl)",
    )
    parser.add_argument(
        "--dictionary", default="sematok/dictionary.json",
        help="Compression dictionary (default: sematok/dictionary.json)",
    )
    parser.add_argument(
        "--output", default="out/correctness_results.json",
        help="Output JSON path (default: out/correctness_results.json)",
    )
    parser.add_argument("--max-files", type=int, default=200, help="Files to evaluate (default: 200)")
    parser.add_argument("--gen-tokens", type=int, default=128, help="Tokens to generate (default: 128)")
    parser.add_argument("--prefix-ratio", type=float, default=0.5, help="Prefix ratio (default: 0.5)")
    parser.add_argument("--min-tokens", type=int, default=64, help="Min tokens to include file (default: 64)")
    parser.add_argument("--max-prefix-tokens", type=int, default=512, help="Max prefix tokens (default: 512)")
    parser.add_argument("--max-seq-length", type=int, default=2048, help="Max sequence length (default: 2048)")
    parser.add_argument("--load-in-4bit", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")

    args = parser.parse_args()

    start = time.time()
    run_correctness_eval(args)
    elapsed = time.time() - start
    h, m, s = int(elapsed // 3600), int(elapsed % 3600 // 60), int(elapsed % 60)
    print(f"\nTotal time: {h}h {m}m {s}s")


if __name__ == "__main__":
    main()
