# Sematok

Semantic token compression for C# code in LLMs.

Sematok replaces multi-token C# boilerplate patterns with single macro tokens, then fine-tunes an existing pre-trained model to understand them. At inference time, macros are expanded back to original code -- lossless, transparent to the end user.

```
public static void Main(string[] args)    -->  <|M026|>
{ get; set; }                             -->  <|M047|>
throw new ArgumentNullException(nameof(x)) --> <|T005:x|>
```

8 Qwen tokens become 1. Every occurrence, every file, across the entire context window.

## The Problem

BPE tokenizers (used by all major LLMs) are trained on a massive mix of languages and text, so they optimize for the average and can't specialize deeply in any one language's patterns. They compress at the subword level but can never look at a 45-character boilerplate string and decide "this should be 1 token."

C# has more boilerplate than most languages -- access modifiers, property accessors, using directives, XML doc comments, exception patterns -- all consuming tokens that carry zero reasoning value.

## The Approach

1. **Mine** recurring boilerplate patterns from a large C# corpus (122K files, 44 permissively-licensed repos)
2. **Build** a dictionary of macro tokens scored on the full training corpus, filtered by minimum file-frequency and repo-diversity thresholds
3. **Compress** training data by replacing patterns with macros (75% compressed, 25% original)
4. **Expand** the model's tokenizer with the new tokens
5. **Fine-tune** the model via LoRA so it learns the macros without forgetting its other capabilities
6. **Evaluate** both macro comprehension and capability retention

The macro layer works as pre/post-processing -- the model's original tokenizer stays untouched in production. Deployment requires fine-tuning (cheap), not full retraining.

## Reproducing the Results

### Prerequisites

- Python 3.12+
- CUDA-capable GPU (tested on RTX 4060 8GB)
- ~10 GB disk space (corpus + model)

### Setup

```bash
git clone https://github.com/asarnaout/sematok.git
cd sematok
pip install -r requirements.txt
```

For fine-tuning (Step 6), also install:
```bash
pip install unsloth peft bitsandbytes accelerate
```

### Step 1: Download Corpus

Clone 44 permissively-licensed C# repos (MIT, Apache-2.0, BSD-3-Clause) and extract source files:

```bash
python -m data.download --output data/raw_cs
```

This produces ~122K `.cs` files (592 MB) with a `metadata.jsonl` mapping each file to its source repo.

### Step 2a: Choose Mining Parameters

Mining requires two parameters that depend on your corpus. Both have analysis tools that help you pick the right values.

**`--min-repos`** — minimum number of distinct source repos a pattern must appear in. Filters out project-internal patterns (e.g., `PlatformDetection` from dotnet/runtime) while keeping domain-specific patterns used across multiple projects. Run the repo distribution analysis on your corpus:

```bash
python -m sematok.repo_distribution
```

This prints a threshold table showing how many entries survive at each value. We chose `--min-repos 2`.

**`--min-files`** — minimum number of training files a pattern must appear in (after compression). Controls dictionary size and ensures every macro gets enough training exposure. The mining pipeline prints a threshold analysis table at the end of every run, so run mining once with `--min-files 0` to see the full picture:

```bash
python -m sematok.mining \
    --corpus data/raw_cs \
    --output /dev/null \
    --exclude-repos microsoft--garnet kgrzybek--modular-monolith-with-ddd \
        nunit--nunit JamesNK--Newtonsoft.Json AngleSharp--AngleSharp \
        ThreeMammals--Ocelot fullstackhero--dotnet-starter-kit \
    --min-files 0 \
    --min-repos 2
```

The output includes:

```
--min-files threshold analysis (use this to choose --min-files):
  Threshold    Entries    Total impact   % of impact
  ------------ ---------- -------------- ------------
  >= 0         11186           482,103       100.0%
  >= 100       1094            481,136        99.8%
  >= 500       868             469,159        97.3%
  >= 1000      600             422,764        87.7%
  >= 5000      71              155,824        32.3%
```

We chose `--min-files 500` because 868 entries capture 97.3% of total compression -- adding more yields diminishing returns, and every macro appears in enough training files for the model to learn it.

### Step 2b: Mine Dictionary

The dictionary (`sematok/dictionary.json`) is already committed. To re-mine from scratch using the parameters chosen above:

```bash
python -m sematok.mining \
    --corpus data/raw_cs \
    --output sematok/dictionary.json \
    --exclude-repos microsoft--garnet kgrzybek--modular-monolith-with-ddd \
        nunit--nunit JamesNK--Newtonsoft.Json AngleSharp--AngleSharp \
        ThreeMammals--Ocelot fullstackhero--dotnet-starter-kit \
    --min-files 500 \
    --min-repos 2
```

This mines ~11K candidates, scores each by actual Qwen token savings on **all** training files (116K), and keeps entries that pass both filters. The seven excluded repos are held out for evaluation.

`--max-entries N` sets a hard upper limit on dictionary size (default: 999). The 3-digit macro ID format (`<|M001|>` through `<|M999|>`) enforces an absolute ceiling of 999 exact macros and 999 templates. If any limit is hit, mining prints a warning showing how many eligible entries were dropped.

### Step 3: Measure Compression

```bash
python -m sematok.measure --corpus data/raw_cs --dictionary sematok/dictionary.json
```

### Step 4: Prepare Training Data

```bash
python -m data.prepare --corpus data/raw_cs --output data/finetune
```

Produces `train.jsonl` and `eval.jsonl`. Training data uses a 75/25 compressed/original mix. Eval data is 100% compressed. Split is repo-balanced: train on 37 repos (~116K files), evaluate on 7 held-out repos (~6.3K files).

### Step 5: Expand Tokenizer

```bash
python -m training.expand_tokenizer --output models/qwen-sematok-base
```

Adds 1000 tokens to Qwen2.5-Coder-1.5B-Instruct and initializes their embeddings via mean-of-expansion (averages the embeddings of the tokens each macro expands to).

### Step 6: Fine-Tune

```bash
python -m training.fine_tune \
    --model models/qwen-sematok-base \
    --train data/finetune/train.jsonl \
    --eval data/finetune/eval.jsonl \
    --output models/qwen-sematok-finetuned
```

LoRA fine-tuning with Unsloth. Continued pre-training (CLM) -- not instruction tuning. QLoRA 4-bit, rank 16, 1 epoch (~5,592 steps).

### Step 7: Evaluate

```bash
python -m training.evaluate --all \
    --base-model models/qwen-sematok-base \
    --finetuned-model models/qwen-sematok-finetuned-merged
```

Four-configuration perplexity comparison measuring macro comprehension and capability retention. Use `--max-files 500` for a quick validation run (~6 min).

### Step 7b: Functional Correctness

```bash
python -m training.evaluate_correctness \
    --base-model models/qwen-sematok-base \
    --finetuned-model models/qwen-sematok-finetuned-merged
```

Splits eval files into prefix and suffix, generates continuations, and measures similarity to ground truth. Compares base model on uncompressed C# vs finetuned model on compressed C#. Use `--max-files 50 --gen-tokens 64` for a quick run.

### Running Tests

```bash
pytest
```

96 tests covering the compression engine, mining pipeline, and data preparation.

## How Compression Works

The compressor uses a greedy longest-match-first strategy with two passes:

1. **Exact macros** (`<|M...|>`): Direct string replacement, longest patterns first. Null-byte placeholders prevent delimiter collisions during the cascade.
2. **Template macros** (`<|T...:args|>`): Compiled regex patterns with identifier capture. Variables are normalized to positional slots (`{0}`, `{1}`).

Compression only happens in **safe zones** -- regions outside string literals, comments, and character literals, identified by tree-sitter parsing. XML doc comments (`///`) are optionally treated as safe.

Decompression is lossless: `decompress(compress(source)) == source` for all inputs.

## Project Structure

```
sematok/                    # Compression engine (language-agnostic)
    languages/              # Language-specific configurations
        csharp.py           # C# patterns, node types, repos
    dictionary.py           # Pattern <-> macro mapping
    compressor.py           # Greedy compression with safe zones
    decompressor.py         # Lossless macro expansion
    lexer.py                # Tree-sitter safe zone detection
    mining.py               # Pattern mining with corpus impact scoring
    repo_distribution.py    # Per-repo diversity analysis for --min-repos tuning
    ngram_mining.py         # N-gram substring frequency mining
    template_mining.py      # AST-guided template discovery
    ast_mining.py           # Full AST subtree mining
    measure.py              # Compression ratio measurement
    dictionary.json         # The dictionary (tracked in git)
data/
    download.py             # Corpus download from GitHub
    prepare.py              # JSONL generation for fine-tuning
training/
    expand_tokenizer.py     # Vocabulary expansion + embedding init
    fine_tune.py            # LoRA fine-tuning with Unsloth
    evaluate.py             # Four-config perplexity evaluation
    evaluate_correctness.py # Functional correctness (generation test)
tests/                      # 96 tests
```

The engine is language-agnostic. All C#-specific knowledge lives in `sematok/languages/csharp.py`. To add a new language, create a config file and install the tree-sitter grammar.

## Base Model

**Qwen2.5-Coder-1.5B-Instruct** -- code-specialized, 152K vocab, proven on RTX 4060 8GB. No special token conflicts with our `<|M###|>` / `<|T###:...|>` format.

## Prior Art

This work builds on ideas from several research efforts:

- **Token Sugar** (ASE'25, [arxiv 2512.08266](https://arxiv.org/abs/2512.08266)) -- AST-based frequent subtree mining for Python, 799 patterns, trained from scratch on ~1B models. Demonstrated that reversible code compression can work for LLMs.
- **Meta-Tokens** ([arxiv 2506.00307](https://arxiv.org/abs/2506.00307)) -- Per-prompt lossless token sequence compression, greedy longest-first, LoRA fine-tuned. Validated the greedy compression strategy and showed LoRA can teach compressed representations.
- **Token Distillation** ([arxiv 2505.20133](https://arxiv.org/abs/2505.20133)) -- Embedding initialization for new tokens via hidden state distillation. Informed our mean-of-expansion embedding initialization approach.
- **SimPy** (ISSTA'24, [arxiv 2404.16333](https://arxiv.org/abs/2404.16333)) -- AI-oriented Python grammar that rewrites syntax for token efficiency. Showed 8-13% reduction on code-specialized tokenizers.

## License

MIT
