# Sematok

Semantic token compression for code LLMs.

Sematok replaces multi-token boilerplate patterns with single macro tokens, then fine-tunes a pre-trained model to understand them. At inference time, macros are expanded back to original code -- lossless and transparent to the end user.

```
public static void Main(string[] args)     -->  <|M026|>
{ get; set; }                              -->  <|M047|>
throw new ArgumentNullException(nameof(x)) -->  <|T005:x|>
```

Multiple BPE tokens collapse to 1. The saved context goes to code that actually matters.

## Why

BPE tokenizers have a fixed vocabulary budget shared across every language and domain in their training data. A 45-character C# boilerplate string will never outcompete cross-domain patterns for a vocabulary slot, no matter how often it appears in code.

Every language has boilerplate -- access modifiers, import statements, type annotations, decorator patterns -- that consumes tokens carrying zero reasoning value. Sematok reclaims that capacity.

## Supported Languages

| Language | Dictionary |
|----------|------------|
| C# | `sematok/languages/csharp/dictionary.json` |
| Python | `sematok/languages/python/dictionary.json` |
| Java | `sematok/languages/java/dictionary.json` |
| TypeScript | `sematok/languages/typescript/dictionary.json` |
| Go | `sematok/languages/go/dictionary.json` |
| *Your language* | [Add it](docs/adding-a-language.md) |

## Training a Model

Install dependencies:

```bash
pip install tree-sitter tree-sitter-c-sharp tree-sitter-python tree-sitter-java tree-sitter-typescript tree-sitter-go tqdm
pip install transformers torch unsloth peft bitsandbytes accelerate datasets
```

### Step 1: Download Corpus

```bash
python -m data.download --language csharp
```

### Step 2: Mine Dictionary

```bash
python -m sematok.mining --corpus data/raw_csharp --language csharp --auto
```

`--auto` mines the full corpus, scores every candidate, and automatically
selects quality thresholds:

- **`--min-files`** is auto-selected (highest threshold retaining ≥90% of
  total compression impact)
- **`--min-repos`** defaults to 2 (filters single-repo patterns). Override
  with `--auto --min-repos 3` if needed.

If the dictionary has too many entries for your model, trim it instantly
without re-mining:

```bash
python -m sematok.mining \
    --refilter sematok/languages/csharp/dictionary_scores.json \
    --output sematok/languages/csharp/dictionary.json \
    --language csharp \
    --max-entries 800
```

This uses scoring data saved during `--auto` to keep only the top N
entries by impact. You can also raise `--min-files` to be more selective.

For full manual control over all thresholds, see
[Manual Mining Workflow](docs/manual-mining.md).

### Step 3: Measure Compression

```bash
python -m sematok.measure --corpus data/raw_csharp --language csharp
```

### Step 4: Prepare Training Data

```bash
python -m data.prepare --corpus data/raw_csharp --language csharp --output data/finetune
```

Produces `train.jsonl` and `eval.jsonl`. Training data uses a 75/25 compressed/original mix (configurable via `--compress-ratio`). Eval data is 100% compressed.

### Step 5: Expand Tokenizer

```bash
python -m training.expand_tokenizer \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --output models/sematok-base \
    --corpus data/raw_csharp \
    --language csharp
```

Adds macro tokens and initializes embeddings via Token Distillation ([arXiv:2505.20133](https://arxiv.org/abs/2505.20133)). Use `--no-distill` for faster mean-of-expansion initialization.

### Step 5.5: Embedding Warmup

```bash
python -m training.warmup_embeddings \
    --model models/sematok-base \
    --train data/finetune/train.jsonl
```

Freezes the transformer body and trains only new macro token embedding rows for 2 epochs.

### Step 6: Fine-Tune

```bash
python -m training.fine_tune \
    --model models/sematok-base \
    --train data/finetune/train.jsonl \
    --eval data/finetune/eval.jsonl \
    --output models/sematok-finetuned
```

LoRA continued pre-training (CLM). QLoRA 4-bit, rank 16, 1 epoch. LoRA targets assume a LLaMA-family architecture (Qwen, Mistral, LLaMA). Use `--target-modules` to override for non-standard architectures.

### Step 7: Evaluate

```bash
python -m training.evaluate --all \
    --base-model models/sematok-base \
    --finetuned-model models/sematok-finetuned-merged

python -m training.evaluate_correctness \
    --base-model models/sematok-base \
    --finetuned-model models/sematok-finetuned-merged
```

Four-configuration perplexity comparison plus functional correctness (generation similarity). Use `--max-files 500` for a quick validation run.

For training a single model across multiple languages, see [Multi-Language Training](docs/multi-language-training.md).

## How It Works

### Compression

The compressor uses a greedy longest-match-first strategy with two passes:

1. **Exact macros** (`<|M...|>`): Direct string replacement, longest patterns first. Null-byte placeholders prevent delimiter collisions during the cascade.
2. **Template macros** (`<|T...:args|>`): Compiled regex patterns with identifier capture. Variables are normalized to positional slots (`{0}`, `{1}`).

Compression only happens in **safe zones** -- regions outside string literals, comments, and character literals, identified by tree-sitter parsing. Languages can override specific comment types as safe (e.g., C# XML doc comments `///`, TypeScript JSDoc `/** ... */`) to compress repetitive documentation boilerplate.

Decompression is lossless: `decompress(compress(source)) == source` for all inputs.

### Mining Pipeline

The dictionary is built from a corpus of source files:

1. **Candidate extraction** -- regex patterns, AST subtrees, and n-gram frequency analysis surface recurring boilerplate
2. **Corpus scoring** -- each candidate is scored by actual BPE token savings across all training files
3. **Filtering** -- minimum file-frequency (`--min-files`) and repo-diversity (`--min-repos`) thresholds eliminate rare or project-specific patterns
4. **Template generalization** -- AST-guided identifier normalization discovers parameterized patterns (e.g., `throw new ArgumentNullException(nameof({0}))`)

### Training Pipeline

1. **Expand tokenizer** -- add macro tokens to the base model's vocabulary
2. **Initialize embeddings** -- Token Distillation optimizes new embeddings so early-layer hidden states match the original sub-tokens
3. **Warmup** -- freeze the transformer body, train only new embedding rows
4. **Fine-tune** -- LoRA continued pre-training on a mix of compressed and original code
5. **Evaluate** -- perplexity comparison and functional correctness tests

## API Reference

```python
from sematok import (
    CompressionDictionary,  # Load/save pattern <-> macro mappings
    Compressor,             # Compress source code using a dictionary
    Decompressor,           # Lossless macro expansion
    LanguageConfig,         # Language-specific configuration dataclass
    get_language,           # Load a registered language config by name
    available_languages,    # List registered language names
    get_dictionary_path,    # Path to shipped dictionary for a language
    get_safe_ranges,        # Tree-sitter safe zone detection
)
```

## Project Structure

```
sematok/                    # Compression engine (language-agnostic)
    __init__.py             # Public API
    languages/
        __init__.py         # Language registry + get_dictionary_path()
        TEMPLATE.py         # Skeleton for adding a new language
        csharp/             # C# config + shipped dictionary
        python/             # Python config
        java/               # Java config
        typescript/         # TypeScript config
        go/                 # Go config
    dictionary.py           # Pattern <-> macro mapping
    compressor.py           # Greedy compression with safe zones
    decompressor.py         # Lossless macro expansion
    lexer.py                # Tree-sitter safe zone detection
    mining.py               # Pattern mining with corpus impact scoring
    template_mining.py      # AST-guided template discovery
    ast_mining.py           # Full AST subtree mining
    ngram_mining.py         # N-gram substring frequency mining
    repo_distribution.py    # Per-repo diversity analysis
    measure.py              # Compression ratio measurement
    merge.py                # Merge per-language dictionaries
data/
    download.py             # Corpus download from GitHub
    prepare.py              # JSONL generation for fine-tuning
training/
    expand_tokenizer.py     # Vocabulary expansion + embedding init
    warmup_embeddings.py    # Embedding-only warmup phase
    fine_tune.py            # LoRA fine-tuning
    evaluate.py             # Perplexity evaluation
    evaluate_correctness.py # Functional correctness (generation test)
    vocab_utils.py          # Vocabulary validation utilities
tests/                      # 96 tests
```

## Additional Documentation

- [Adding a Language](docs/adding-a-language.md) — How to add tree-sitter support for a new language
- [Manual Mining Workflow](docs/manual-mining.md) — Step-by-step mining with full control over `--min-files`, `--min-repos`, and `--max-entries`
- [Multi-Language Training](docs/multi-language-training.md) — Train a single model across multiple languages

## Prior Art

- **Token Sugar** (ASE'25, [arxiv 2512.08266](https://arxiv.org/abs/2512.08266)) -- AST-based frequent subtree mining for Python, 799 patterns, trained from scratch on ~1B models. Demonstrated that reversible code compression can work for LLMs.
- **Meta-Tokens** ([arxiv 2506.00307](https://arxiv.org/abs/2506.00307)) -- Per-prompt lossless token sequence compression, greedy longest-first, LoRA fine-tuned. Validated the greedy compression strategy and showed LoRA can teach compressed representations.
- **Token Distillation** ([arxiv 2505.20133](https://arxiv.org/abs/2505.20133)) -- Embedding initialization for new tokens via hidden state distillation.
- **SimPy** (ISSTA'24, [arxiv 2404.16333](https://arxiv.org/abs/2404.16333)) -- AI-oriented Python grammar that rewrites syntax for token efficiency. Showed 8-13% reduction on code-specialized tokenizers.

## License

MIT
