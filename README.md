# Sematok

Semantic token compression for code LLMs.

Sematok replaces multi-token boilerplate patterns with single macro tokens, then fine-tunes a pre-trained model to understand them. At inference time, macros are expanded back to original code -- lossless and transparent to the end user.

```
public static void Main(string[] args)     -->  <|M026|>
{ get; set; }                              -->  <|M047|>
throw new ArgumentNullException(nameof(x)) -->  <|T005:x|>
```

Multiple BPE tokens become 1. Every occurrence, every file, across the entire context window.

## Why

BPE tokenizers are trained on a massive mix of languages and text. They optimize for the average and can never look at a 45-character boilerplate string and decide "this should be 1 token."

Every language has boilerplate -- access modifiers, import statements, type annotations, decorator patterns -- that consumes tokens carrying zero reasoning value. Sematok reclaims that capacity.

## Quick Start

```python
from sematok import Compressor, Decompressor, CompressionDictionary, get_safe_ranges

# Load the shipped C# dictionary
d = CompressionDictionary.load("sematok/languages/csharp/dictionary.json")

# Compress
compressor = Compressor(d, language="csharp")
safe_ranges = get_safe_ranges(source_code)
compressed = compressor.compress(source_code, safe_ranges=safe_ranges)

# Decompress (lossless)
decompressor = Decompressor(d)
original = decompressor.decompress(compressed)
assert original == source_code
```

Install dependencies:

```bash
pip install tree-sitter tree-sitter-c-sharp tqdm
```

For mining and training, also install:

```bash
pip install transformers torch unsloth peft bitsandbytes accelerate datasets
```

## Supported Languages

| Language | Status | Dictionary |
|----------|--------|------------|
| C# | Shipped | `sematok/languages/csharp/dictionary.json` |
| *Your language* | [Add it](#adding-a-language) | — |

## How It Works

### Compression

The compressor uses a greedy longest-match-first strategy with two passes:

1. **Exact macros** (`<|M...|>`): Direct string replacement, longest patterns first. Null-byte placeholders prevent delimiter collisions during the cascade.
2. **Template macros** (`<|T...:args|>`): Compiled regex patterns with identifier capture. Variables are normalized to positional slots (`{0}`, `{1}`).

Compression only happens in **safe zones** -- regions outside string literals, comments, and character literals, identified by tree-sitter parsing. XML doc comments (`///`) are optionally treated as safe.

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

## Adding a Language

1. Copy the template to a new package:
   ```bash
   cp sematok/languages/TEMPLATE.py sematok/languages/yourlang/__init__.py
   ```

2. Fill in the `LanguageConfig` fields. See `sematok/languages/csharp/__init__.py` for a complete example. You need:
   - A tree-sitter grammar (`pip install tree-sitter-<grammar>`)
   - Unsafe node types (comments, strings) for safe zone detection
   - Candidate patterns (regexes matching boilerplate in your language)
   - AST node types for subtree mining
   - Identifier normalization rules for template discovery
   - A list of repos to mine from

3. Register the language in `sematok/languages/__init__.py`:
   ```python
   _REGISTRY["yourlang"] = "sematok.languages.yourlang"
   ```

4. Mine a dictionary, prepare data, and train -- all pipeline commands accept `--language yourlang`.

## Training a Model

All training commands accept `--model` to specify any HuggingFace model. The default is `Qwen/Qwen2.5-Coder-1.5B-Instruct`. Token Distillation and LoRA target modules assume a LLaMA-family architecture (`model.model.layers`, `model.model.embed_tokens`). This covers Qwen, Mistral, LLaMA, and most recent open-weight code models.

Use `--target-modules` in the fine-tuning step to override LoRA targets for non-standard architectures.

### Step 1: Download Corpus

```bash
python -m data.download --language csharp --output data/raw
```

### Step 2: Mine Dictionary

First, analyze your corpus to choose mining parameters:

```bash
# See how many entries survive at each --min-repos threshold
python -m sematok.repo_distribution --language csharp --corpus data/raw

# Run mining with --min-files 0 to see the threshold analysis table
python -m sematok.mining \
    --corpus data/raw \
    --language csharp \
    --output /dev/null \
    --min-files 0 \
    --min-repos 2
```

Then mine the dictionary with your chosen parameters:

```bash
python -m sematok.mining \
    --corpus data/raw \
    --language csharp \
    --output sematok/languages/csharp/dictionary.json \
    --exclude-repos microsoft--garnet kgrzybek--modular-monolith-with-ddd \
        nunit--nunit JamesNK--Newtonsoft.Json AngleSharp--AngleSharp \
        ThreeMammals--Ocelot fullstackhero--dotnet-starter-kit \
    --min-files 500 \
    --min-repos 2
```

`--max-entries N` sets a hard upper limit on dictionary size (default: 999). The 5-digit macro ID format (`<|M00001|>` through `<|M99999|>`) supports up to 99,999 exact macros and 99,999 templates.

### Step 3: Measure Compression

```bash
python -m sematok.measure --corpus data/raw --language csharp
```

### Step 4: Prepare Training Data

```bash
python -m data.prepare --corpus data/raw --language csharp --output data/finetune
```

Produces `train.jsonl` and `eval.jsonl`. Training data uses a 75/25 compressed/original mix (configurable via `--compress-ratio`). Eval data is 100% compressed.

### Step 5: Expand Tokenizer

```bash
python -m training.expand_tokenizer \
    --model Qwen/Qwen2.5-Coder-1.5B-Instruct \
    --output models/sematok-base \
    --corpus data/raw \
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

LoRA continued pre-training (CLM). QLoRA 4-bit, rank 16, 1 epoch.

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

## Multi-Language Training

To train a single model that handles multiple languages:

1. Mine each language independently:
   ```bash
   python -m sematok.mining --language csharp --corpus data/raw_cs \
       --output sematok/languages/csharp/dictionary.json --min-files 500
   python -m sematok.mining --language python --corpus data/raw_py \
       --output sematok/languages/python/dictionary.json --min-files 500
   ```

2. Merge dictionaries (assigns non-overlapping macro IDs):
   ```bash
   python -m sematok.merge \
       sematok/languages/csharp/dictionary.json \
       sematok/languages/python/dictionary.json \
       --output merged_dictionary.json
   ```

3. Prepare training data per language, then concatenate:
   ```bash
   python -m data.prepare --language csharp --corpus data/raw_cs --output data/ft_cs
   python -m data.prepare --language python --corpus data/raw_py --output data/ft_py
   cat data/ft_cs/train.jsonl data/ft_py/train.jsonl > data/ft_mixed/train.jsonl
   ```

4. Expand tokenizer and train using the merged dictionary:
   ```bash
   python -m training.expand_tokenizer \
       --dictionary merged_dictionary.json \
       --output models/sematok-multi-base
   python -m training.fine_tune \
       --model models/sematok-multi-base \
       --train data/ft_mixed/train.jsonl \
       --output models/sematok-multi-finetuned
   ```

Each language's macros use distinct ID ranges, so there are no collisions. The `--max-entries` default of 999 per language keeps individual dictionaries reasonable.

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
        csharp/
            __init__.py     # C# patterns, node types, repos
            dictionary.json # Shipped C# dictionary
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

## Running Tests

```bash
pytest
```

## Prior Art

- **Token Sugar** (ASE'25, [arxiv 2512.08266](https://arxiv.org/abs/2512.08266)) -- AST-based frequent subtree mining for Python, 799 patterns, trained from scratch on ~1B models. Demonstrated that reversible code compression can work for LLMs.
- **Meta-Tokens** ([arxiv 2506.00307](https://arxiv.org/abs/2506.00307)) -- Per-prompt lossless token sequence compression, greedy longest-first, LoRA fine-tuned. Validated the greedy compression strategy and showed LoRA can teach compressed representations.
- **Token Distillation** ([arxiv 2505.20133](https://arxiv.org/abs/2505.20133)) -- Embedding initialization for new tokens via hidden state distillation.
- **SimPy** (ISSTA'24, [arxiv 2404.16333](https://arxiv.org/abs/2404.16333)) -- AI-oriented Python grammar that rewrites syntax for token efficiency. Showed 8-13% reduction on code-specialized tokenizers.

## License

MIT
