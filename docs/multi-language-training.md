# Multi-Language Training

There are two approaches to training a model across multiple languages: a single merged model, or per-language LoRA adapters.

## Option A: Per-Language Adapters (recommended for many languages)

Train a separate LoRA adapter for each language. Each adapter has a smaller vocabulary expansion, which keeps fine-tuning stable and avoids degradation from adding too many new tokens at once.

1. Mine each language independently (follow Step 2 in the README for each):
   ```bash
   python -m sematok.mining --language csharp --corpus data/raw_csharp --auto
   python -m sematok.mining --language python --corpus data/raw_python --auto
   ```

2. Prepare training data per language using its own dictionary:
   ```bash
   python -m data.prepare --language csharp --corpus data/raw_csharp --output data/ft_cs
   python -m data.prepare --language python --corpus data/raw_python --output data/ft_py
   ```

3. Expand, warmup, and fine-tune each language separately:
   ```bash
   # C#
   python -m training.expand_tokenizer \
       --model Qwen/Qwen2.5-Coder-7B-Instruct \
       --corpus data/raw_csharp --language csharp \
       --output models/sematok-csharp-base
   python -m training.warmup_embeddings \
       --model models/sematok-csharp-base \
       --train data/ft_cs/train.jsonl
   python -m training.fine_tune \
       --model models/sematok-csharp-base \
       --train data/ft_cs/train.jsonl \
       --eval data/ft_cs/eval.jsonl \
       --output models/sematok-csharp-finetuned

   # Python
   python -m training.expand_tokenizer \
       --model Qwen/Qwen2.5-Coder-7B-Instruct \
       --corpus data/raw_python --language python \
       --output models/sematok-python-base
   python -m training.warmup_embeddings \
       --model models/sematok-python-base \
       --train data/ft_py/train.jsonl
   python -m training.fine_tune \
       --model models/sematok-python-base \
       --train data/ft_py/train.jsonl \
       --eval data/ft_py/eval.jsonl \
       --output models/sematok-python-finetuned
   ```

At inference time, load the adapter matching the input language. This is natural since the compressor already requires a language-specific dictionary.

**When to use this approach:** When the total macro count across all languages is large (thousands per language). Large vocabulary expansions can destabilize LoRA fine-tuning -- research suggests keeping new tokens under ~2,500-3,000 per adapter for reliable results with LoRA rank 16.

## Option B: Single Merged Model

Merge all dictionaries into one and train a single model. Simpler deployment, but the vocabulary expansion is larger.

1. Mine each language independently (follow Step 2 in the README for each):
   ```bash
   python -m sematok.mining --language csharp --corpus data/raw_csharp --auto
   python -m sematok.mining --language python --corpus data/raw_python --auto
   ```

2. Merge dictionaries (assigns non-overlapping macro IDs, deduplicates shared patterns):
   ```bash
   python -m sematok.merge \
       sematok/languages/csharp/dictionary.json \
       sematok/languages/python/dictionary.json \
       --output merged_dictionary.json
   ```

3. Prepare training data per language using the **merged** dictionary, then concatenate. This is important -- the merged dictionary has different IDs than the per-language ones, and the training data must match the tokenizer:
   ```bash
   python -m data.prepare --language csharp --corpus data/raw_csharp \
       --dictionary merged_dictionary.json --output data/ft_cs
   python -m data.prepare --language python --corpus data/raw_python \
       --dictionary merged_dictionary.json --output data/ft_py
   cat data/ft_cs/train.jsonl data/ft_py/train.jsonl > data/ft_mixed/train.jsonl
   cat data/ft_cs/eval.jsonl data/ft_py/eval.jsonl > data/ft_mixed/eval.jsonl
   ```

4. Expand tokenizer with `--no-distill` (distillation requires a single-language corpus; the embedding warmup in Step 4.5 handles multi-language):
   ```bash
   python -m training.expand_tokenizer \
       --dictionary merged_dictionary.json \
       --output models/sematok-multi-base \
       --no-distill
   ```

5. Warmup and fine-tune on the mixed data:
   ```bash
   python -m training.warmup_embeddings \
       --model models/sematok-multi-base \
       --train data/ft_mixed/train.jsonl
   python -m training.fine_tune \
       --model models/sematok-multi-base \
       --train data/ft_mixed/train.jsonl \
       --eval data/ft_mixed/eval.jsonl \
       --output models/sematok-multi-finetuned
   ```

Each language's macros use distinct ID ranges, so there are no collisions.

**When to use this approach:** When total macro count is small (e.g., 2-3 languages with trimmed dictionaries), or when using full continual pretraining instead of LoRA. If using LoRA, consider trimming each dictionary with `--max-entries` to keep the total under ~3,000 tokens.
