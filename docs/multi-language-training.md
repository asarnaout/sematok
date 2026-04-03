# Multi-Language Training

To train a single model that handles multiple languages:

1. Mine each language independently (follow Step 2 in the README for each):
   ```bash
   python -m sematok.mining --language csharp --corpus data/raw_csharp --auto
   python -m sematok.mining --language python --corpus data/raw_python --auto
   ```

2. Merge dictionaries (assigns non-overlapping macro IDs):
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

4. Expand tokenizer with `--no-distill` (distillation requires a single-language corpus; the embedding warmup in Step 5.5 handles multi-language):
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
