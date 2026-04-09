# Experiment Results

## Mining

| Language | Corpus Files | Exact | Templates | Total | Auto --min-files | --min-repos |
|----------|-------------|-------|-----------|-------|-----------------|-------------|
| C# | 144,210 | 2,413 | 105 | 2,518 | 200 | 2 |
| Python | 81,085 | 3,583 | 227 | 3,810 | 100 | 2 |
| Java | 229,232 | 2,705 | 233 | 2,938 | 200 | 2 |
| TypeScript | 132,153 | 2,956 | 86 | 3,042 | 100 | 2 |
| Go | 114,578 | 1,273 | 155 | 1,428 | 200 | 2 |

## Compression

Measured with `sematok.measure` (2000-file sample, Qwen tokenizer).

| Language | Original Tokens | Exact Saves | Template Saves | Total Saved | Ratio | Avg Macros/File |
|----------|----------------|-------------|----------------|-------------|-------|-----------------|
| C# | 3,071,439 | 292,903 | 4,028 | 296,931 | 9.67% | 53.9 |
| Python | 4,300,056 | 326,395 | 5,276 | 331,671 | 7.71% | 59.6 |
| Java | 2,579,839 | 292,386 | 4,824 | 297,210 | 11.52% | 48.1 |
| TypeScript | 2,074,408 | 181,117 | 1,102 | 182,219 | 8.78% | 37.5 |
| Go | 5,884,020 | 422,151 | 12,507 | 434,658 | 7.39% | 70.2 |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Base model | Qwen/Qwen2.5-Coder-7B-Instruct |
| Approach | Per-language LoRA adapters (one LoRA per language) |
| Embedding init | Token Distillation (arXiv:2505.20133), layer 4, 50 steps, 10 contexts |
| Embedding warmup | 1,500 steps, frozen transformer, LR 1e-3, cosine schedule |
| LoRA method | QLoRA 4-bit, rank 16, alpha 32, RSLoRA |
| LoRA targets | q/k/v/o_proj, gate/up/down_proj, embed_tokens, lm_head |
| LoRA dropout | 0 |
| Fine-tune LR | 2e-4 (embeddings: 2e-5) |
| Fine-tune schedule | Cosine, 500-step warmup |
| Batch size | 2 x 8 gradient accumulation = 16 effective |
| Epochs | 1 |
| Precision | float16 model, float32 distillation optimizer |
| Hardware | RunPod RTX 5090 1x, 32 GB VRAM |

## Evaluation

| Language | Base PPL | Fine-Tuned PPL | PPL Change | Correctness |
|----------|---------|----------------|------------|-------------|
| C# | | | | |
| Python | | | | |
| Java | | | | |
| TypeScript | | | | |
| Go | | | | |
