# Experiment Results

Base model: `Qwen/Qwen2.5-Coder-1.5B-Instruct`

## Mining

| Language | Corpus Files | Exact | Templates | Total | Auto --min-files | --min-repos |
|----------|-------------|-------|-----------|-------|-----------------|-------------|
| C# | 207,570 | 2,880 | 172 | 3,052 | 200 | 2 |
| Python | | | | | | |
| Java | | | | | | |
| TypeScript | | | | | | |
| Go | | | | | | |

## Compression

Measured with `sematok.measure` (2000-file sample, Qwen tokenizer).

| Language | Original Tokens | Exact Saves | Template Saves | Total Saved | Ratio | Avg Macros/File |
|----------|----------------|-------------|----------------|-------------|-------|-----------------|
| C# | 1,981,354 | 232,139 | 4,557 | 236,696 | 11.95% | 41.8 |
| Python | | | | | | |
| Java | | | | | | |
| TypeScript | | | | | | |
| Go | | | | | | |

## Training

| Language | Distill Loss | Warmup Loss | Fine-Tune Loss | Epochs |
|----------|-------------|-------------|----------------|--------|
| C# | | | | |
| Python | | | | |
| Java | | | | |
| TypeScript | | | | |
| Go | | | | |

## Evaluation

| Language | Base PPL | Fine-Tuned PPL | PPL Change | Correctness |
|----------|---------|----------------|------------|-------------|
| C# | | | | |
| Python | | | | |
| Java | | | | |
| TypeScript | | | | |
| Go | | | | |
