# Experiment Results

## Mining

| Language | Corpus Files | Exact | Templates | Total | Auto --min-files | --min-repos |
|----------|-------------|-------|-----------|-------|-----------------|-------------|
| C# | 207,570 | 2,880 | 172 | 3,052 | 200 | 2 |
| Python | | | | | | |
| Java | | | | | | |
| TypeScript | | | | | | |
| Go | 51,459 | 850 | 56 | 906 | 200 | 2 |

## Compression

Measured with `sematok.measure` (2000-file sample, Qwen tokenizer).

| Language | Original Tokens | Exact Saves | Template Saves | Total Saved | Ratio | Avg Macros/File |
|----------|----------------|-------------|----------------|-------------|-------|-----------------|
| C# | 1,981,354 | 232,139 | 4,557 | 236,696 | 11.95% | 41.8 |
| Python | | | | | | |
| Java | | | | | | |
| TypeScript | | | | | | |
| Go | 3,262,635 | 299,801 | 3,040 | 302,841 | 9.28% | 54.8 |

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
