# Experiment Results

## Mining

| Language | Corpus Files | Exact | Templates | Total | Auto --min-files | --min-repos |
|----------|-------------|-------|-----------|-------|-----------------|-------------|
| C# | 207,851 | 2,567 | 150 | 2,717 | 200 | 2 |
| Python | 67,789 | 3,909 | 334 | 4,243 | 50 | 2 |
| Java | | | | | | |
| TypeScript | 171,541 | 3,115 | 83 | 3,198 | 100 | 2 |
| Go | 51,459 | 850 | 56 | 906 | 200 | 2 |

## Compression

Measured with `sematok.measure` (2000-file sample, Qwen tokenizer).

| Language | Original Tokens | Exact Saves | Template Saves | Total Saved | Ratio | Avg Macros/File |
|----------|----------------|-------------|----------------|-------------|-------|-----------------|
| C# | 1,981,354 | 233,744 | 3,616 | 237,360 | 11.98% | 42.0 |
| Python | 3,203,562 | 262,078 | 5,039 | 267,117 | 8.34% | 48.0 |
| Java | | | | | | |
| TypeScript | 1,416,011 | 138,067 | 825 | 138,892 | 9.81% | 28.4 |
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
