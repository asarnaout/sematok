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
