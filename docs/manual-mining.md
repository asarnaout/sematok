# Manual Mining Workflow

This guide covers the full manual mining workflow with explicit control
over all thresholds. For the simple path, use `--auto` (see README).

## Overview

Three parameters control what ends up in the final dictionary:

- **`--min-files`** -- minimum number of files a pattern must appear in
  (filters rare patterns)
- **`--min-repos`** -- minimum number of repos a pattern must appear in
  (filters project-specific patterns). Affects the mining phase, so
  changing it requires a full re-mine.
- **`--max-entries`** -- hard cap on dictionary size (limits how many new
  tokens the model needs to learn). This is a training capacity
  constraint, not a quality filter.

All three values are derived from the data, not guessed upfront.

## Step 1: Analysis Mine

Run a full mine with relaxed thresholds. This mines broadly and scores
every candidate against the full corpus. Use `--scores-output` to save
the scoring data for later re-filtering:

```bash
python -m sematok.mining \
    --corpus data/raw_csharp \
    --language csharp \
    --output out/analysis_csharp.json \
    --min-files 1 \
    --min-repos 0 \
    --max-entries 0 \
    --scores-output out/analysis_csharp_scores.json
```

This is the slow step (hours). It produces:
- `out/analysis_csharp.json` -- unfiltered dictionary with all candidates
- `out/analysis_csharp_scores.json` -- per-entry scores, file counts, and
  repo counts (used by `--refilter` to avoid re-mining)
- A `--min-files threshold analysis` table printed to stdout (shows how
  many entries and what % of impact survives at each threshold)

## Step 2: Choose `--min-repos`

Run `repo_distribution` on the analysis dictionary. This shows how many
entries survive at each `--min-repos` threshold:

```bash
python -m sematok.repo_distribution \
    --language csharp \
    --corpus data/raw_csharp \
    --dictionary out/analysis_csharp.json
```

## Step 3: Final Mine

Mine the final dictionary with your chosen `--min-files` and `--min-repos`:

```bash
python -m sematok.mining \
    --corpus data/raw_csharp \
    --language csharp \
    --output sematok/languages/csharp/dictionary.json \
    --min-files <chosen> \
    --min-repos <chosen> \
    --scores-output sematok/languages/csharp/dictionary_scores.json
```

Check how many entries survived. If the count is too high for training,
add `--max-entries N` to cap it, or raise `--min-files`.

## Re-filtering Without Re-mining

After a full mine, you can adjust `--min-files` and `--max-entries`
instantly using saved scoring data (no re-mining or re-scoring needed):

```bash
python -m sematok.mining \
    --refilter sematok/languages/csharp/dictionary_scores.json \
    --output sematok/languages/csharp/dictionary.json \
    --language csharp \
    --min-files <new_value> \
    --max-entries <new_value>
```

Note: `--min-repos` cannot be changed via `--refilter` because it affects
the mining phase. Changing it requires a full re-mine.
