# Decomp

This project builds dataset-decomposition artifacts inspired by
`Learning to Solve Complex Problems via Dataset Decomposition`.

The current scope is data construction only:

- load AIME 2024 and AIME 2025 source problems;
- use AveMujicaAPI through the OpenAI-compatible SDK;
- use `gpt-5.5` by default;
- generate recursive simpler subproblems, concept tags, dependency hints, solutions, and verification records;
- write JSON/JSONL artifacts for later model training or analysis.

## Setup

```bash
uv sync
export AVEMUJICA_API_KEY="your_key_here"
```

## Dry run

Fetch raw AIME data and write root-only outputs without calling the model:

```bash
uv run python build_decomp_dataset.py --dry-run
```

Outputs are written under `data/decomp/`:

- `raw/aime24.json`
- `raw/aime25.json`
- `processed/aime24_nodes.jsonl`
- `processed/aime25_nodes.jsonl`
- `processed/aime24_trees.json`
- `processed/aime25_trees.json`
- `processed/*_summary.json`

See `DATASET_SCHEMA.md` for the exact artifact schema.

## Build a small sample

```bash
uv run python build_decomp_dataset.py \
  --datasets aime24 aime25 \
  --model gpt-5.5 \
  --max-problems 1 \
  --max-depth 1
```

## Build the full dataset

```bash
uv run python build_decomp_dataset.py \
  --datasets aime24 aime25 \
  --model gpt-5.5 \
  --max-depth 2
```

Useful flags:

- `--max-problems N`: limit problems per dataset while testing.
- `--max-depth N`: recursion depth for generated subproblems.
- `--no-verify`: skip verification calls for generated subproblems.
- `--refresh-raw`: re-fetch AIME source data.
- `--refresh-llm`: ignore cached LLM responses and regenerate.
- `--sleep-seconds N`: pause between model calls.

LLM calls are cached in `data/decomp/llm_cache/`, so interrupted builds can be resumed.

## Validate Artifacts

For dry-run or root-only artifacts:

```bash
uv run python validate_decomp_dataset.py
```

After a full decomposed build:

```bash
uv run python validate_decomp_dataset.py --require-decomposed --require-verified
```
