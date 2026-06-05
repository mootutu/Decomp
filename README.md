# Decomp

This project builds dataset-decomposition artifacts inspired by
`Learning to Solve Complex Problems via Dataset Decomposition`.

The current scope is data construction only. This repository does not include
model training, fine-tuning, or evaluation code.

- load AIME 2024 and AIME 2025 source problems;
- use AveMujicaAPI through the OpenAI-compatible SDK;
- use `gpt-5.5` by default;
- generate recursive simpler subproblems, concept tags, dependency hints, solutions, and verification records;
- write JSON/JSONL artifacts for later analysis or downstream training in a separate project.

## Setup

```bash
uv sync
cp .env.example .env
```

Edit `.env` and set `AVEMUJICA_API_KEY`. The CLI loads `.env` automatically.
Environment variables and explicit command-line flags still take precedence.

## Dry run

Fetch raw AIME data and write root-only outputs without calling the model:

```bash
uv run python build_decomp_dataset.py --dry-run
```

Outputs are written under `data/decomp/`:

- `raw/aime24.json`
- `raw/aime25.json`
- `processed_depth2/aime24_nodes.jsonl`
- `processed_depth2/aime25_nodes.jsonl`
- `processed_depth2/aime24_trees.json`
- `processed_depth2/aime25_trees.json`
- `processed_depth2/*_summary.json`

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

The full build should finish with AIME24 and AIME25 artifacts under
`data/decomp/processed_depth2/`. Because every LLM response is cached, rerunning the
same command resumes from cached calls instead of regenerating completed steps.

Useful flags:

- `--max-problems N`: limit problems per dataset while testing.
- `--max-depth N`: recursion depth for generated subproblems.
- `--processed-subdir NAME`: final artifact directory under `data/decomp/`.
- `--no-verify`: skip verification calls for generated subproblems.
- `--refresh-raw`: re-fetch AIME source data.
- `--refresh-llm`: ignore cached LLM responses and regenerate.
- `--sleep-seconds N`: pause between model calls.

LLM calls are cached in `data/decomp/llm_cache/`, so interrupted builds can be resumed.

## Build a depth-1 variant

To keep the default depth-2 artifacts in `data/decomp/processed_depth2/`, write the
depth-1 variant to a separate processed directory:

```bash
uv run python build_decomp_dataset.py \
  --datasets aime24 aime25 \
  --model gpt-5.5 \
  --max-depth 1 \
  --processed-subdir processed_depth1
```

## Validate Artifacts

For dry-run or root-only artifacts:

```bash
uv run python validate_decomp_dataset.py
```

After a full decomposed build:

```bash
uv run python validate_decomp_dataset.py --require-decomposed --require-verified
```

For the depth-1 variant:

```bash
uv run python validate_decomp_dataset.py \
  --processed-subdir processed_depth1 \
  --require-decomposed \
  --require-verified \
  --expected-max-depth 1
```

That command is the completion check for this repository's dataset-building
scope: it requires both datasets to have generated subproblem nodes and
verification records.
