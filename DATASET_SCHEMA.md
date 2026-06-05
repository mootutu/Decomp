# Dataset Schema

The builder writes artifacts under `data/decomp/`.

## Raw Files

- `raw/aime24.json`
- `raw/aime25.json`

These files cache the original Hugging Face rows used by the builder.

## Flat Nodes

`processed_depth2/{dataset}_nodes.jsonl` contains one node per line. It is intentionally flat.
Use `--processed-subdir NAME` to write an alternate variant, such as
`processed_depth1/{dataset}_nodes.jsonl`, without replacing the default
`processed_depth2/` artifacts.

Fields:

- `node_id`: stable generated node id.
- `dataset`: `aime24` or `aime25`.
- `problem_id`: source problem id.
- `depth`: recursion depth, where original AIME problems have depth `0`.
- `problem`: original or generated problem text.
- `answer`: known or generated answer.
- `solution`: source or generated solution text, if available.
- `parent_node_id`: parent id, or `null` for root AIME problems.
- `concept_tags`: concept labels for the node.
- `summary`: decomposition or step summary.
- `depends_on`: sibling step ids the node depends on.
- `verification`: LLM verification object for generated subproblems, or `null`.
- `child_node_ids`: direct children in the decomposition tree.
- `difficulty`: simple structural difficulty features.

## Trees

`processed_depth2/{dataset}_trees.json` contains recursive root trees. Each tree uses the same node fields as JSONL and additionally includes nested `children`.

## Summary

`processed_depth2/{dataset}_summary.json` contains counts and the union of concept tags.

## Validation

Dry-run/root-only outputs:

```bash
uv run python validate_decomp_dataset.py
```

Full decomposed outputs:

```bash
uv run python validate_decomp_dataset.py --require-decomposed --require-verified
```

Alternate processed directory:

```bash
uv run python validate_decomp_dataset.py \
  --processed-subdir processed_depth1 \
  --require-decomposed \
  --require-verified \
  --expected-max-depth 1
```
