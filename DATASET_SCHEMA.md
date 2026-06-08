# Dataset Schema

The builder writes artifacts under per-dataset directories in `data/`.

## Raw Files

- `data/aime24/raw/aime24.json`
- `data/aime25/raw/aime25.json`

These files cache the original Hugging Face rows used by the builder.

## Flat Nodes

`data/{dataset}/processed_depth2/{dataset}_nodes.jsonl` contains one node per line. It is intentionally flat.
Use `--processed-subdir NAME` to write an alternate variant, such as
`data/{dataset}/processed_depth1/{dataset}_nodes.jsonl`, without replacing the default
`data/{dataset}/processed_depth2/` artifacts.

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
- `verification`: independent-solve verification object for generated subproblems, or `null`.
- `child_node_ids`: direct children in the decomposition tree.
- `difficulty`: concept-graph difficulty features:
  - `score`: `alpha_structural * structural_complexity + alpha_concept * conceptual_depth`.
  - `structural_complexity`: number of direct child subproblems.
  - `conceptual_depth`: maximum depth of the node's canonical concept tags in the concept dependency graph.
  - `alpha_structural` and `alpha_concept`: weights used to compute `score`.
  - `concept_tag_depths`: canonical concept tags mapped to graph depths.

## Trees

`data/{dataset}/processed_depth2/{dataset}_trees.json` contains recursive root trees. Each tree uses the same node fields as JSONL and additionally includes nested `children`.

## Summary

`data/{dataset}/processed_depth2/{dataset}_summary.json` contains counts, the union of concept tags,
and `concept_graph` metadata with canonical tag depths.

## Repair Existing Artifacts

To remove historical invalid verification records and recompute concept-graph
difficulty without overwriting the existing processed directory:

```bash
uv run python repair_decomp_dataset.py \
  --processed-subdir processed_depth2 \
  --output-subdir processed_depth2_repaired
```

Omit `--output-subdir` to overwrite the selected processed directory.

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
