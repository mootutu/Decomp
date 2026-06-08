#!/usr/bin/env python3
"""Validate dataset-decomposition output artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any


REQUIRED_NODE_FIELDS = {
    "node_id",
    "dataset",
    "problem_id",
    "depth",
    "problem",
    "answer",
    "solution",
    "parent_node_id",
    "concept_tags",
    "summary",
    "depends_on",
    "verification",
    "child_node_ids",
    "difficulty",
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate AIME decomposition dataset artifacts.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--processed-subdir",
        default="processed_depth2",
        help="Subdirectory under data-dir containing final artifacts.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aime24", "aime25"],
        choices=["aime24", "aime25"],
    )
    parser.add_argument("--expected-roots", type=int, default=30)
    parser.add_argument(
        "--require-decomposed",
        action="store_true",
        help="Require at least one generated child node per dataset.",
    )
    parser.add_argument(
        "--require-verified",
        action="store_true",
        help="Require at least one verification record per dataset.",
    )
    parser.add_argument(
        "--expected-max-depth",
        type=int,
        default=None,
        help="Require summary and node depths to match this maximum depth.",
    )
    args = parser.parse_args()

    errors: list[str] = []
    for dataset in args.datasets:
        errors.extend(
            validate_dataset(
                args.data_dir,
                args.processed_subdir,
                dataset,
                args.expected_roots,
                args.require_decomposed,
                args.require_verified,
                args.expected_max_depth,
            )
        )

    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1

    print("Dataset artifacts are valid.")
    return 0


def validate_dataset(
    data_dir: Path,
    processed_subdir: str,
    dataset: str,
    expected_roots: int,
    require_decomposed: bool,
    require_verified: bool,
    expected_max_depth: int | None = None,
) -> list[str]:
    processed_dir = dataset_processed_dir(data_dir, processed_subdir, dataset)
    nodes_path = processed_dir / f"{dataset}_nodes.jsonl"
    trees_path = processed_dir / f"{dataset}_trees.json"
    summary_path = processed_dir / f"{dataset}_summary.json"
    errors: list[str] = []

    for path in [nodes_path, trees_path, summary_path]:
        if not path.exists():
            errors.append(f"{dataset}: missing {path}")
    if errors:
        return errors

    nodes = read_jsonl(nodes_path)
    trees = read_json(trees_path)
    summary = read_json(summary_path)

    if not isinstance(trees, list):
        errors.append(f"{dataset}: trees file must contain a list")
        trees = []
    if not isinstance(summary, dict):
        errors.append(f"{dataset}: summary file must contain an object")
        summary = {}

    root_nodes = [node for node in nodes if node.get("parent_node_id") is None]
    if len(root_nodes) != expected_roots:
        errors.append(f"{dataset}: expected {expected_roots} root nodes, found {len(root_nodes)}")
    if len(trees) != expected_roots:
        errors.append(f"{dataset}: expected {expected_roots} root trees, found {len(trees)}")

    node_ids: set[str] = set()
    child_ids: set[str] = set()
    verified_count = 0
    generated_count = 0
    for idx, node in enumerate(nodes, start=1):
        missing = REQUIRED_NODE_FIELDS - set(node)
        if missing:
            errors.append(f"{dataset}: node line {idx} missing fields {sorted(missing)}")
        if node.get("dataset") != dataset:
            errors.append(f"{dataset}: node line {idx} has mismatched dataset {node.get('dataset')!r}")
        if not node.get("node_id"):
            errors.append(f"{dataset}: node line {idx} has empty node_id")
        if not node.get("problem"):
            errors.append(f"{dataset}: node line {idx} has empty problem")
        if str(node.get("answer", "")).strip() == "":
            errors.append(f"{dataset}: node line {idx} has empty answer")
        if "children" in node:
            errors.append(f"{dataset}: node line {idx} should be flat and must not contain nested children")
        node_ids.add(str(node.get("node_id")))
        child_ids.update(str(child_id) for child_id in node.get("child_node_ids", []))
        if node.get("parent_node_id") is not None:
            generated_count += 1
        if node.get("verification") is not None:
            verified_count += 1
            if node["verification"].get("valid") is not True:
                errors.append(f"{dataset}: node line {idx} has invalid verification")

    missing_children = sorted(child_ids - node_ids)
    if missing_children:
        errors.append(f"{dataset}: child ids missing from flat nodes: {missing_children[:5]}")
    if require_decomposed and len(nodes) <= len(root_nodes):
        errors.append(f"{dataset}: expected generated subproblem nodes, found only roots")
    if require_verified and verified_count != generated_count:
        errors.append(
            f"{dataset}: expected every generated node to have verification, "
            f"found {verified_count}/{generated_count}"
        )

    if summary.get("num_root_problems") != len(root_nodes):
        errors.append(f"{dataset}: summary num_root_problems does not match nodes")
    if summary.get("num_total_nodes") != len(nodes):
        errors.append(f"{dataset}: summary num_total_nodes does not match nodes")
    actual_max_depth = max((int(node.get("depth", 0)) for node in nodes), default=0)
    if summary.get("max_depth") != actual_max_depth:
        errors.append(f"{dataset}: summary max_depth does not match nodes")
    if expected_max_depth is not None and actual_max_depth != expected_max_depth:
        errors.append(f"{dataset}: expected max depth {expected_max_depth}, found {actual_max_depth}")

    return errors


def dataset_processed_dir(data_dir: Path, processed_subdir: str, dataset: str) -> Path:
    new_layout = data_dir / dataset / processed_subdir
    if new_layout.exists():
        return new_layout
    flat_layout = data_dir / processed_subdir
    if flat_layout.exists():
        return flat_layout
    return data_dir / "decomp" / processed_subdir


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path} contains a non-object JSONL row")
            records.append(value)
    return records


if __name__ == "__main__":
    raise SystemExit(main())
