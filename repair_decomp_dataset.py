#!/usr/bin/env python3
"""Repair existing decomposed dataset artifacts in place or into a new directory."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from dataset_decomp.pipeline import canonicalize_tag, longest_depths


def main() -> int:
    parser = argparse.ArgumentParser(description="Repair decomposed dataset artifacts.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--processed-subdir", default="processed_depth2")
    parser.add_argument("--output-subdir", default=None, help="Defaults to overwriting --processed-subdir.")
    parser.add_argument("--datasets", nargs="+", default=["aime24", "aime25"], choices=["aime24", "aime25"])
    parser.add_argument("--structural-weight", type=float, default=1.0)
    parser.add_argument("--concept-weight", type=float, default=1.0)
    args = parser.parse_args()

    for dataset in args.datasets:
        source_dir = dataset_processed_dir(args.data_dir, args.processed_subdir, dataset)
        output_dir = args.data_dir / dataset / (args.output_subdir or args.processed_subdir)
        repair_dataset(
            source_dir,
            output_dir,
            dataset,
            structural_weight=args.structural_weight,
            concept_weight=args.concept_weight,
    )
    return 0


def dataset_processed_dir(data_dir: Path, processed_subdir: str, dataset: str) -> Path:
    new_layout = data_dir / dataset / processed_subdir
    if new_layout.exists():
        return new_layout
    flat_layout = data_dir / processed_subdir
    if flat_layout.exists():
        return flat_layout
    return data_dir / "decomp" / processed_subdir


def repair_dataset(
    source_dir: Path,
    output_dir: Path,
    dataset: str,
    structural_weight: float = 1.0,
    concept_weight: float = 1.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    nodes = read_jsonl(source_dir / f"{dataset}_nodes.jsonl")
    kept = filter_valid_nodes(nodes)
    children_by_parent = build_children_by_parent(kept)
    concept_depths = build_concept_depths_from_records(kept, children_by_parent)
    records = [
        record_with_computed_fields(record, children_by_parent, concept_depths, structural_weight, concept_weight)
        for record in kept
    ]
    trees = [
        tree_for(root, children_by_parent, concept_depths, structural_weight, concept_weight)
        for root in records
        if root.get("parent_node_id") is None
    ]
    summary = summarize_records(dataset, records, concept_depths)

    write_jsonl(output_dir / f"{dataset}_nodes.jsonl", records)
    write_json(output_dir / f"{dataset}_trees.json", trees)
    write_json(output_dir / f"{dataset}_summary.json", summary)


def filter_valid_nodes(nodes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_id = {str(node["node_id"]): dict(node) for node in nodes}
    children_by_parent: dict[str, list[str]] = {}
    for node in nodes:
        parent_id = node.get("parent_node_id")
        if parent_id is not None:
            children_by_parent.setdefault(str(parent_id), []).append(str(node["node_id"]))

    invalid_ids = {
        str(node["node_id"])
        for node in nodes
        if node.get("verification") is not None and node["verification"].get("valid") is not True
    }
    remove_ids = set(invalid_ids)
    queue = list(invalid_ids)
    while queue:
        node_id = queue.pop(0)
        for child_id in children_by_parent.get(node_id, []):
            if child_id not in remove_ids:
                remove_ids.add(child_id)
                queue.append(child_id)

    return [by_id[str(node["node_id"])] for node in nodes if str(node["node_id"]) not in remove_ids]


def build_children_by_parent(nodes: list[dict[str, Any]]) -> dict[str | None, list[dict[str, Any]]]:
    children_by_parent: dict[str | None, list[dict[str, Any]]] = {}
    for node in nodes:
        children_by_parent.setdefault(node.get("parent_node_id"), []).append(node)
    return children_by_parent


def record_with_computed_fields(
    record: dict[str, Any],
    children_by_parent: dict[str | None, list[dict[str, Any]]],
    concept_depths: dict[str, int],
    structural_weight: float,
    concept_weight: float,
) -> dict[str, Any]:
    output = dict(record)
    output.pop("children", None)
    children = children_by_parent.get(str(output["node_id"]), [])
    output["child_node_ids"] = [child["node_id"] for child in children]
    output["difficulty"] = estimate_record_difficulty(output, children, concept_depths, structural_weight, concept_weight)
    return output


def tree_for(
    record: dict[str, Any],
    children_by_parent: dict[str | None, list[dict[str, Any]]],
    concept_depths: dict[str, int],
    structural_weight: float,
    concept_weight: float,
) -> dict[str, Any]:
    output = record_with_computed_fields(record, children_by_parent, concept_depths, structural_weight, concept_weight)
    output["children"] = [
        tree_for(child, children_by_parent, concept_depths, structural_weight, concept_weight)
        for child in children_by_parent.get(str(record["node_id"]), [])
    ]
    return output


def estimate_record_difficulty(
    record: dict[str, Any],
    children: list[dict[str, Any]],
    concept_depths: dict[str, int],
    structural_weight: float,
    concept_weight: float,
) -> dict[str, Any]:
    tag_depths = {
        canonicalize_tag(tag): concept_depths.get(canonicalize_tag(tag), 0)
        for tag in record.get("concept_tags", [])
    }
    conceptual_depth = max(tag_depths.values(), default=0)
    structural_complexity = len(children)
    return {
        "score": structural_weight * structural_complexity + concept_weight * conceptual_depth,
        "structural_complexity": structural_complexity,
        "conceptual_depth": conceptual_depth,
        "alpha_structural": structural_weight,
        "alpha_concept": concept_weight,
        "concept_tag_depths": tag_depths,
    }


def build_concept_depths_from_records(
    nodes: list[dict[str, Any]],
    children_by_parent: dict[str | None, list[dict[str, Any]]],
) -> dict[str, int]:
    tags = {canonicalize_tag(tag) for node in nodes for tag in node.get("concept_tags", []) if canonicalize_tag(tag)}
    edges: set[tuple[str, str]] = set()
    for parent in nodes:
        parent_tags = [canonicalize_tag(tag) for tag in parent.get("concept_tags", []) if canonicalize_tag(tag)]
        for child in children_by_parent.get(str(parent["node_id"]), []):
            child_tags = [canonicalize_tag(tag) for tag in child.get("concept_tags", []) if canonicalize_tag(tag)]
            for parent_tag in parent_tags:
                for child_tag in child_tags:
                    if parent_tag != child_tag:
                        edges.add((parent_tag, child_tag))
    return longest_depths(tags, edges)


def summarize_records(dataset: str, records: list[dict[str, Any]], concept_depths: dict[str, int]) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "num_root_problems": sum(1 for record in records if record.get("parent_node_id") is None),
        "num_total_nodes": len(records),
        "max_depth": max((int(record.get("depth", 0)) for record in records), default=0),
        "num_verified_nodes": sum(1 for record in records if record.get("verification") is not None),
        "concept_tags": sorted({tag for record in records for tag in record.get("concept_tags", [])}),
        "concept_graph": {
            "num_nodes": len(concept_depths),
            "max_concept_depth": max(concept_depths.values(), default=0),
            "concept_depths": dict(sorted(concept_depths.items())),
        },
    }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, values: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(value, ensure_ascii=False) + "\n" for value in values), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
