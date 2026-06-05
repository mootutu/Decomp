from __future__ import annotations

import json
from pathlib import Path

from validate_decomp_dataset import validate_dataset


def test_validate_dataset_accepts_minimal_root_only_artifacts(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    node = {
        "node_id": "root",
        "dataset": "aime24",
        "problem_id": "p1",
        "depth": 0,
        "problem": "Problem",
        "answer": "4",
        "solution": None,
        "parent_node_id": None,
        "concept_tags": [],
        "summary": "Dry run",
        "depends_on": [],
        "verification": None,
        "child_node_ids": [],
        "difficulty": {
            "score": 0,
            "depth": 0,
            "num_children": 0,
            "num_concept_tags": 0,
            "num_child_dependencies": 0,
        },
    }
    write_jsonl(processed / "aime24_nodes.jsonl", [node])
    write_json(processed / "aime24_trees.json", [node | {"children": []}])
    write_json(
        processed / "aime24_summary.json",
        {
            "dataset": "aime24",
            "num_root_problems": 1,
            "num_total_nodes": 1,
            "max_depth": 0,
            "num_verified_nodes": 0,
            "concept_tags": [],
        },
    )

    assert validate_dataset(tmp_path, "processed", "aime24", 1, False, False) == []


def test_validate_dataset_rejects_nested_children_in_jsonl(tmp_path: Path) -> None:
    processed = tmp_path / "processed"
    processed.mkdir()
    node = {
        "node_id": "root",
        "dataset": "aime24",
        "problem_id": "p1",
        "depth": 0,
        "problem": "Problem",
        "answer": "4",
        "solution": None,
        "parent_node_id": None,
        "concept_tags": [],
        "summary": "Dry run",
        "depends_on": [],
        "verification": None,
        "child_node_ids": [],
        "children": [],
        "difficulty": {},
    }
    write_jsonl(processed / "aime24_nodes.jsonl", [node])
    write_json(processed / "aime24_trees.json", [node])
    write_json(processed / "aime24_summary.json", {"num_root_problems": 1, "num_total_nodes": 1})

    errors = validate_dataset(tmp_path, "processed", "aime24", 1, False, False)

    assert any("must not contain nested children" in error for error in errors)


def test_validate_dataset_accepts_custom_processed_subdir(tmp_path: Path) -> None:
    processed = tmp_path / "processed_depth1"
    processed.mkdir()
    root = {
        "node_id": "root",
        "dataset": "aime24",
        "problem_id": "p1",
        "depth": 0,
        "problem": "Problem",
        "answer": "4",
        "solution": None,
        "parent_node_id": None,
        "concept_tags": [],
        "summary": "Root",
        "depends_on": [],
        "verification": None,
        "child_node_ids": ["child"],
        "difficulty": {},
    }
    child = root | {
        "node_id": "child",
        "depth": 1,
        "problem": "Subproblem",
        "parent_node_id": "root",
        "child_node_ids": [],
        "verification": {"is_correct": True},
    }
    write_jsonl(processed / "aime24_nodes.jsonl", [root, child])
    write_json(processed / "aime24_trees.json", [root | {"children": [child | {"children": []}]}])
    write_json(
        processed / "aime24_summary.json",
        {
            "dataset": "aime24",
            "num_root_problems": 1,
            "num_total_nodes": 2,
            "max_depth": 1,
            "num_verified_nodes": 1,
            "concept_tags": [],
        },
    )

    assert validate_dataset(tmp_path, "processed_depth1", "aime24", 1, True, True, 1) == []


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path: Path, values: list[object]) -> None:
    path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
