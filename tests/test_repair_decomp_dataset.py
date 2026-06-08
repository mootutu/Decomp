from __future__ import annotations

import json
from pathlib import Path

from repair_decomp_dataset import repair_dataset


def test_repair_dataset_removes_invalid_nodes_and_descendants(tmp_path: Path) -> None:
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    root = make_node("root", None, ["bad"], verification=None)
    bad = make_node("bad", "root", ["bad_child"], verification={"valid": False})
    descendant = make_node("descendant", "bad", [], verification={"valid": True})
    good = make_node("good", "root", [], verification={"valid": True})
    root["child_node_ids"] = ["bad", "good"]
    bad["child_node_ids"] = ["descendant"]

    write_jsonl(source / "aime24_nodes.jsonl", [root, bad, descendant, good])

    repair_dataset(source, output, "aime24")

    records = read_jsonl(output / "aime24_nodes.jsonl")
    ids = [record["node_id"] for record in records]
    repaired_root = records[0]
    summary = json.loads((output / "aime24_summary.json").read_text(encoding="utf-8"))

    assert ids == ["root", "good"]
    assert repaired_root["child_node_ids"] == ["good"]
    assert summary["num_total_nodes"] == 2
    assert summary["num_verified_nodes"] == 1


def make_node(node_id: str, parent_id: str | None, tags: list[str], verification: dict | None) -> dict:
    return {
        "node_id": node_id,
        "dataset": "aime24",
        "problem_id": "p1",
        "depth": 0 if parent_id is None else 1,
        "problem": f"Problem {node_id}",
        "answer": "1",
        "solution": "Solution.",
        "parent_node_id": parent_id,
        "concept_tags": tags,
        "summary": "Summary.",
        "depends_on": [],
        "verification": verification,
        "child_node_ids": [],
        "difficulty": {},
    }


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record) + "\n" for record in records), encoding="utf-8")
