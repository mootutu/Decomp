from __future__ import annotations

from dataset_decomp.pipeline import DecompNode, flatten_nodes, node_to_record, summarize


def test_node_to_record_can_be_flat() -> None:
    child = DecompNode(
        node_id="child",
        dataset="aime24",
        problem_id="p1",
        depth=1,
        problem="Subproblem",
        answer="2",
        solution="Because.",
        parent_node_id="root",
    )
    root = DecompNode(
        node_id="root",
        dataset="aime24",
        problem_id="p1",
        depth=0,
        problem="Problem",
        answer="4",
        solution="Solve it.",
        parent_node_id=None,
        children=[child],
    )

    record = node_to_record(root, include_children=False)

    assert "children" not in record
    assert record["child_node_ids"] == ["child"]
    assert record["difficulty"]["num_children"] == 1


def test_flatten_and_summarize_tree() -> None:
    child = DecompNode(
        node_id="child",
        dataset="aime25",
        problem_id="p1",
        depth=1,
        problem="Subproblem",
        answer="2",
        solution=None,
        parent_node_id="root",
        concept_tags=["modular arithmetic"],
    )
    root = DecompNode(
        node_id="root",
        dataset="aime25",
        problem_id="p1",
        depth=0,
        problem="Problem",
        answer="4",
        solution=None,
        parent_node_id=None,
        children=[child],
    )

    assert [node.node_id for node in flatten_nodes([root])] == ["root", "child"]
    assert summarize("aime25", [root]) == {
        "dataset": "aime25",
        "num_root_problems": 1,
        "num_total_nodes": 2,
        "max_depth": 1,
        "num_verified_nodes": 0,
        "concept_tags": ["modular arithmetic"],
    }
