from __future__ import annotations

from pathlib import Path
import time

from dataset_decomp import pipeline
from dataset_decomp.aime_sources import AimeProblem
from dataset_decomp.pipeline import (
    BuildConfig,
    DecompNode,
    answers_match,
    build_concept_depths,
    build_datasets,
    build_problem_trees,
    canonicalize_tag,
    expand_node,
    flatten_nodes,
    node_to_record,
    summarize,
)


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
    assert record["difficulty"]["structural_complexity"] == 1


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
        "concept_graph": {
            "num_nodes": 1,
            "max_concept_depth": 0,
            "concept_depths": {"modular arithmetic": 0},
        },
    }


def test_expand_node_skips_invalid_verified_child(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_llm_json(client, config, prompt: str, cache_key: str):
        calls.append(cache_key)
        if cache_key.endswith("/decompose"):
            return {
                "summary": "Split into steps.",
                "concept_tags": ["addition"],
                "is_atomic": False,
                "steps": [
                    {
                        "step_id": "s1",
                        "description": "bad step",
                        "concept_tags": ["addition"],
                        "subproblem": "What is 2+2?",
                        "expected_answer": "5",
                        "solution": "2+2=5",
                        "depends_on": [],
                    }
                ],
            }
        return {
            "solution": "2+2=4",
            "final_answer": "4",
        }

    monkeypatch.setattr(pipeline, "llm_json", fake_llm_json)
    root = DecompNode(
        node_id="root",
        dataset="aime24",
        problem_id="p1",
        depth=0,
        problem="Problem",
        answer="4",
        solution="Solve it.",
        parent_node_id=None,
    )
    config = BuildConfig(
        datasets=["aime24"],
        output_dir=tmp_path,
        model="test-model",
        max_depth=1,
        verify_retries=0,
    )

    expand_node(object(), config, root)

    assert root.children == []
    assert len(calls) == 2


def test_expand_node_regenerates_invalid_verified_child(monkeypatch, tmp_path: Path) -> None:
    calls = []

    def fake_llm_json(client, config, prompt: str, cache_key: str):
        calls.append(cache_key)
        if cache_key.endswith("/decompose"):
            return {
                "summary": "Split into steps.",
                "concept_tags": ["addition"],
                "is_atomic": False,
                "steps": [
                    {
                        "step_id": "s1",
                        "description": "addition step",
                        "concept_tags": ["addition"],
                        "subproblem": "What is 2+2?",
                        "expected_answer": "5",
                        "solution": "2+2=5",
                        "depends_on": [],
                    }
                ],
            }
        if "/regenerate_1" in cache_key:
            return {
                "step_id": "s1",
                "description": "addition step",
                "concept_tags": ["addition"],
                "subproblem": "What is 2+3?",
                "expected_answer": "5",
                "solution": "2+3=5",
                "depends_on": [],
            }
        if cache_key.endswith("/solve_for_verify"):
            return {
                "solution": "2+2=4",
                "final_answer": "4",
            }
        return {
            "solution": "2+3=5",
            "final_answer": "5",
        }

    monkeypatch.setattr(pipeline, "llm_json", fake_llm_json)
    root = DecompNode(
        node_id="root",
        dataset="aime24",
        problem_id="p1",
        depth=0,
        problem="Problem",
        answer="4",
        solution="Solve it.",
        parent_node_id=None,
    )
    config = BuildConfig(datasets=["aime24"], output_dir=tmp_path, model="test-model", max_depth=1)

    expand_node(object(), config, root)

    assert len(root.children) == 1
    assert root.children[0].problem == "What is 2+3?"
    assert root.children[0].verification == {
        "valid": True,
        "corrected_answer": "5",
        "corrected_solution": "2+3=5",
        "reason": "Independent solve matched proposed answer.",
        "method": "independent_solve",
        "proposed_answer": "5",
        "independent_answer": "5",
        "independent_solution": "2+3=5",
    }
    assert any("/regenerate_1" in call for call in calls)
    assert any(call.endswith("/solve_for_verify_retry_1") for call in calls)


def test_answers_match_normalizes_common_answer_formats() -> None:
    assert answers_match("004", "4")
    assert answers_match("\\boxed{12}", "12")
    assert answers_match("\\frac{2}{4}", "1/2")
    assert answers_match("$x+1$", "x + 1")
    assert not answers_match("5", "4")


def test_build_concept_depths_uses_parent_child_tag_graph() -> None:
    child = DecompNode(
        node_id="child",
        dataset="aime24",
        problem_id="p1",
        depth=1,
        problem="Subproblem",
        answer="2",
        solution="Because.",
        parent_node_id="root",
        concept_tags=["Vieta formulas"],
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
        concept_tags=["Quadratic Equations"],
        children=[child],
    )

    depths = build_concept_depths([root, child])
    record = node_to_record(child, concept_depths=depths)

    assert canonicalize_tag("Quadratic Equations") == "quadratic equation"
    assert depths == {"quadratic equation": 0, "vieta formula": 1}
    assert record["difficulty"]["conceptual_depth"] == 1
    assert record["difficulty"]["score"] == 1.0


def test_longest_depths_handles_tag_cycles() -> None:
    depths = pipeline.longest_depths({"a", "b", "c"}, {("a", "b"), ("b", "a"), ("b", "c")})

    assert depths["a"] == 0
    assert depths["b"] == 0
    assert depths["c"] == 1


def test_build_problem_trees_preserves_source_order_with_workers(monkeypatch, tmp_path: Path) -> None:
    problems = [
        AimeProblem("aime24", "p1", "Question 1", "1"),
        AimeProblem("aime24", "p2", "Question 2", "2"),
    ]

    def fake_build_problem_tree(client, config, problem):
        if problem.problem_id == "p1":
            time.sleep(0.01)
        return DecompNode(
            node_id=problem.problem_id,
            dataset=problem.dataset,
            problem_id=problem.problem_id,
            depth=0,
            problem=problem.question,
            answer=problem.answer,
            solution=None,
            parent_node_id=None,
        )

    monkeypatch.setattr(pipeline, "build_problem_tree", fake_build_problem_tree)
    config = BuildConfig(datasets=["aime24"], output_dir=tmp_path, model="test-model", max_workers=2)

    roots = build_problem_trees(None, config, "aime24", problems)

    assert [root.problem_id for root in roots] == ["p1", "p2"]


def test_build_datasets_writes_per_dataset_directories(monkeypatch, tmp_path: Path) -> None:
    def fake_fetch_dataset(dataset_key, cache_dir, refresh=False):
        assert cache_dir == tmp_path / dataset_key / "raw"
        return [AimeProblem(dataset_key, "p1", "Question 1", "1", "Solution.")]

    monkeypatch.setattr(pipeline, "fetch_dataset", fake_fetch_dataset)
    config = BuildConfig(
        datasets=["aime24"],
        output_dir=tmp_path,
        model="test-model",
        processed_subdir="processed_test",
        dry_run=True,
    )

    build_datasets(None, config)

    assert (tmp_path / "aime24" / "processed_test" / "aime24_nodes.jsonl").exists()
    assert (tmp_path / "aime24" / "processed_test" / "aime24_trees.json").exists()
    assert (tmp_path / "aime24" / "processed_test" / "aime24_summary.json").exists()
