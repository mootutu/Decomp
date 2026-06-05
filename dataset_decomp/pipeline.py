"""Recursive dataset-decomposition pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from openai import OpenAI

from .aime_sources import AimeProblem, fetch_dataset
from .json_utils import parse_json_object
from .prompts import (
    DECOMPOSE_PROMPT,
    DECOMPOSITION_SYSTEM,
    ORIGINAL_SOLUTION_PROMPT,
    VERIFY_PROMPT,
)


@dataclass
class BuildConfig:
    datasets: list[str]
    output_dir: Path
    model: str
    processed_subdir: str = "processed_depth2"
    max_depth: int = 2
    max_problems: int | None = None
    refresh_raw: bool = False
    refresh_llm: bool = False
    verify_subproblems: bool = True
    sleep_seconds: float = 0.0
    dry_run: bool = False


@dataclass
class DecompNode:
    node_id: str
    dataset: str
    problem_id: str
    depth: int
    problem: str
    answer: str
    solution: str | None
    parent_node_id: str | None
    concept_tags: list[str] = field(default_factory=list)
    summary: str | None = None
    depends_on: list[str] = field(default_factory=list)
    verification: dict[str, Any] | None = None
    children: list["DecompNode"] = field(default_factory=list)


def build_datasets(client: OpenAI | None, config: BuildConfig) -> None:
    config.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = config.output_dir / "raw"
    processed_dir = config.output_dir / config.processed_subdir
    processed_dir.mkdir(parents=True, exist_ok=True)

    for dataset_key in config.datasets:
        problems = fetch_dataset(dataset_key, raw_dir, refresh=config.refresh_raw)
        if config.max_problems is not None:
            problems = problems[: config.max_problems]

        roots: list[DecompNode] = []
        for idx, problem in enumerate(problems, start=1):
            print(f"[{dataset_key}] {idx}/{len(problems)} {problem.problem_id}")
            root = build_problem_tree(client, config, problem)
            roots.append(root)

        flat_nodes = flatten_nodes(roots)
        write_jsonl(
            processed_dir / f"{dataset_key}_nodes.jsonl",
            [node_to_record(node, include_children=False) for node in flat_nodes],
        )
        write_json(processed_dir / f"{dataset_key}_trees.json", [node_to_record(root) for root in roots])
        write_json(processed_dir / f"{dataset_key}_summary.json", summarize(dataset_key, roots))


def build_problem_tree(client: OpenAI | None, config: BuildConfig, problem: AimeProblem) -> DecompNode:
    solution = problem.solution
    if not solution and not config.dry_run:
        require_client(client)
        solution_data = llm_json(
            client,
            config,
            prompt=ORIGINAL_SOLUTION_PROMPT.format(problem=problem.question, answer=problem.answer),
            cache_key=f"{problem.dataset}/{problem.problem_id}/original_solution",
        )
        solution = str(solution_data.get("solution") or "")

    root = DecompNode(
        node_id=stable_id(problem.dataset, problem.problem_id, "root"),
        dataset=problem.dataset,
        problem_id=problem.problem_id,
        depth=0,
        problem=problem.question,
        answer=problem.answer,
        solution=solution,
        parent_node_id=None,
    )
    expand_node(client, config, root)
    return root


def expand_node(client: OpenAI | None, config: BuildConfig, node: DecompNode) -> None:
    if node.depth >= config.max_depth:
        return
    if config.dry_run:
        node.summary = "Dry run: LLM decomposition skipped."
        return
    require_client(client)

    decomp = llm_json(
        client,
        config,
        prompt=DECOMPOSE_PROMPT.format(
            problem=node.problem,
            answer=node.answer,
            solution=node.solution or "No solution provided.",
        ),
        cache_key=f"{node.dataset}/{node.problem_id}/{node.node_id}/decompose",
    )

    node.summary = as_optional_str(decomp.get("summary"))
    node.concept_tags = string_list(decomp.get("concept_tags"))
    if bool(decomp.get("is_atomic")):
        return

    for step in list_value(decomp.get("steps")):
        child = step_to_child(node, step)
        if config.verify_subproblems:
            verification = llm_json(
                client,
                config,
                prompt=VERIFY_PROMPT.format(
                    problem=child.problem,
                    answer=child.answer,
                    solution=child.solution or "",
                ),
                cache_key=f"{node.dataset}/{node.problem_id}/{child.node_id}/verify",
            )
            child.verification = verification
            if verification.get("corrected_answer"):
                child.answer = str(verification["corrected_answer"])
            if verification.get("corrected_solution"):
                child.solution = str(verification["corrected_solution"])

        node.children.append(child)
        expand_node(client, config, child)


def step_to_child(parent: DecompNode, step: dict[str, Any]) -> DecompNode:
    step_id = str(step.get("step_id") or f"s{len(parent.children) + 1}")
    child_id = stable_id(parent.node_id, step_id, str(step.get("subproblem") or ""))
    return DecompNode(
        node_id=child_id,
        dataset=parent.dataset,
        problem_id=parent.problem_id,
        depth=parent.depth + 1,
        problem=str(step.get("subproblem") or ""),
        answer=str(step.get("expected_answer") or ""),
        solution=as_optional_str(step.get("solution")),
        parent_node_id=parent.node_id,
        concept_tags=string_list(step.get("concept_tags")),
        summary=as_optional_str(step.get("description")),
        depends_on=string_list(step.get("depends_on")),
    )


def llm_json(client: OpenAI | None, config: BuildConfig, prompt: str, cache_key: str) -> dict[str, Any]:
    cache_path = cache_file(config.output_dir / "llm_cache", cache_key)
    if cache_path.exists() and not config.refresh_llm:
        return json.loads(cache_path.read_text(encoding="utf-8"))["parsed"]

    response = require_client(client).chat.completions.create(
        model=config.model,
        messages=[
            {"role": "system", "content": DECOMPOSITION_SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    content = response.choices[0].message.content or ""
    parsed = parse_json_object(content)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(
        json.dumps({"prompt": prompt, "response": content, "parsed": parsed}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if config.sleep_seconds:
        time.sleep(config.sleep_seconds)
    return parsed


def flatten_nodes(nodes: list[DecompNode]) -> list[DecompNode]:
    flattened: list[DecompNode] = []
    for node in nodes:
        flattened.append(node)
        flattened.extend(flatten_nodes(node.children))
    return flattened


def node_to_record(node: DecompNode, include_children: bool = True) -> dict[str, Any]:
    record = asdict(node)
    record["child_node_ids"] = [child.node_id for child in node.children]
    if include_children:
        record["children"] = [node_to_record(child) for child in node.children]
    else:
        record.pop("children", None)
    record["difficulty"] = estimate_difficulty(node)
    return record


def estimate_difficulty(node: DecompNode) -> dict[str, Any]:
    child_count = len(node.children)
    tag_count = len(set(node.concept_tags))
    dependency_count = sum(len(child.depends_on) for child in node.children)
    score = node.depth + child_count + tag_count + dependency_count
    return {
        "score": score,
        "depth": node.depth,
        "num_children": child_count,
        "num_concept_tags": tag_count,
        "num_child_dependencies": dependency_count,
    }


def summarize(dataset_key: str, roots: list[DecompNode]) -> dict[str, Any]:
    nodes = flatten_nodes(roots)
    return {
        "dataset": dataset_key,
        "num_root_problems": len(roots),
        "num_total_nodes": len(nodes),
        "max_depth": max((node.depth for node in nodes), default=0),
        "num_verified_nodes": sum(1 for node in nodes if node.verification is not None),
        "concept_tags": sorted({tag for node in nodes for tag in node.concept_tags}),
    }


def cache_file(cache_dir: Path, cache_key: str) -> Path:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def stable_id(*parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"n_{digest}"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def list_value(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def as_optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def require_client(client: OpenAI | None) -> OpenAI:
    if client is None:
        raise RuntimeError("This operation requires an AveMujica API client.")
    return client
