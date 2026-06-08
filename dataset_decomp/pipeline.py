"""Recursive dataset-decomposition pipeline."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from fractions import Fraction
import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm

from .aime_sources import AimeProblem, fetch_dataset
from .json_utils import parse_json_object
from .prompts import (
    DECOMPOSE_PROMPT,
    DECOMPOSITION_SYSTEM,
    ORIGINAL_SOLUTION_PROMPT,
    REGENERATE_SUBPROBLEM_PROMPT,
    SOLVE_SUBPROBLEM_PROMPT,
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
    verify_retries: int = 3
    difficulty_structural_weight: float = 1.0
    difficulty_concept_weight: float = 1.0
    max_workers: int = 1
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

    for dataset_key in config.datasets:
        dataset_config = replace(config, output_dir=dataset_output_dir(config.output_dir, dataset_key))
        dataset_config.output_dir.mkdir(parents=True, exist_ok=True)
        raw_dir = dataset_config.output_dir / "raw"
        processed_dir = dataset_config.output_dir / dataset_config.processed_subdir
        processed_dir.mkdir(parents=True, exist_ok=True)

        problems = fetch_dataset(dataset_key, raw_dir, refresh=config.refresh_raw)
        if config.max_problems is not None:
            problems = problems[: config.max_problems]

        roots = build_problem_trees(client, dataset_config, dataset_key, problems)

        flat_nodes = flatten_nodes(roots)
        concept_depths = build_concept_depths(flat_nodes)
        write_jsonl(
            processed_dir / f"{dataset_key}_nodes.jsonl",
            [
                node_to_record(node, include_children=False, concept_depths=concept_depths, config=dataset_config)
                for node in flat_nodes
            ],
        )
        write_json(
            processed_dir / f"{dataset_key}_trees.json",
            [node_to_record(root, concept_depths=concept_depths, config=dataset_config) for root in roots],
        )
        write_json(processed_dir / f"{dataset_key}_summary.json", summarize(dataset_key, roots, concept_depths))


def dataset_output_dir(output_dir: Path, dataset_key: str) -> Path:
    return output_dir / dataset_key


def build_problem_trees(
    client: OpenAI | None,
    config: BuildConfig,
    dataset_key: str,
    problems: list[AimeProblem],
) -> list[DecompNode]:
    if config.max_workers <= 1:
        roots = []
        iterator = tqdm(problems, desc=f"{dataset_key}", unit="problem")
        for problem in iterator:
            iterator.set_postfix_str(problem.problem_id)
            roots.append(build_problem_tree(client, config, problem))
        return roots

    roots_by_index: list[DecompNode | None] = [None] * len(problems)
    with ThreadPoolExecutor(max_workers=config.max_workers) as executor:
        future_to_index = {
            executor.submit(build_problem_tree, client, config, problem): idx
            for idx, problem in enumerate(problems)
        }
        progress = tqdm(as_completed(future_to_index), total=len(problems), desc=f"{dataset_key}", unit="problem")
        for future in progress:
            idx = future_to_index[future]
            progress.set_postfix_str(problems[idx].problem_id)
            roots_by_index[idx] = future.result()

    return [root for root in roots_by_index if root is not None]


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
            child = verify_or_regenerate_child(client, config, node, step, child)
            if child is None:
                continue

        node.children.append(child)
        expand_node(client, config, child)


def verify_or_regenerate_child(
    client: OpenAI | None,
    config: BuildConfig,
    parent: DecompNode,
    step: dict[str, Any],
    child: DecompNode,
) -> DecompNode | None:
    current_child = child
    last_verification: dict[str, Any] | None = None

    for attempt in range(config.verify_retries + 1):
        if attempt > 0:
            current_child = regenerate_child(client, config, parent, step, current_child, last_verification, attempt)

        verification = verify_child(client, config, parent, current_child, attempt)
        current_child.verification = verification
        if verification.get("valid") is not True:
            last_verification = verification
            continue
        if verification.get("corrected_answer"):
            current_child.answer = str(verification["corrected_answer"])
        if verification.get("corrected_solution"):
            current_child.solution = str(verification["corrected_solution"])
        return current_child

    return None


def verify_child(
    client: OpenAI | None,
    config: BuildConfig,
    parent: DecompNode,
    child: DecompNode,
    attempt: int,
) -> dict[str, Any]:
    solve_suffix = "solve_for_verify" if attempt == 0 else f"solve_for_verify_retry_{attempt}"
    solved = llm_json(
        client,
        config,
        prompt=SOLVE_SUBPROBLEM_PROMPT.format(problem=child.problem),
        cache_key=f"{parent.dataset}/{parent.problem_id}/{child.node_id}/{solve_suffix}",
    )
    independent_answer = str(solved.get("final_answer") or solved.get("answer") or "")
    independent_solution = as_optional_str(solved.get("solution"))
    valid = answers_match(child.answer, independent_answer)
    return {
        "valid": valid,
        "corrected_answer": independent_answer if valid else child.answer,
        "corrected_solution": independent_solution if valid else child.solution,
        "reason": (
            "Independent solve matched proposed answer."
            if valid
            else f"Independent answer {independent_answer!r} did not match proposed answer {child.answer!r}."
        ),
        "method": "independent_solve",
        "proposed_answer": child.answer,
        "independent_answer": independent_answer,
        "independent_solution": independent_solution,
    }


def regenerate_child(
    client: OpenAI | None,
    config: BuildConfig,
    parent: DecompNode,
    step: dict[str, Any],
    invalid_child: DecompNode,
    verification: dict[str, Any] | None,
    attempt: int,
) -> DecompNode:
    regenerated = llm_json(
        client,
        config,
        prompt=REGENERATE_SUBPROBLEM_PROMPT.format(
            parent_problem=parent.problem,
            parent_answer=parent.answer,
            parent_solution=parent.solution or "No solution provided.",
            step_description=step.get("description") or "",
            concept_tags=json.dumps(string_list(step.get("concept_tags")), ensure_ascii=False),
            invalid_problem=invalid_child.problem,
            invalid_answer=invalid_child.answer,
            invalid_solution=invalid_child.solution or "",
            failure_reason=(verification or {}).get("reason") or "Verification failed.",
            step_id=str(step.get("step_id") or ""),
        ),
        cache_key=f"{parent.dataset}/{parent.problem_id}/{invalid_child.node_id}/regenerate_{attempt}",
    )
    return step_to_child(parent, regenerated)


def answers_match(proposed: str, independent: str) -> bool:
    proposed_norm = normalize_answer_for_compare(proposed)
    independent_norm = normalize_answer_for_compare(independent)
    if not proposed_norm or not independent_norm:
        return False
    if proposed_norm == independent_norm:
        return True

    proposed_fraction = parse_fraction(proposed_norm)
    independent_fraction = parse_fraction(independent_norm)
    return proposed_fraction is not None and proposed_fraction == independent_fraction


def normalize_answer_for_compare(answer: str) -> str:
    text = str(answer).strip()
    if not text:
        return ""

    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]

    text = text.strip()
    text = re.sub(r"^\$+|\$+$", "", text)
    text = text.replace("\\(", "").replace("\\)", "")
    text = text.replace("\\[", "").replace("\\]", "")
    text = text.replace(",", "")
    text = normalize_latex_fraction(text)
    if re.fullmatch(r"-?\d+", text):
        return str(int(text))

    text = re.sub(r"\s+", "", text)
    return text.lower()


def parse_fraction(text: str) -> Fraction | None:
    if not re.fullmatch(r"-?\d+(?:/\d+)?", text):
        return None
    try:
        return Fraction(text)
    except ZeroDivisionError:
        return None


def normalize_latex_fraction(text: str) -> str:
    return re.sub(
        r"\\frac\s*\{([^{}]+)\}\s*\{([^{}]+)\}",
        lambda match: f"{match.group(1)}/{match.group(2)}",
        text,
    )


def canonicalize_tag(tag: str) -> str:
    value = str(tag).strip().lower()
    value = value.replace("&", " and ")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    words = [singularize_word(word) for word in value.split()]
    return " ".join(word for word in words if word)


def singularize_word(word: str) -> str:
    if len(word) <= 3:
        return word
    if word.endswith("ies"):
        return f"{word[:-3]}y"
    if word.endswith("ses"):
        return word[:-2]
    if word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


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
    write_json_atomic(cache_path, {"prompt": prompt, "response": content, "parsed": parsed})
    if config.sleep_seconds:
        time.sleep(config.sleep_seconds)
    return parsed


def flatten_nodes(nodes: list[DecompNode]) -> list[DecompNode]:
    flattened: list[DecompNode] = []
    for node in nodes:
        flattened.append(node)
        flattened.extend(flatten_nodes(node.children))
    return flattened


def node_to_record(
    node: DecompNode,
    include_children: bool = True,
    concept_depths: dict[str, int] | None = None,
    config: BuildConfig | None = None,
) -> dict[str, Any]:
    record = asdict(node)
    record["child_node_ids"] = [child.node_id for child in node.children]
    if include_children:
        record["children"] = [
            node_to_record(child, concept_depths=concept_depths, config=config) for child in node.children
        ]
    else:
        record.pop("children", None)
    record["difficulty"] = estimate_difficulty(node, concept_depths, config)
    return record


def estimate_difficulty(
    node: DecompNode,
    concept_depths: dict[str, int] | None = None,
    config: BuildConfig | None = None,
) -> dict[str, Any]:
    concept_depths = concept_depths or {}
    structural_weight = config.difficulty_structural_weight if config else 1.0
    concept_weight = config.difficulty_concept_weight if config else 1.0
    child_count = len(node.children)
    tag_depths = [concept_depths.get(canonicalize_tag(tag), 0) for tag in node.concept_tags]
    concept_depth = max(tag_depths, default=0)
    score = structural_weight * child_count + concept_weight * concept_depth
    return {
        "score": score,
        "structural_complexity": child_count,
        "conceptual_depth": concept_depth,
        "alpha_structural": structural_weight,
        "alpha_concept": concept_weight,
        "concept_tag_depths": {
            canonicalize_tag(tag): concept_depths.get(canonicalize_tag(tag), 0) for tag in node.concept_tags
        },
    }


def summarize(
    dataset_key: str,
    roots: list[DecompNode],
    concept_depths: dict[str, int] | None = None,
) -> dict[str, Any]:
    nodes = flatten_nodes(roots)
    concept_depths = concept_depths or build_concept_depths(nodes)
    return {
        "dataset": dataset_key,
        "num_root_problems": len(roots),
        "num_total_nodes": len(nodes),
        "max_depth": max((node.depth for node in nodes), default=0),
        "num_verified_nodes": sum(1 for node in nodes if node.verification is not None),
        "concept_tags": sorted({tag for node in nodes for tag in node.concept_tags}),
        "concept_graph": {
            "num_nodes": len(concept_depths),
            "max_concept_depth": max(concept_depths.values(), default=0),
            "concept_depths": dict(sorted(concept_depths.items())),
        },
    }


def build_concept_depths(nodes: list[DecompNode]) -> dict[str, int]:
    tags = {canonicalize_tag(tag) for node in nodes for tag in node.concept_tags if canonicalize_tag(tag)}
    edges: set[tuple[str, str]] = set()
    for parent in nodes:
        parent_tags = [canonicalize_tag(tag) for tag in parent.concept_tags if canonicalize_tag(tag)]
        for child in parent.children:
            child_tags = [canonicalize_tag(tag) for tag in child.concept_tags if canonicalize_tag(tag)]
            for parent_tag in parent_tags:
                for child_tag in child_tags:
                    if parent_tag != child_tag:
                        edges.add((parent_tag, child_tag))

    return longest_depths(tags, edges)


def longest_depths(nodes: set[str], edges: set[tuple[str, str]]) -> dict[str, int]:
    components = strongly_connected_components(nodes, edges)
    component_by_node = {
        node: component_idx for component_idx, component in enumerate(components) for node in component
    }
    component_nodes = set(range(len(components)))
    component_edges = {
        (component_by_node[src], component_by_node[dst])
        for src, dst in edges
        if src in component_by_node and dst in component_by_node and component_by_node[src] != component_by_node[dst]
    }
    component_depths = longest_dag_depths(component_nodes, component_edges)
    return {
        node: component_depths.get(component_by_node[node], 0)
        for node in nodes
    }


def longest_dag_depths(nodes: set[int], edges: set[tuple[int, int]]) -> dict[int, int]:
    incoming: dict[int, set[int]] = {node: set() for node in nodes}
    outgoing: dict[int, set[int]] = {node: set() for node in nodes}
    for src, dst in edges:
        if src not in nodes or dst not in nodes:
            continue
        outgoing[src].add(dst)
        incoming[dst].add(src)

    depths = {node: 0 for node in nodes}
    queue = sorted(node for node in nodes if not incoming[node])
    visited = 0
    while queue:
        node = queue.pop(0)
        visited += 1
        for child in sorted(outgoing[node]):
            depths[child] = max(depths[child], depths[node] + 1)
            incoming[child].discard(node)
            if not incoming[child]:
                queue.append(child)

    if visited == len(nodes):
        return depths

    return depths


def strongly_connected_components(nodes: set[str], edges: set[tuple[str, str]]) -> list[set[str]]:
    outgoing: dict[str, list[str]] = {node: [] for node in nodes}
    for src, dst in edges:
        if src in nodes and dst in nodes:
            outgoing[src].append(dst)

    index = 0
    stack: list[str] = []
    on_stack: set[str] = set()
    indices: dict[str, int] = {}
    lowlinks: dict[str, int] = {}
    components: list[set[str]] = []

    def visit(node: str) -> None:
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack.add(node)

        for child in outgoing[node]:
            if child not in indices:
                visit(child)
                lowlinks[node] = min(lowlinks[node], lowlinks[child])
            elif child in on_stack:
                lowlinks[node] = min(lowlinks[node], indices[child])

        if lowlinks[node] == indices[node]:
            component = set()
            while True:
                member = stack.pop()
                on_stack.remove(member)
                component.add(member)
                if member == node:
                    break
            components.append(component)

    for node in sorted(nodes):
        if node not in indices:
            visit(node)

    return components


def cache_file(cache_dir: Path, cache_key: str) -> Path:
    digest = hashlib.sha256(cache_key.encode("utf-8")).hexdigest()
    return cache_dir / f"{digest}.json"


def stable_id(*parts: str) -> str:
    digest = hashlib.sha1("::".join(parts).encode("utf-8")).hexdigest()[:12]
    return f"n_{digest}"


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def write_json_atomic(path: Path, data: Any) -> None:
    tmp_path = path.with_name(f".{path.name}.{os.getpid()}.{time.monotonic_ns()}.tmp")
    tmp_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(tmp_path, path)


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
