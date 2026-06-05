"""Load AIME 2024/2025 problems from public Hugging Face dataset mirrors."""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Any

import requests


DATASETS_SERVER_ROWS = "https://datasets-server.huggingface.co/rows"

SOURCES: dict[str, dict[str, str]] = {
    "aime24": {
        "dataset": "Maxwell-Jia/AIME_2024",
        "config": "default",
        "split": "train",
    },
    "aime25": {
        "dataset": "math-ai/aime25",
        "config": "default",
        "split": "test",
    },
}


@dataclass(frozen=True)
class AimeProblem:
    dataset: str
    problem_id: str
    question: str
    answer: str
    solution: str | None = None
    source_metadata: dict[str, Any] | None = None


def fetch_dataset(dataset_key: str, cache_dir: Path, refresh: bool = False) -> list[AimeProblem]:
    if dataset_key not in SOURCES:
        known = ", ".join(sorted(SOURCES))
        raise ValueError(f"Unknown dataset {dataset_key!r}. Expected one of: {known}")

    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{dataset_key}.json"
    if cache_path.exists() and not refresh:
        rows = json.loads(cache_path.read_text(encoding="utf-8"))
    else:
        spec = SOURCES[dataset_key]
        rows = fetch_hf_rows(
            dataset=spec["dataset"],
            config=spec["config"],
            split=spec["split"],
        )
        cache_path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    return [normalize_row(dataset_key, idx, row) for idx, row in enumerate(rows, start=1)]


def fetch_hf_rows(dataset: str, config: str, split: str) -> list[dict[str, Any]]:
    offset = 0
    length = 100
    all_rows: list[dict[str, Any]] = []

    while True:
        response = requests.get(
            DATASETS_SERVER_ROWS,
            params={
                "dataset": dataset,
                "config": config,
                "split": split,
                "offset": offset,
                "length": length,
            },
            timeout=60,
        )
        response.raise_for_status()
        payload = response.json()
        rows = [item["row"] for item in payload.get("rows", [])]
        all_rows.extend(rows)

        num_rows_total = payload.get("num_rows_total")
        offset += len(rows)
        if not rows or (num_rows_total is not None and offset >= num_rows_total):
            break

    if not all_rows:
        raise RuntimeError(f"No rows returned for {dataset}/{config}/{split}")
    return all_rows


def normalize_row(dataset_key: str, index: int, row: dict[str, Any]) -> AimeProblem:
    question = first_text(row, "Problem", "problem", "question", "Question")
    answer = first_text(row, "Answer", "answer", "final_answer", "target")
    solution = optional_text(row, "Solution", "solution", "rationale", "cot")

    if not question:
        raise ValueError(f"{dataset_key} row {index} has no question field: {row}")
    if not answer:
        answer = extract_answer(solution or "")
    if not answer:
        raise ValueError(f"{dataset_key} row {index} has no answer field: {row}")

    problem_id = optional_text(row, "ID", "id", "problem_id") or f"{dataset_key}_{index:02d}"
    if dataset_key == "aime25" and problem_id.isdigit():
        problem_id = f"{dataset_key}_{index:02d}"
    return AimeProblem(
        dataset=dataset_key,
        problem_id=str(problem_id),
        question=question.strip(),
        answer=normalize_answer(answer),
        solution=solution.strip() if solution else None,
        source_metadata=row,
    )


def first_text(row: dict[str, Any], *keys: str) -> str:
    value = optional_text(row, *keys)
    return value or ""


def optional_text(row: dict[str, Any], *keys: str) -> str | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value)
    return None


def normalize_answer(answer: str) -> str:
    match = re.search(r"-?\d+", answer)
    if match:
        return match.group(0)
    return answer.strip()


def extract_answer(text: str) -> str:
    if not text:
        return ""
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        return boxed[-1]
    numbers = re.findall(r"-?\d+", text)
    return numbers[-1] if numbers else ""
