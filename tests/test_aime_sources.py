from __future__ import annotations

from dataset_decomp.aime_sources import extract_answer, normalize_row


def test_normalize_aime24_row() -> None:
    problem = normalize_row(
        "aime24",
        1,
        {
            "ID": "2024-I-1",
            "Problem": "What is 2+2?",
            "Solution": "Compute directly. \\boxed{004}",
            "Answer": 4,
        },
    )

    assert problem.problem_id == "2024-I-1"
    assert problem.question == "What is 2+2?"
    assert problem.answer == "4"
    assert problem.solution == "Compute directly. \\boxed{004}"


def test_normalize_aime25_numeric_id() -> None:
    problem = normalize_row(
        "aime25",
        7,
        {
            "id": "6",
            "problem": "Find x.",
            "answer": "123",
        },
    )

    assert problem.problem_id == "aime25_07"
    assert problem.answer == "123"


def test_extract_answer_prefers_boxed_value() -> None:
    assert extract_answer("first 12 then \\boxed{033}") == "033"
