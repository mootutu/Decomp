from __future__ import annotations

import pytest

from dataset_decomp.json_utils import parse_json_object


def test_parse_plain_json_object() -> None:
    assert parse_json_object('{"valid": true, "answer": "42"}') == {
        "valid": True,
        "answer": "42",
    }


def test_parse_json_object_from_markdown_fence() -> None:
    assert parse_json_object('```json\n{"x": 1}\n```') == {"x": 1}


def test_parse_json_object_embedded_in_text() -> None:
    assert parse_json_object('Here is the result: {"x": {"y": 2}} done.') == {
        "x": {"y": 2},
    }


def test_parse_json_object_repairs_latex_backslashes() -> None:
    text = r'{"solution": "Use \left(x+1\right) and \frac{1}{2}.", "valid": true}'

    assert parse_json_object(text) == {
        "solution": r"Use \left(x+1\right) and \frac{1}{2}.",
        "valid": True,
    }


def test_parse_json_object_repairs_non_letter_latex_backslashes() -> None:
    text = r'{"solution": "Use set notation \{1,2,3\} and norm \|x\|.", "valid": true}'

    assert parse_json_object(text) == {
        "solution": r"Use set notation \{1,2,3\} and norm \|x\|.",
        "valid": True,
    }


def test_parse_json_object_rejects_arrays() -> None:
    with pytest.raises(ValueError, match="Expected a JSON object"):
        parse_json_object("[1, 2, 3]")
