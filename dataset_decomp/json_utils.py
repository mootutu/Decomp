"""Utilities for extracting JSON from LLM responses."""

from __future__ import annotations

import json
import re
from typing import Any


def parse_json_object(text: str) -> dict[str, Any]:
    cleaned = strip_code_fence(text)
    candidates = [cleaned]
    try:
        extracted = extract_balanced_object(cleaned)
    except ValueError:
        extracted = None
    if extracted and extracted != cleaned:
        candidates.append(extracted)

    last_error: json.JSONDecodeError | None = None
    for candidate in candidates:
        for repaired in [escape_invalid_backslashes(candidate), candidate]:
            try:
                value = json.loads(repaired)
                break
            except json.JSONDecodeError as exc:
                last_error = exc
        else:
            continue
        break
    else:
        if last_error is None:
            raise ValueError("No JSON object found in response")
        raise last_error

    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object, got {type(value).__name__}")
    return value


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", stripped, flags=re.DOTALL)
    return match.group(1).strip() if match else stripped


def extract_balanced_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise ValueError("No JSON object found in response")

    depth = 0
    in_string = False
    escape = False
    for idx, char in enumerate(text[start:], start=start):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : idx + 1]

    raise ValueError("Unterminated JSON object in response")


def escape_invalid_backslashes(text: str) -> str:
    r"""Escape LaTeX-style backslashes that break JSON strings.

    LLMs often return JSON-shaped text containing LaTeX snippets such as
    ``\left`` or ``\{``. JSON only allows a small set of escapes, so when
    repair is needed we protect every backslash that is not already starting
    a valid JSON escape.
    """

    return re.sub(r'\\(?!["\\/u])', r"\\\\", text)
