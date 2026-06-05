"""AveMujica OpenAI-compatible client helpers."""

from __future__ import annotations

from openai import OpenAI


BASE_URL = "https://api.avemujica.moe/v1"
DEFAULT_MODEL = "gpt-5.5"
USER_AGENT = "curl/8.7.1"


def make_client(api_key: str) -> OpenAI:
    return OpenAI(
        api_key=api_key,
        base_url=BASE_URL,
        default_headers={
            "Accept": "application/json",
            "User-Agent": USER_AGENT,
        },
    )

