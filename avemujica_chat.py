#!/usr/bin/env python3
"""Call AveMujicaAPI's OpenAI-compatible chat completions endpoint."""

from __future__ import annotations

import argparse
import os
import sys

from openai import APIConnectionError, APIError, OpenAI


BASE_URL = "https://api.avemujica.moe/v1"
API_KEY = os.getenv("AVEMUJICA_API_KEY", "")
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


def list_models(client: OpenAI) -> list[str]:
    models = client.models.list()
    return [model.id for model in models.data]


def chat(client: OpenAI, model: str, prompt: str, system_prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
    )

    content = response.choices[0].message.content
    if content is None:
        raise RuntimeError("Model returned an empty message.")
    return content


def main() -> int:
    parser = argparse.ArgumentParser(description="Chat with an AveMujicaAPI model.")
    parser.add_argument("prompt", nargs="?", help="User message to send to the model.")
    parser.add_argument("--model", default=os.getenv("AVEMUJICA_MODEL"), help="Model id to use.")
    parser.add_argument("--api-key", default=API_KEY, help="API key. Defaults to AVEMUJICA_API_KEY.")
    parser.add_argument("--list-models", action="store_true", help="Print available model ids and exit.")
    parser.add_argument(
        "--system",
        default="You are a concise and helpful assistant.",
        help="System prompt.",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Missing API key. Set AVEMUJICA_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    try:
        client = make_client(args.api_key)

        if args.list_models:
            for model_id in list_models(client):
                print(model_id)
            return 0

        prompt = args.prompt or input("You: ").strip()
        if not prompt:
            print("Prompt is empty.", file=sys.stderr)
            return 2

        model = args.model
        if not model:
            models = list_models(client)
            if not models:
                print("No available models returned by /v1/models.", file=sys.stderr)
                return 1
            model = models[0]
            print(f"Using model: {model}", file=sys.stderr)

        print(chat(client, model, prompt, args.system))
        return 0
    except APIConnectionError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    except APIError as exc:
        message = str(exc)
        if exc.status_code == 403 and "1010" in message:
            print(
                "HTTP 403 from Cloudflare before reaching AveMujica API: "
                f"{message}. Try another network, disable proxy/VPN, "
                "or ask AveMujica support to allow your client/IP.",
                file=sys.stderr,
            )
            return 1
        print(f"HTTP {exc.status_code}: {message}", file=sys.stderr)
        return 1
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
