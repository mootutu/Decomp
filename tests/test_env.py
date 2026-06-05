from __future__ import annotations

import os
from pathlib import Path

from dataset_decomp.env import load_dotenv


def test_load_dotenv_sets_missing_values(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "\n".join(
            [
                "# local settings",
                "AVEMUJICA_API_KEY='abc123'",
                'AVEMUJICA_MODEL="gpt-5.5"',
            ]
        ),
        encoding="utf-8",
    )

    old_key = os.environ.pop("AVEMUJICA_API_KEY", None)
    old_model = os.environ.pop("AVEMUJICA_MODEL", None)
    try:
        load_dotenv(env_path)

        assert os.environ["AVEMUJICA_API_KEY"] == "abc123"
        assert os.environ["AVEMUJICA_MODEL"] == "gpt-5.5"
    finally:
        restore_env("AVEMUJICA_API_KEY", old_key)
        restore_env("AVEMUJICA_MODEL", old_model)


def test_load_dotenv_does_not_override_environment(tmp_path: Path) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("AVEMUJICA_MODEL=gpt-5.5\n", encoding="utf-8")

    old_model = os.environ.get("AVEMUJICA_MODEL")
    os.environ["AVEMUJICA_MODEL"] = "custom-model"
    try:
        load_dotenv(env_path)

        assert os.environ["AVEMUJICA_MODEL"] == "custom-model"
    finally:
        restore_env("AVEMUJICA_MODEL", old_model)


def restore_env(key: str, value: str | None) -> None:
    if value is None:
        os.environ.pop(key, None)
    else:
        os.environ[key] = value
