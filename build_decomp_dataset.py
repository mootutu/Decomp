#!/usr/bin/env python3
"""Build decomposed AIME datasets with AveMujicaAPI."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys

from openai import APIConnectionError, APIError

from dataset_decomp.avemujica import DEFAULT_MODEL, make_client
from dataset_decomp.env import load_dotenv
from dataset_decomp.pipeline import BuildConfig, build_datasets


def main() -> int:
    load_dotenv()

    parser = argparse.ArgumentParser(description="Build AIME dataset-decomposition artifacts.")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["aime24", "aime25"],
        choices=["aime24", "aime25"],
        help="Datasets to build.",
    )
    parser.add_argument("--model", default=os.getenv("AVEMUJICA_MODEL", DEFAULT_MODEL))
    parser.add_argument("--api-key", default=os.getenv("AVEMUJICA_API_KEY"))
    parser.add_argument("--output-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--processed-subdir",
        default="processed_depth2",
        help="Subdirectory under output-dir for final artifacts.",
    )
    parser.add_argument("--max-depth", type=int, default=2)
    parser.add_argument("--max-problems", type=int, default=None)
    parser.add_argument("--refresh-raw", action="store_true")
    parser.add_argument("--refresh-llm", action="store_true")
    parser.add_argument("--no-verify", action="store_true", help="Skip LLM verification of generated subproblems.")
    parser.add_argument(
        "--verify-retries",
        type=int,
        default=3,
        help="Regenerate and re-verify a generated subproblem this many times after verification fails.",
    )
    parser.add_argument("--difficulty-structural-weight", type=float, default=1.0)
    parser.add_argument("--difficulty-concept-weight", type=float, default=1.0)
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help="Number of source problems to process concurrently. Defaults to serial execution.",
    )
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--dry-run", action="store_true", help="Fetch raw data and write root-only artifacts.")
    args = parser.parse_args()

    if args.max_depth < 0:
        print("--max-depth must be >= 0", file=sys.stderr)
        return 2
    if args.verify_retries < 0:
        print("--verify-retries must be >= 0", file=sys.stderr)
        return 2
    if args.max_workers < 1:
        print("--max-workers must be >= 1", file=sys.stderr)
        return 2
    if not args.dry_run and not args.api_key:
        print("Missing API key. Set AVEMUJICA_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    client = None if args.dry_run else make_client(args.api_key)
    config = BuildConfig(
        datasets=args.datasets,
        output_dir=args.output_dir,
        model=args.model,
        processed_subdir=args.processed_subdir,
        max_depth=args.max_depth,
        max_problems=args.max_problems,
        refresh_raw=args.refresh_raw,
        refresh_llm=args.refresh_llm,
        verify_subproblems=not args.no_verify,
        verify_retries=args.verify_retries,
        difficulty_structural_weight=args.difficulty_structural_weight,
        difficulty_concept_weight=args.difficulty_concept_weight,
        max_workers=args.max_workers,
        sleep_seconds=args.sleep_seconds,
        dry_run=args.dry_run,
    )

    try:
        build_datasets(client, config)
    except APIConnectionError as exc:
        print(f"Request failed: {exc}", file=sys.stderr)
        return 1
    except APIError as exc:
        print(f"HTTP {exc.status_code}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
