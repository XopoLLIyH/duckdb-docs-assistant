from __future__ import annotations

import argparse
import json
from pathlib import Path

from duckdb_docs_assistant.corpus import build_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a versioned DuckDB documentation corpus.")
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config" / "corpus.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / ".cache",
    )
    parser.add_argument(
        "--tree-index",
        type=Path,
        help="Optional GitHub tree API response; avoids downloading the tree index.",
    )
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_corpus(
        config_path=args.config,
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        tree_index=args.tree_index,
        workers=args.workers,
    )
    print(json.dumps(manifest | {"source_paths": "<omitted>"}, indent=2))


if __name__ == "__main__":
    main()
