from __future__ import annotations

import argparse
import json
from pathlib import Path

from duckdb_docs_assistant.analysis import analyze_corpus, render_markdown_report
from duckdb_docs_assistant.evaluation import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the generated DuckDB corpus.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "duckdb_docs.jsonl",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=PROJECT_ROOT / "reports"
    )
    args = parser.parse_args()

    analysis = analyze_corpus(load_jsonl(args.corpus))
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "corpus_analysis.json").write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.report_dir / "corpus_analysis.md").write_text(
        render_markdown_report(analysis), encoding="utf-8"
    )
    print(json.dumps(analysis, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
