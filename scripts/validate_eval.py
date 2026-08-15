from __future__ import annotations

import argparse
import json
from pathlib import Path

from duckdb_docs_assistant.evaluation import build_qrels, load_jsonl, validate_evaluation

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate evaluation questions and build qrels.")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "duckdb_docs.jsonl",
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "questions.jsonl",
    )
    parser.add_argument(
        "--qrels",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "qrels.jsonl",
    )
    args = parser.parse_args()

    corpus = load_jsonl(args.corpus)
    questions = load_jsonl(args.questions)
    errors = validate_evaluation(questions, corpus)
    if errors:
        raise SystemExit("Evaluation validation failed:\n- " + "\n- ".join(errors))

    qrels = build_qrels(questions, corpus)
    with args.qrels.open("w", encoding="utf-8", newline="\n") as output:
        for row in qrels:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    answerable = sum(question["answerable"] for question in questions)
    print(
        json.dumps(
            {
                "questions": len(questions),
                "answerable": answerable,
                "unanswerable": len(questions) - answerable,
                "qrels": sum(bool(row["relevant_chunk_ids"]) for row in qrels),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
