from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from duckdb_docs_assistant.evaluation import load_jsonl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DIMENSIONS = ("correctness", "completeness", "citation_entailment", "language_quality")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _scope_metrics(labels: list[dict[str, Any]]) -> dict[str, Any]:
    dimension_scores = {
        dimension: sum(row[dimension] for row in labels) for dimension in DIMENSIONS
    }
    total = sum(dimension_scores.values())
    maximum = len(labels) * len(DIMENSIONS) * 2
    return {
        "queries": len(labels),
        "normalized_score": _ratio(total, maximum),
        "strict_pass_rate": _ratio(
            sum(all(row[dimension] == 2 for dimension in DIMENSIONS) for row in labels),
            len(labels),
        ),
        "dimension_full_score_rates": {
            dimension: _ratio(sum(row[dimension] == 2 for row in labels), len(labels))
            for dimension in DIMENSIONS
        },
        "dimension_scores": dimension_scores,
        "score": total,
        "max_score": maximum,
    }


def _render(report: dict[str, Any], labels: list[dict[str, Any]]) -> str:
    lines = [
        "# Manual generation review",
        "",
        (
            "Rubric: `data/eval/generation_review_rubric.md`. One non-independent reviewer; "
            "scores require a second review before use as a reliable human benchmark."
        ),
        "",
        "| Scope | Queries | Normalized score | Strict pass | Correct | Complete | Entailed | Language |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for scope in ("overall", "en", "ru"):
        metrics = report["metrics"][scope]
        rates = metrics["dimension_full_score_rates"]
        lines.append(
            f"| {scope} | {metrics['queries']} | {metrics['normalized_score']:.4f} | "
            f"{metrics['strict_pass_rate']:.4f} | {rates['correctness']:.4f} | "
            f"{rates['completeness']:.4f} | {rates['citation_entailment']:.4f} | "
            f"{rates['language_quality']:.4f} |"
        )
    audit = report["qrel_audit"]
    lines.extend(
        [
            "",
            "## Qrel audit",
            "",
            f"- Answerable questions without an annotated qrel in context: {audit['qrel_misses']}",
            f"- Of those, answers receiving full correctness: {audit['full_correct_despite_qrel_miss']}",
            "- Confirmed retrieval mismatch: `q010_ru` selected the Docker UI subsection for a CLI question.",
            "",
            (
                "The other nine qrel misses cite directly relevant official sections, including CSV "
                "type detection, JSON readers, S3 credentials, Parquet export, and resource limits. "
                "This is evidence that the current qrels are incomplete; they should be independently "
                "adjudicated rather than expanded automatically from model-selected sources."
            ),
            "",
            "## Findings requiring action",
            "",
            "| Query | C | K | E | L | Error types | Review note |",
            "|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for row in labels:
        if all(row[dimension] == 2 for dimension in DIMENSIONS):
            continue
        lines.append(
            f"| {row['query_id']} | {row['correctness']} | {row['completeness']} | "
            f"{row['citation_entailment']} | {row['language_quality']} | "
            f"{', '.join(row['error_types'])} | {row['note']} |"
        )
    lines.extend(
        [
            "",
            "## Error taxonomy",
            "",
        ]
    )
    for error_type, count in report["error_counts"].items():
        lines.append(f"- `{error_type}`: {count}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The normalized score summarizes this rubric only; it is not an accuracy confidence "
                "interval. Protocol validity was 100%, while manual review found one misleading "
                "retrieval result, five incomplete answers, two citation gaps, and two presentation "
                "defects. The next defensible step is a blinded second review plus qrel adjudication."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and summarize manual generation labels.")
    parser.add_argument(
        "--questions", type=Path, default=PROJECT_ROOT / "data" / "eval" / "questions.jsonl"
    )
    parser.add_argument(
        "--generation-run", type=Path, default=PROJECT_ROOT / "reports" / "generation_run.jsonl"
    )
    parser.add_argument(
        "--labels",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "generation_human_labels.jsonl",
    )
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()

    questions = load_jsonl(args.questions)
    generation_rows = load_jsonl(args.generation_run)
    labels = load_jsonl(args.labels)
    questions_by_id = {row["query_id"]: row for row in questions}
    generation_by_id = {row["query_id"]: row for row in generation_rows}
    labels_by_id = {row["query_id"]: row for row in labels}
    expected_ids = set(questions_by_id)
    if set(generation_by_id) != expected_ids or set(labels_by_id) != expected_ids:
        raise ValueError("Questions, generation run, and human labels must have identical query IDs")
    if len(labels_by_id) != len(labels):
        raise ValueError("Human labels contain duplicate query IDs")
    for row in labels:
        for dimension in DIMENSIONS:
            if row.get(dimension) not in {0, 1, 2}:
                raise ValueError(f"Invalid {dimension} score for {row['query_id']}")
        row["language"] = questions_by_id[row["query_id"]]["language"]

    ordered_labels = [labels_by_id[question["query_id"]] for question in questions]
    qrel_misses = [
        row
        for row in generation_rows
        if row["answerable"]
        and not set(row["context_chunk_ids"]) & set(row["relevant_chunk_ids"])
    ]
    error_counts = Counter(
        error_type for row in labels for error_type in row.get("error_types", [])
    )
    report = {
        "rubric": "data/eval/generation_review_rubric.md",
        "review": {
            "version": "manual_single_reviewer_v1",
            "independent": False,
            "blinded": False,
        },
        "inputs": {
            "generation_run_sha256": _sha256(args.generation_run),
            "labels_sha256": _sha256(args.labels),
        },
        "metrics": {
            "overall": _scope_metrics(ordered_labels),
            "en": _scope_metrics([row for row in ordered_labels if row["language"] == "en"]),
            "ru": _scope_metrics([row for row in ordered_labels if row["language"] == "ru"]),
        },
        "qrel_audit": {
            "qrel_misses": len(qrel_misses),
            "full_correct_despite_qrel_miss": sum(
                labels_by_id[row["query_id"]]["correctness"] == 2 for row in qrel_misses
            ),
            "query_ids": [row["query_id"] for row in qrel_misses],
        },
        "error_counts": dict(sorted(error_counts.items())),
        "per_query": ordered_labels,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "generation_human_eval.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.report_dir / "generation_human_eval.md").write_text(
        _render(report, ordered_labels), encoding="utf-8"
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
