from __future__ import annotations

import argparse
import json
from pathlib import Path

from duckdb_docs_assistant.evaluation import load_jsonl
from duckdb_docs_assistant.metrics import aggregate_metrics, ranking_metrics
from duckdb_docs_assistant.retrieval import BM25Index

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = (1, 3, 5, 10)


def _render_report(report: dict) -> str:
    lines = [
        "# BM25 baseline",
        "",
        (
            "Okapi BM25 indexes the chunk title, heading path and text. Metrics are "
            "macro-averaged over answerable queries only."
        ),
        "",
        "| Language | Queries | Results | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group_name in ("overall", "en", "ru"):
        metrics = report["metrics"][group_name]
        lines.append(
            f"| {group_name} | {metrics['queries']} | {metrics['queries_with_results']} | "
            f"{metrics['recall@5']:.4f} | {metrics['recall@10']:.4f} | "
            f"{metrics['mrr@10']:.4f} | {metrics['ndcg@10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Missed queries at K=10",
            "",
        ]
    )
    misses = [
        row for row in report["per_query"] if row["metrics"]["recall@10"] == 0.0
    ]
    if not misses:
        lines.append("None.")
    else:
        for row in misses:
            lines.append(f"- `{row['query_id']}` ({row['language']}): {row['question']}")
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            (
                "The documentation is English. English BM25 measures the lexical baseline, "
                "while Russian BM25 is intentionally a cross-lingual stress test. A large "
                "language gap supports using multilingual dense retrieval or explicit query "
                "translation."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate an in-memory BM25 baseline.")
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
    parser.add_argument(
        "--report-dir", type=Path, default=PROJECT_ROOT / "reports"
    )
    parser.add_argument("--k1", type=float, default=1.5)
    parser.add_argument("--b", type=float, default=0.75)
    args = parser.parse_args()

    corpus = load_jsonl(args.corpus)
    questions = [row for row in load_jsonl(args.questions) if row["answerable"]]
    qrels = {
        row["query_id"]: set(row["relevant_chunk_ids"]) for row in load_jsonl(args.qrels)
    }
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in corpus}
    index = BM25Index(corpus, k1=args.k1, b=args.b)

    per_query = []
    run_rows = []
    for question in questions:
        results = index.search(question["question"], top_k=max(CUTOFFS))
        retrieved_ids = [result.chunk_id for result in results]
        metrics = ranking_metrics(retrieved_ids, qrels[question["query_id"]], CUTOFFS)
        per_query.append(
            {
                "query_id": question["query_id"],
                "language": question["language"],
                "question": question["question"],
                "relevant_count": len(qrels[question["query_id"]]),
                "retrieved_ids": retrieved_ids,
                "metrics": {key: round(value, 4) for key, value in metrics.items()},
            }
        )
        for result in results:
            chunk = chunks_by_id[result.chunk_id]
            run_rows.append(
                {
                    "query_id": question["query_id"],
                    "rank": result.rank,
                    "score": result.score,
                    "chunk_id": result.chunk_id,
                    "source_path": chunk["source_path"],
                    "section": chunk["section"],
                    "relevant": result.chunk_id in qrels[question["query_id"]],
                }
            )

    report = {
        "retriever": "okapi_bm25",
        "parameters": {"k1": args.k1, "b": args.b, "cutoffs": list(CUTOFFS)},
        "corpus_chunks": len(corpus),
        "metrics": aggregate_metrics(per_query, CUTOFFS),
        "per_query": per_query,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "bm25_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.report_dir / "bm25_metrics.md").write_text(
        _render_report(report), encoding="utf-8"
    )
    with (args.report_dir / "bm25_run.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for row in run_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
