from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path

from duckdb_docs_assistant.evaluation import load_jsonl
from duckdb_docs_assistant.metrics import aggregate_metrics, ranking_metrics
from duckdb_docs_assistant.reranker import (
    CrossEncoderScorer,
    RerankCandidate,
    rerank,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = (1, 3, 5, 10)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)] if ordered else 0.0


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _render_report(report: dict) -> str:
    lines = [
        "# Multilingual cross-encoder reranker",
        "",
        f"Model: `{report['model']['name']}` at revision `{report['model']['revision']}`.",
        (
            "The cross-encoder scores each query-passage pair jointly and reranks the top "
            f"{report['parameters']['candidate_pool']} Hybrid RRF candidates."
        ),
        "",
        "| Language | Queries | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group_name in ("overall", "en", "ru"):
        metrics = report["metrics"][group_name]
        lines.append(
            f"| {group_name} | {metrics['queries']} | {metrics['recall@5']:.4f} | "
            f"{metrics['recall@10']:.4f} | {metrics['mrr@10']:.4f} | "
            f"{metrics['ndcg@10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Runtime",
            "",
            f"- Device: `{report['runtime']['device']}`",
            f"- Model load: {report['runtime']['model_load_seconds']:.3f} s",
            f"- Query reranking median: {report['runtime']['query_latency_ms_median']:.3f} ms",
            f"- Query reranking P95: {report['runtime']['query_latency_ms_p95']:.3f} ms",
            "",
        ]
    )
    return "\n".join(lines)


def _render_comparison(reports: list[tuple[str, dict]]) -> str:
    lines = [
        "# Retrieval comparison",
        "",
        "| Retriever | Language | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for retriever_name, report in reports:
        for language in ("en", "ru"):
            metrics = report["metrics"][language]
            lines.append(
                f"| {retriever_name} | {language} | {metrics['recall@5']:.4f} | "
                f"{metrics['recall@10']:.4f} | {metrics['mrr@10']:.4f} | "
                f"{metrics['ndcg@10']:.4f} |"
            )
    lines.extend(
        [
            "",
            "All stages use the same answerable queries, qrels and cutoffs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate cross-encoder reranking.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "reranker.json"
    )
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
        "--qrels", type=Path, default=PROJECT_ROOT / "data" / "eval" / "qrels.jsonl"
    )
    parser.add_argument(
        "--hybrid-run",
        type=Path,
        default=PROJECT_ROOT / "reports" / "hybrid_run.jsonl",
    )
    parser.add_argument(
        "--model-cache", type=Path, default=PROJECT_ROOT / ".hf_cache"
    )
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    corpus = load_jsonl(args.corpus)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in corpus}
    questions = [row for row in load_jsonl(args.questions) if row["answerable"]]
    qrels = {
        row["query_id"]: set(row["relevant_chunk_ids"]) for row in load_jsonl(args.qrels)
    }
    candidate_rows: dict[str, list[dict]] = defaultdict(list)
    for row in load_jsonl(args.hybrid_run):
        candidate_rows[row["query_id"]].append(row)
    for rows in candidate_rows.values():
        rows.sort(key=lambda row: row["rank"])

    model_started = time.perf_counter()
    scorer = CrossEncoderScorer(
        config["model"],
        config["revision"],
        config["device"],
        args.model_cache,
        config["max_length"],
    )
    model_load_seconds = time.perf_counter() - model_started

    per_query = []
    run_rows = []
    query_times_ms: list[float] = []
    pool_size = config["candidate_pool"]
    for question in questions:
        rows = candidate_rows.get(question["query_id"], [])[:pool_size]
        if len(rows) != pool_size:
            raise ValueError(
                f"Expected {pool_size} candidates for {question['query_id']}, got {len(rows)}"
            )
        candidates = [
            RerankCandidate(row["chunk_id"], rank=row["rank"], score=row["score"])
            for row in rows
        ]
        started = time.perf_counter()
        results = rerank(
            question["question"],
            candidates,
            chunks_by_id,
            scorer,
            batch_size=config["batch_size"],
            top_k=max(CUTOFFS),
        )
        query_times_ms.append((time.perf_counter() - started) * 1000)
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
                    "retrieval_rank": result.retrieval_rank,
                    "retrieval_score": result.retrieval_score,
                    "chunk_id": result.chunk_id,
                    "source_path": chunk["source_path"],
                    "section": chunk["section"],
                    "relevant": result.chunk_id in qrels[question["query_id"]],
                }
            )

    report = {
        "retriever": "hybrid_rrf_cross_encoder",
        "model": {"name": config["model"], "revision": config["revision"]},
        "parameters": {
            "candidate_pool": pool_size,
            "batch_size": config["batch_size"],
            "max_length": config["max_length"],
            "text_template": config["text_template"],
            "cutoffs": list(CUTOFFS),
            "candidate_run": str(args.hybrid_run.relative_to(PROJECT_ROOT)),
            "candidate_run_sha256": _sha256(args.hybrid_run),
        },
        "corpus_chunks": len(corpus),
        "metrics": aggregate_metrics(per_query, CUTOFFS),
        "runtime": {
            "device": scorer.device,
            "model_load_seconds": round(model_load_seconds, 3),
            "query_latency_ms_mean": round(statistics.fmean(query_times_ms), 3),
            "query_latency_ms_median": round(statistics.median(query_times_ms), 3),
            "query_latency_ms_p95": round(_percentile(query_times_ms, 0.95), 3),
        },
        "per_query": per_query,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "reranker_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.report_dir / "reranker_metrics.md").write_text(
        _render_report(report), encoding="utf-8"
    )
    with (args.report_dir / "reranker_run.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for row in run_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    comparison_inputs = []
    for name, filename in (
        ("BM25", "bm25_metrics.json"),
        ("Dense E5", "dense_metrics.json"),
        ("Hybrid RRF", "hybrid_metrics.json"),
    ):
        path = args.report_dir / filename
        if path.exists():
            comparison_inputs.append((name, json.loads(path.read_text(encoding="utf-8"))))
    comparison_inputs.append(("Reranker", report))
    (args.report_dir / "retrieval_comparison.md").write_text(
        _render_comparison(comparison_inputs), encoding="utf-8"
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
