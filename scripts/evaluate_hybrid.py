from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from duckdb_docs_assistant.dense import (
    DenseIndex,
    SentenceTransformerEncoder,
    load_or_encode_documents,
)
from duckdb_docs_assistant.evaluation import load_jsonl
from duckdb_docs_assistant.fusion import detect_query_language, reciprocal_rank_fusion
from duckdb_docs_assistant.metrics import aggregate_metrics, ranking_metrics
from duckdb_docs_assistant.retrieval import BM25Index

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = (1, 3, 5, 10)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)] if ordered else 0.0


def _render_report(report: dict) -> str:
    parameters = report["parameters"]
    lines = [
        "# Hybrid retrieval with Reciprocal Rank Fusion",
        "",
        (
            "BM25 and multilingual E5 each retrieve an independent candidate pool. "
            "Reciprocal Rank Fusion combines ranks rather than incomparable raw scores."
        ),
        "",
        f"- RRF k: `{parameters['rrf_k']}`",
        f"- Candidate pool per retriever: `{parameters['candidate_pool']}`",
        f"- English weights: `{parameters['weights_by_language']['en']}`",
        f"- Russian weights: `{parameters['weights_by_language']['ru']}`",
        "- Weight selection status: development seed; held-out validation required",
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
            f"- Document cache reused: `{report['runtime']['document_cache_reused']}`",
            f"- Query latency median: {report['runtime']['query_latency_ms_median']:.3f} ms",
            f"- Query latency P95: {report['runtime']['query_latency_ms_p95']:.3f} ms",
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
            "All retrievers use the same answerable queries, qrels and cutoffs.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate BM25 + dense RRF retrieval.")
    parser.add_argument(
        "--hybrid-config", type=Path, default=PROJECT_ROOT / "config" / "hybrid.json"
    )
    parser.add_argument(
        "--dense-config", type=Path, default=PROJECT_ROOT / "config" / "dense.json"
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
        "--embedding-cache",
        type=Path,
        default=PROJECT_ROOT / "storage" / "embeddings",
    )
    parser.add_argument(
        "--model-cache", type=Path, default=PROJECT_ROOT / ".hf_cache"
    )
    parser.add_argument("--report-dir", type=Path, default=PROJECT_ROOT / "reports")
    args = parser.parse_args()

    hybrid_config = json.loads(args.hybrid_config.read_text(encoding="utf-8"))
    dense_config = json.loads(args.dense_config.read_text(encoding="utf-8"))
    corpus = load_jsonl(args.corpus)
    questions = load_jsonl(args.questions)
    qrels = {
        row["query_id"]: set(row["relevant_chunk_ids"]) for row in load_jsonl(args.qrels)
    }
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in corpus}

    bm25_index = BM25Index(corpus)
    encoder = SentenceTransformerEncoder(
        dense_config["model"],
        dense_config["revision"],
        dense_config["device"],
        args.model_cache,
        dense_config["normalize_embeddings"],
    )
    document_embeddings, cache_manifest, cache_reused = load_or_encode_documents(
        corpus, encoder, dense_config, args.embedding_cache
    )
    dense_index = DenseIndex(corpus, document_embeddings)

    pool_size = hybrid_config["candidate_pool"]
    weights_by_language = hybrid_config["weights_by_language"]
    per_query = []
    run_rows = []
    query_times_ms: list[float] = []
    for question in questions:
        started = time.perf_counter()
        bm25_results = bm25_index.search(question["question"], top_k=pool_size)
        query_embedding = encoder.encode(
            [dense_config["query_prefix"] + question["question"]], batch_size=1
        )
        dense_results = dense_index.search(query_embedding, top_k=pool_size)
        detected_language = detect_query_language(question["question"])
        weights = weights_by_language.get(detected_language, weights_by_language["default"])
        results = reciprocal_rank_fusion(
            {"bm25": bm25_results, "dense": dense_results},
            rrf_k=hybrid_config["rrf_k"],
            weights=weights,
            top_k=pool_size,
        )
        query_times_ms.append((time.perf_counter() - started) * 1000)
        retrieved_ids = [result.chunk_id for result in results]
        metrics = ranking_metrics(retrieved_ids, qrels[question["query_id"]], CUTOFFS)
        per_query.append(
            {
                "query_id": question["query_id"],
                "language": question["language"],
                "detected_language": detected_language,
                "question": question["question"],
                "answerable": question["answerable"],
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
                    "source_ranks": result.source_ranks,
                    "chunk_id": result.chunk_id,
                    "source_path": chunk["source_path"],
                    "section": chunk["section"],
                    "relevant": result.chunk_id in qrels[question["query_id"]],
                }
            )

    report = {
        "retriever": "bm25_dense_rrf",
        "model": {
            "name": dense_config["model"],
            "revision": dense_config["revision"],
            "dimension": encoder.dimension,
        },
        "parameters": {
            "rrf_k": hybrid_config["rrf_k"],
            "candidate_pool": pool_size,
            "weights_by_language": weights_by_language,
            "cutoffs": list(CUTOFFS),
            "embedding_cache_manifest": cache_manifest,
        },
        "corpus_chunks": len(corpus),
        "candidate_queries": len(questions),
        "answerable_metric_queries": sum(row["answerable"] for row in questions),
        "metrics": aggregate_metrics(
            [row for row in per_query if row["answerable"]], CUTOFFS
        ),
        "runtime": {
            "device": encoder.device,
            "document_cache_reused": cache_reused,
            "query_latency_ms_mean": round(statistics.fmean(query_times_ms), 3),
            "query_latency_ms_median": round(statistics.median(query_times_ms), 3),
            "query_latency_ms_p95": round(_percentile(query_times_ms, 0.95), 3),
        },
        "per_query": per_query,
    }

    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "hybrid_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.report_dir / "hybrid_metrics.md").write_text(
        _render_report(report), encoding="utf-8"
    )
    with (args.report_dir / "hybrid_run.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for row in run_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    comparison_inputs = []
    for name, filename in (
        ("BM25", "bm25_metrics.json"),
        ("Dense E5", "dense_metrics.json"),
    ):
        path = args.report_dir / filename
        if path.exists():
            comparison_inputs.append((name, json.loads(path.read_text(encoding="utf-8"))))
    comparison_inputs.append(("Hybrid RRF", report))
    reranker_path = args.report_dir / "reranker_metrics.json"
    if reranker_path.exists():
        comparison_inputs.append(
            ("Reranker", json.loads(reranker_path.read_text(encoding="utf-8")))
        )
    (args.report_dir / "retrieval_comparison.md").write_text(
        _render_comparison(comparison_inputs), encoding="utf-8"
    )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
