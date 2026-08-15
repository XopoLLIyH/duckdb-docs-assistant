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
from duckdb_docs_assistant.metrics import aggregate_metrics, ranking_metrics

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUTOFFS = (1, 3, 5, 10)


def _render_dense_report(report: dict) -> str:
    lines = [
        "# Multilingual dense retrieval",
        "",
        f"Model: `{report['model']['name']}` at revision `{report['model']['revision']}`.",
        (
            "Documents use the `passage:` prefix; questions use `query:`. Embeddings are L2 "
            "normalized and ranked by dot product."
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
            f"- Document cache reused: `{report['runtime']['document_cache_reused']}`",
            f"- Document encoding: {report['runtime']['document_encoding_seconds']:.3f} s",
            f"- Query encoding median: {report['runtime']['query_encoding_ms_median']:.3f} ms",
            f"- Query encoding P95: {report['runtime']['query_encoding_ms_p95']:.3f} ms",
            "",
        ]
    )
    return "\n".join(lines)


def _render_comparison(
    dense: dict,
    bm25: dict,
    hybrid: dict | None = None,
    reranker: dict | None = None,
) -> str:
    lines = [
        "# Retrieval comparison",
        "",
        "| Retriever | Language | Recall@5 | Recall@10 | MRR@10 | nDCG@10 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    reports = [("BM25", bm25), ("Dense E5", dense)]
    if hybrid is not None:
        reports.append(("Hybrid RRF", hybrid))
    if reranker is not None:
        reports.append(("Reranker", reranker))
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
            "Dense and BM25 use the same answerable queries, qrels and cutoffs.",
            "",
        ]
    )
    return "\n".join(lines)


def _percentile(values: list[float], percentile: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * percentile)] if ordered else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate multilingual dense retrieval.")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "dense.json"
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
        "--qrels",
        type=Path,
        default=PROJECT_ROOT / "data" / "eval" / "qrels.jsonl",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=PROJECT_ROOT / "storage" / "embeddings",
    )
    parser.add_argument(
        "--model-cache",
        type=Path,
        default=PROJECT_ROOT / ".hf_cache",
    )
    parser.add_argument(
        "--report-dir", type=Path, default=PROJECT_ROOT / "reports"
    )
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="utf-8"))
    chunks = load_jsonl(args.corpus)
    questions = [row for row in load_jsonl(args.questions) if row["answerable"]]
    qrels = {
        row["query_id"]: set(row["relevant_chunk_ids"]) for row in load_jsonl(args.qrels)
    }
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    encoder = SentenceTransformerEncoder(
        config["model"],
        config["revision"],
        config["device"],
        args.model_cache,
        config["normalize_embeddings"],
    )
    encoding_started = time.perf_counter()
    document_embeddings, cache_manifest, cache_reused = load_or_encode_documents(
        chunks, encoder, config, args.cache_dir
    )
    document_encoding_seconds = time.perf_counter() - encoding_started
    index = DenseIndex(chunks, document_embeddings)

    per_query = []
    run_rows = []
    query_encoding_times_ms: list[float] = []
    for question in questions:
        started = time.perf_counter()
        query_embedding = encoder.encode(
            [config["query_prefix"] + question["question"]], batch_size=1
        )
        query_encoding_times_ms.append((time.perf_counter() - started) * 1000)
        results = index.search(query_embedding, top_k=max(CUTOFFS))
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
        "retriever": "multilingual_dense",
        "model": {
            "name": config["model"],
            "revision": config["revision"],
            "dimension": encoder.dimension,
        },
        "parameters": {"cutoffs": list(CUTOFFS), "cache_manifest": cache_manifest},
        "corpus_chunks": len(chunks),
        "metrics": aggregate_metrics(per_query, CUTOFFS),
        "runtime": {
            "device": encoder.device,
            "document_cache_reused": cache_reused,
            "document_encoding_seconds": round(document_encoding_seconds, 3),
            "query_encoding_ms_mean": round(statistics.fmean(query_encoding_times_ms), 3),
            "query_encoding_ms_median": round(statistics.median(query_encoding_times_ms), 3),
            "query_encoding_ms_p95": round(_percentile(query_encoding_times_ms, 0.95), 3),
        },
        "per_query": per_query,
    }
    args.report_dir.mkdir(parents=True, exist_ok=True)
    (args.report_dir / "dense_metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (args.report_dir / "dense_metrics.md").write_text(
        _render_dense_report(report), encoding="utf-8"
    )
    with (args.report_dir / "dense_run.jsonl").open(
        "w", encoding="utf-8", newline="\n"
    ) as output:
        for row in run_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    bm25_path = args.report_dir / "bm25_metrics.json"
    if bm25_path.exists():
        bm25 = json.loads(bm25_path.read_text(encoding="utf-8"))
        hybrid_path = args.report_dir / "hybrid_metrics.json"
        hybrid = (
            json.loads(hybrid_path.read_text(encoding="utf-8"))
            if hybrid_path.exists()
            else None
        )
        reranker_path = args.report_dir / "reranker_metrics.json"
        reranker = (
            json.loads(reranker_path.read_text(encoding="utf-8"))
            if reranker_path.exists()
            else None
        )
        (args.report_dir / "retrieval_comparison.md").write_text(
            _render_comparison(report, bm25, hybrid, reranker), encoding="utf-8"
        )
    print(json.dumps(report["metrics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
