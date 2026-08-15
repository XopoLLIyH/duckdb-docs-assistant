from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from duckdb_docs_assistant.dense import (
    DenseIndex,
    SentenceTransformerEncoder,
    load_or_encode_documents,
)
from duckdb_docs_assistant.evaluation import load_jsonl
from duckdb_docs_assistant.fusion import detect_query_language, reciprocal_rank_fusion
from duckdb_docs_assistant.generation import (
    ANSWER_SCHEMA,
    SYSTEM_PROMPT,
    build_context,
    build_user_prompt,
    render_answer,
    validate_answer,
)
from duckdb_docs_assistant.ollama_client import OllamaClient
from duckdb_docs_assistant.reranker import (
    CrossEncoderScorer,
    RerankCandidate,
    rerank,
)
from duckdb_docs_assistant.retrieval import BM25Index

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_config(name: str) -> dict:
    path = PROJECT_ROOT / "config" / name
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask the grounded DuckDB assistant.")
    parser.add_argument("question", help="DuckDB question in English or Russian")
    parser.add_argument(
        "--corpus",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "duckdb_docs.jsonl",
    )
    parser.add_argument(
        "--embedding-cache",
        type=Path,
        default=PROJECT_ROOT / "storage" / "embeddings",
    )
    parser.add_argument(
        "--model-cache", type=Path, default=PROJECT_ROOT / ".hf_cache"
    )
    parser.add_argument(
        "--preview-context",
        action="store_true",
        help="Print the grounded prompt without calling Ollama",
    )
    args = parser.parse_args()

    started = time.perf_counter()
    corpus = load_jsonl(args.corpus)
    chunks_by_id = {chunk["chunk_id"]: chunk for chunk in corpus}
    dense_config = _load_config("dense.json")
    hybrid_config = _load_config("hybrid.json")
    reranker_config = _load_config("reranker.json")
    generation_config = _load_config("generation.json")

    bm25_index = BM25Index(corpus)
    dense_encoder = SentenceTransformerEncoder(
        dense_config["model"],
        dense_config["revision"],
        dense_config["device"],
        args.model_cache,
        dense_config["normalize_embeddings"],
    )
    document_embeddings, _, _ = load_or_encode_documents(
        corpus, dense_encoder, dense_config, args.embedding_cache
    )
    dense_index = DenseIndex(corpus, document_embeddings)

    candidate_pool = hybrid_config["candidate_pool"]
    bm25_results = bm25_index.search(args.question, top_k=candidate_pool)
    query_embedding = dense_encoder.encode(
        [dense_config["query_prefix"] + args.question], batch_size=1
    )
    dense_results = dense_index.search(query_embedding, top_k=candidate_pool)
    language = detect_query_language(args.question)
    weights = hybrid_config["weights_by_language"].get(
        language, hybrid_config["weights_by_language"]["default"]
    )
    fused_results = reciprocal_rank_fusion(
        {"bm25": bm25_results, "dense": dense_results},
        rrf_k=hybrid_config["rrf_k"],
        weights=weights,
        top_k=reranker_config["candidate_pool"],
    )

    reranker = CrossEncoderScorer(
        reranker_config["model"],
        reranker_config["revision"],
        reranker_config["device"],
        args.model_cache,
        reranker_config["max_length"],
    )
    reranked = rerank(
        args.question,
        [
            RerankCandidate(result.chunk_id, rank=result.rank, score=result.score)
            for result in fused_results
        ],
        chunks_by_id,
        reranker,
        batch_size=reranker_config["batch_size"],
        top_k=reranker_config["candidate_pool"],
    )

    context_config = generation_config["context"]
    context_budget = (
        context_config["max_input_tokens"] - context_config["reserved_prompt_tokens"]
    )
    bundle = build_context(
        args.question,
        [result.chunk_id for result in reranked],
        chunks_by_id,
        token_budget=context_budget,
        max_sources=context_config["max_sources"],
    )
    user_prompt = build_user_prompt(bundle)
    if args.preview_context:
        print(SYSTEM_PROMPT)
        print("\n--- USER ---\n")
        print(user_prompt)
        print(
            f"\nSources: {len(bundle.sources)}; estimated context tokens: "
            f"{bundle.estimated_tokens}",
            file=sys.stderr,
        )
        return

    client = OllamaClient(generation_config)
    response = client.chat(SYSTEM_PROMPT, user_prompt, ANSWER_SCHEMA)
    answer = validate_answer(
        response.content, {source.source_id for source in bundle.sources}
    )
    print(render_answer(answer, bundle.sources))
    elapsed = time.perf_counter() - started
    print(
        f"\n[model={response.model}; prompt_tokens={response.prompt_tokens}; "
        f"completion_tokens={response.completion_tokens}; elapsed={elapsed:.2f}s]",
        file=sys.stderr,
    )


if __name__ == "__main__":
    main()
