"""DuckDB documentation ingestion and evaluation utilities."""

from .analysis import analyze_corpus
from .corpus import build_corpus, chunk_markdown, select_source_paths
from .evaluation import build_qrels, validate_evaluation
from .metrics import aggregate_metrics, ranking_metrics
from .retrieval import BM25Index, tokenize

__all__ = [
    "BM25Index",
    "aggregate_metrics",
    "analyze_corpus",
    "build_corpus",
    "build_qrels",
    "chunk_markdown",
    "ranking_metrics",
    "select_source_paths",
    "tokenize",
    "validate_evaluation",
]
