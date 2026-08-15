"""DuckDB documentation ingestion and evaluation utilities."""

from .analysis import analyze_corpus
from .corpus import build_corpus, chunk_markdown, select_source_paths
from .evaluation import build_qrels, validate_evaluation

__all__ = [
    "analyze_corpus",
    "build_corpus",
    "build_qrels",
    "chunk_markdown",
    "select_source_paths",
    "validate_evaluation",
]
