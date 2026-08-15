"""DuckDB documentation ingestion utilities."""

from .corpus import build_corpus, chunk_markdown, select_source_paths

__all__ = ["build_corpus", "chunk_markdown", "select_source_paths"]
