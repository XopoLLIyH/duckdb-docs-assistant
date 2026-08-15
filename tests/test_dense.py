from __future__ import annotations

import json

import numpy as np

from duckdb_docs_assistant.dense import (
    DenseIndex,
    chunk_ids_sha256,
    load_or_encode_documents,
)


class FakeEncoder:
    dimension = 2

    def __init__(self) -> None:
        self.calls = 0

    def encode(self, texts: list[str], *, batch_size: int) -> np.ndarray:
        self.calls += 1
        return np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)


def _chunks() -> list[dict]:
    return [
        {
            "chunk_id": "first",
            "title": "First",
            "heading_path": ["One"],
            "text": "Alpha",
            "source_commit": "source-sha",
        },
        {
            "chunk_id": "second",
            "title": "Second",
            "heading_path": ["Two"],
            "text": "Beta",
            "source_commit": "source-sha",
        },
    ]


def _config() -> dict:
    return {
        "model": "test/model",
        "revision": "model-sha",
        "query_prefix": "query: ",
        "document_prefix": "passage: ",
        "text_template": "title_heading_text_v1",
        "normalize_embeddings": True,
        "batch_size": 2,
    }


def test_dense_index_ranks_largest_dot_product_first() -> None:
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    index = DenseIndex(_chunks(), embeddings)

    results = index.search(np.asarray([[0.1, 0.9]], dtype=np.float32), top_k=2)

    assert [result.chunk_id for result in results] == ["second", "first"]


def test_embedding_cache_is_bound_to_chunk_order(tmp_path) -> None:
    chunks = _chunks()
    encoder = FakeEncoder()

    embeddings, manifest, reused = load_or_encode_documents(
        chunks, encoder, _config(), tmp_path
    )

    assert embeddings.shape == (2, 2)
    assert reused is False
    assert manifest["chunk_ids_sha256"] == chunk_ids_sha256(chunks)
    assert encoder.calls == 1

    _, cached_manifest, reused = load_or_encode_documents(
        chunks, encoder, _config(), tmp_path
    )
    assert reused is True
    assert cached_manifest == manifest
    assert encoder.calls == 1

    manifest_path = tmp_path / "test--model.manifest.json"
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == manifest

    _, changed_manifest, reused = load_or_encode_documents(
        list(reversed(chunks)), encoder, _config(), tmp_path
    )
    assert reused is False
    assert changed_manifest["chunk_ids_sha256"] != manifest["chunk_ids_sha256"]
    assert encoder.calls == 2
