from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .retrieval import searchable_text

FloatMatrix = NDArray[np.float32]


class TextEncoder(Protocol):
    dimension: int

    def encode(self, texts: list[str], *, batch_size: int) -> FloatMatrix: ...


class SentenceTransformerEncoder:
    def __init__(
        self,
        model_name: str,
        revision: str,
        device: str,
        cache_folder: Path,
        normalize_embeddings: bool = True,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        resolved_device = None if device == "auto" else device
        self.model = SentenceTransformer(
            model_name,
            revision=revision,
            device=resolved_device,
            cache_folder=str(cache_folder),
        )
        self.dimension = int(self.model.get_embedding_dimension())
        self.device = str(self.model.device)
        self.normalize_embeddings = normalize_embeddings

    def encode(self, texts: list[str], *, batch_size: int) -> FloatMatrix:
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=self.normalize_embeddings,
            convert_to_numpy=True,
            show_progress_bar=len(texts) > 1,
        )
        return np.asarray(embeddings, dtype=np.float32)


def chunk_ids_sha256(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        digest.update(chunk["chunk_id"].encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def document_inputs(chunks: list[dict[str, Any]], prefix: str) -> list[str]:
    return [prefix + searchable_text(chunk) for chunk in chunks]


def expected_manifest(
    chunks: list[dict[str, Any]], config: dict[str, Any], dimension: int
) -> dict[str, Any]:
    source_commits = sorted({chunk["source_commit"] for chunk in chunks})
    return {
        "model": config["model"],
        "revision": config["revision"],
        "dimension": dimension,
        "normalized": config["normalize_embeddings"],
        "query_prefix": config["query_prefix"],
        "document_prefix": config["document_prefix"],
        "text_template": config["text_template"],
        "corpus_source_commits": source_commits,
        "chunks": len(chunks),
        "chunk_ids_sha256": chunk_ids_sha256(chunks),
        "dtype": "float32",
    }


def load_or_encode_documents(
    chunks: list[dict[str, Any]],
    encoder: TextEncoder,
    config: dict[str, Any],
    cache_dir: Path,
) -> tuple[FloatMatrix, dict[str, Any], bool]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = config["model"].replace("/", "--")
    embeddings_path = cache_dir / f"{safe_model_name}.npy"
    manifest_path = cache_dir / f"{safe_model_name}.manifest.json"
    expected = expected_manifest(chunks, config, encoder.dimension)

    if embeddings_path.exists() and manifest_path.exists():
        actual = json.loads(manifest_path.read_text(encoding="utf-8"))
        if actual == expected:
            embeddings = np.load(embeddings_path)
            if embeddings.shape == (len(chunks), encoder.dimension):
                return embeddings, actual, True

    embeddings = encoder.encode(
        document_inputs(chunks, config["document_prefix"]),
        batch_size=config["batch_size"],
    )
    if embeddings.shape != (len(chunks), encoder.dimension):
        raise ValueError(
            f"Unexpected embedding shape {embeddings.shape}; "
            f"expected {(len(chunks), encoder.dimension)}"
        )
    temporary_path = embeddings_path.with_suffix(".tmp.npy")
    np.save(temporary_path, embeddings)
    temporary_path.replace(embeddings_path)
    manifest_path.write_text(
        json.dumps(expected, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return embeddings, expected, False


@dataclass(frozen=True)
class DenseSearchResult:
    chunk_id: str
    score: float
    rank: int


class DenseIndex:
    def __init__(self, chunks: list[dict[str, Any]], embeddings: FloatMatrix) -> None:
        if embeddings.ndim != 2 or embeddings.shape[0] != len(chunks):
            raise ValueError("Embedding rows must match chunks")
        self.chunks = chunks
        self.embeddings = embeddings

    def search(self, query_embedding: FloatMatrix, top_k: int = 10) -> list[DenseSearchResult]:
        query = np.asarray(query_embedding, dtype=np.float32).reshape(-1)
        if query.shape[0] != self.embeddings.shape[1]:
            raise ValueError("Query embedding dimension does not match the index")
        if top_k <= 0:
            return []

        scores = np.asarray(self.embeddings @ query)
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [
            DenseSearchResult(
                chunk_id=self.chunks[int(index)]["chunk_id"],
                score=round(float(scores[index]), 8),
                rank=rank,
            )
            for rank, index in enumerate(indices, start=1)
        ]
