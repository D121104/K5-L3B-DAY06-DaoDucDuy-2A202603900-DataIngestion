"""Store and search vectors using PostgreSQL + pgvector."""
import json
import math
import numbers
from typing import Any

import psycopg

from .models import Chunk


class PgVectorStore:
    def __init__(self, database_url: str, dimensions: int) -> None:
        if isinstance(dimensions, bool) or not isinstance(dimensions, int) or dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.database_url = database_url
        self.dimensions = dimensions

    def _vector(self, values: list[float], name: str) -> str:
        if isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be a sequence of {self.dimensions} finite numbers")
        try:
            vector = list(values)
        except TypeError as exc:
            raise ValueError(f"{name} must be a sequence of {self.dimensions} finite numbers") from exc
        if len(vector) != self.dimensions:
            raise ValueError(f"{name} must have exactly {self.dimensions} dimensions")
        if any(isinstance(value, bool) or not isinstance(value, numbers.Real)
               or not math.isfinite(value) for value in vector):
            raise ValueError(f"{name} must contain only finite numeric values")
        return "[" + ",".join(repr(float(value)) for value in vector) + "]"

    def create_schema(self) -> None:
        """Ensure pgvector and the strategy-scoped chunk table exist."""
        with psycopg.connect(self.database_url) as connection:
            connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
            connection.execute(
                f"""CREATE TABLE IF NOT EXISTS chunks (
                    strategy TEXT NOT NULL,
                    chunk_id TEXT NOT NULL,
                    document_id TEXT NOT NULL,
                    source TEXT NOT NULL,
                    page INTEGER NOT NULL,
                    text TEXT NOT NULL,
                    metadata JSONB NOT NULL DEFAULT '{{}}'::jsonb,
                    embedding vector({self.dimensions}) NOT NULL,
                    PRIMARY KEY (strategy, chunk_id)
                )"""
            )

    def upsert(self, strategy: str, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        """Replace all represented documents for a strategy atomically."""
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")
        prepared = []
        seen_ids: set[str] = set()
        for chunk, vector in zip(chunks, vectors):
            if not isinstance(chunk.metadata, dict):
                raise ValueError("chunk metadata must be a dictionary")
            if chunk.chunk_id in seen_ids:
                raise ValueError(f"duplicate chunk_id in upsert: {chunk.chunk_id}")
            seen_ids.add(chunk.chunk_id)
            prepared.append((chunk, self._vector(vector, "vector")))

        if not prepared:
            return
        documents = list(dict.fromkeys(chunk.document_id for chunk, _ in prepared))
        with psycopg.connect(self.database_url) as connection:
            connection.execute(
                "DELETE FROM chunks WHERE strategy = %s AND document_id = ANY(%s)",
                (strategy, documents),
            )
            for chunk, vector in prepared:
                connection.execute(
                    """INSERT INTO chunks
                       (strategy, chunk_id, document_id, source, page, text, metadata, embedding)
                       VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s::vector)""",
                    (strategy, chunk.chunk_id, chunk.document_id, chunk.source, chunk.page,
                     chunk.text, json.dumps(chunk.metadata, allow_nan=False), vector),
                )

    def search(self, strategy: str, query_vector: list[float], top_k: int = 3,
               filters: dict[str, Any] | None = None) -> list[dict[str, Any]]:
        """Search by cosine similarity, returning source traceability."""
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ValueError("top_k must be an integer")
        if top_k <= 0:
            return []
        vector = self._vector(query_vector, "query_vector")
        if filters is None:
            filters = {}
        if not isinstance(filters, dict):
            raise ValueError("filters must be a dictionary")
        allowed = {"section", "chunk_index", "chunk_strategy", "extraction_status",
                   "extraction_error", "title", "author", "year", "category", "source", "page"}
        unknown = set(filters) - allowed
        if unknown:
            raise ValueError(f"unsupported metadata filter key(s): {', '.join(sorted(map(str, unknown)))}")

        conditions = ["strategy = %s"]
        params: list[Any] = [vector, strategy]
        for key, value in filters.items():
            if key == "source":
                conditions.append("source = %s")
            elif key == "page":
                conditions.append("page = %s")
            else:
                conditions.append("metadata -> %s = %s::jsonb")
                params.extend((key, json.dumps(value, allow_nan=False)))
                continue
            params.append(value)
        params.extend((vector, top_k))
        sql = f"""SELECT chunk_id, text, 1 - (embedding <=> %s::vector) AS score,
                         metadata, document_id, source, page
                  FROM chunks WHERE {' AND '.join(conditions)}
                  ORDER BY embedding <=> %s::vector LIMIT %s"""
        with psycopg.connect(self.database_url) as connection:
            rows = connection.execute(sql, params).fetchall()
        return [
            {"chunk_id": row[0], "text": row[1], "score": row[2], "metadata": row[3],
             "document_id": row[4], "source": row[5], "page": row[6]}
            for row in rows
        ]
