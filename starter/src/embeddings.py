"""Bedrock Cohere Embed v4 implementation of the embedding contract."""
import json
import math
import os
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return one vector per input text, in the same order."""
        ...

    def embed_query(self, question: str) -> list[float]:
        """Return a query vector compatible with the stored document vectors."""
        ...


class MyEmbedder:
    """Generate embeddings with the AWS Bedrock Runtime API."""

    dimensions = 1536
    batch_size = 96

    def __init__(self, model_name: str = "global.cohere.embed-v4:0") -> None:
        self.model_name = model_name
        try:
            import boto3
        except ImportError as exc:
            raise RuntimeError(
                "boto3 is required for Bedrock embeddings; install it with 'pip install boto3'"
            ) from exc

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION") or "ap-southeast-2"
        try:
            self._client = boto3.client("bedrock-runtime", region_name=region)
        except Exception as exc:
            detail = str(exc)
            if "Using the login credential provider requires an additional dependency" in detail:
                detail = (
                    "Using the AWS login credential provider requires an additional dependency. "
                    "Install botocore's CRT extra with 'pip install \"botocore[crt]\"' "
                    "or install the starter requirements, then try again."
                )
            elif detail:
                detail = f": {detail}"
            raise RuntimeError(
                f"Could not initialize Bedrock Runtime client in region {region!r} "
                f"({type(exc).__name__}{detail})"
            ) from exc

    def _embed_batch(self, texts: list[str], input_type: str) -> list[list[float]]:
        try:
            response = self._client.invoke_model(
                modelId=self.model_name,
                body=json.dumps({
                    "texts": texts,
                    "input_type": input_type,
                    "embedding_types": ["float"],
                }),
                contentType="application/json",
                accept="application/json",
            )
        except Exception as exc:
            raise RuntimeError(
                f"Bedrock embedding request failed for model {self.model_name!r} "
                f"({type(exc).__name__})"
            ) from exc

        try:
            body = response["body"]
            if hasattr(body, "read"):
                body = body.read()
            if isinstance(body, bytes):
                body = body.decode("utf-8")
            payload = json.loads(body) if isinstance(body, str) else body
            embeddings = payload["embeddings"]
            if isinstance(embeddings, dict):
                embeddings = embeddings["float"]
            if not isinstance(embeddings, list) or len(embeddings) != len(texts):
                raise ValueError("embedding count does not match input count")

            vectors: list[list[float]] = []
            for vector in embeddings:
                if not isinstance(vector, (list, tuple)) or len(vector) != self.dimensions:
                    raise ValueError(f"each embedding must contain {self.dimensions} values")
                if any(isinstance(value, bool) or not isinstance(value, (int, float)) for value in vector):
                    raise ValueError("embedding contains a non-numeric value")
                converted = [float(value) for value in vector]
                if not all(math.isfinite(value) for value in converted):
                    raise ValueError("embedding contains a non-finite value")
                vectors.append(converted)
            return vectors
        except Exception as exc:
            raise RuntimeError(
                f"Invalid Bedrock embedding response for model {self.model_name!r} "
                f"({type(exc).__name__}: {exc})"
            ) from exc

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(self._embed_batch(texts[start:start + self.batch_size], "search_document"))
        return vectors

    def embed_query(self, question: str) -> list[float]:
        vectors = self._embed_batch([question], "search_query")
        return vectors[0]
