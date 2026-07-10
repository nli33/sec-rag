"""M2: hybrid (dense + sparse, RRF-fused) retrieval, then cross-encoder rerank."""
from dataclasses import dataclass
from typing import Optional

from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models

from secrag.index import COLLECTION_NAME, get_dense_model, get_sparse_model, get_client

RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
PREFETCH_LIMIT = 50
RERANK_TOP_K = 8

_rerank_model = None


def _get_rerank_model() -> TextCrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        _rerank_model = TextCrossEncoder(model_name=RERANK_MODEL_NAME)
    return _rerank_model


@dataclass
class RetrievedChunk:
    text: str
    provenance: dict
    rerank_score: float


def hybrid_search(
    query: str, ticker: Optional[str] = None, prefetch_limit: int = PREFETCH_LIMIT
) -> list[dict]:
    """Dense+sparse hybrid retrieval via Qdrant native RRF fusion. Returns raw payloads."""
    client = get_client()
    dense_vec = list(get_dense_model().embed([query]))[0]
    sparse_vec = list(get_sparse_model().embed([query]))[0]

    query_filter = None
    if ticker:
        query_filter = models.Filter(
            must=[models.FieldCondition(key="ticker", match=models.MatchValue(value=ticker.upper()))]
        )

    result = client.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            models.Prefetch(
                query=dense_vec.tolist(), using="dense", limit=prefetch_limit, filter=query_filter
            ),
            models.Prefetch(
                query=models.SparseVector(indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()),
                using="sparse",
                limit=prefetch_limit,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=prefetch_limit,
    )
    return [point.payload for point in result.points]


def rerank(query: str, payloads: list[dict], top_k: int = RERANK_TOP_K) -> list[RetrievedChunk]:
    """Cross-encoder rerank; returns the top_k highest-scoring chunks."""
    if not payloads:
        return []
    texts = [p["text"] for p in payloads]
    scores = list(_get_rerank_model().rerank(query, texts))
    ranked = sorted(zip(payloads, scores), key=lambda pair: pair[1], reverse=True)
    return [
        RetrievedChunk(text=payload["text"], provenance=payload, rerank_score=float(score))
        for payload, score in ranked[:top_k]
    ]


def retrieve(query: str, ticker: Optional[str] = None, top_k: int = RERANK_TOP_K) -> list[RetrievedChunk]:
    candidates = hybrid_search(query, ticker=ticker)
    return rerank(query, candidates, top_k=top_k)
