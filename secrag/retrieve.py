"""M2: hybrid (dense + sparse, RRF-fused) retrieval, then cross-encoder rerank."""
import os
from dataclasses import dataclass
from typing import Optional

from fastembed.rerank.cross_encoder import TextCrossEncoder
from qdrant_client import models

from secrag.config import INT8_RERANKER_PATH, USE_INT8_RERANKER, USE_QUERY_DECOMPOSITION
from secrag.decompose import decompose
from secrag.index import FASTEMBED_CACHE_DIR, COLLECTION_NAME, get_dense_model, get_sparse_model, get_client

RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
PREFETCH_LIMIT = 20
RERANK_TOP_K = 12
# Per-sub-query prefetch when query decomposition is on — kept smaller than PREFETCH_LIMIT
# so a multi-sub-query merged candidate pool stays close to today's single-query size
# rather than multiplying it (e.g. 3 sub-queries x 10 ~= 20-30, not 3x60).
SUB_QUERY_PREFETCH_LIMIT = 10
# Unlike secrag.index.EMBED_THREADS (capped low for long-running ingest jobs), reranking is
# a short per-query burst, so use all available cores rather than throttling it.
RERANK_THREADS = os.cpu_count() or 1

_rerank_model = None


def _get_rerank_model() -> TextCrossEncoder:
    global _rerank_model
    if _rerank_model is None:
        if USE_INT8_RERANKER:
            if not os.path.isdir(INT8_RERANKER_PATH):
                raise FileNotFoundError(
                    f"USE_INT8_RERANKER is set but {INT8_RERANKER_PATH} doesn't exist — "
                    "run `python scripts/quantize_reranker.py` first"
                )
            _rerank_model = TextCrossEncoder(
                model_name=RERANK_MODEL_NAME, threads=RERANK_THREADS,
                specific_model_path=INT8_RERANKER_PATH, cache_dir=FASTEMBED_CACHE_DIR,
            )
        else:
            _rerank_model = TextCrossEncoder(
                model_name=RERANK_MODEL_NAME, threads=RERANK_THREADS, cache_dir=FASTEMBED_CACHE_DIR
            )
    return _rerank_model


@dataclass
class RetrievedChunk:
    text: str
    provenance: dict
    rerank_score: float


def hybrid_search(
    query: str,
    ticker: Optional[str] = None,
    prefetch_limit: int = PREFETCH_LIMIT,
    collection_name: str = COLLECTION_NAME,
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
        collection_name=collection_name,
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


def retrieve(
    query: str,
    ticker: Optional[str] = None,
    top_k: int = RERANK_TOP_K,
    collection_name: str = COLLECTION_NAME,
) -> list[RetrievedChunk]:
    if not USE_QUERY_DECOMPOSITION:
        candidates = hybrid_search(query, ticker=ticker, collection_name=collection_name)
        return rerank(query, candidates, top_k=top_k)

    sub_queries = decompose(query)
    if len(sub_queries) == 1:
        candidates = hybrid_search(sub_queries[0], ticker=ticker, collection_name=collection_name)
        return rerank(query, candidates, top_k=top_k)

    seen = set()
    merged = []
    for sub_query in sub_queries:
        for payload in hybrid_search(
            sub_query, ticker=ticker, collection_name=collection_name, prefetch_limit=SUB_QUERY_PREFETCH_LIMIT
        ):
            key = (payload.get("item"), payload.get("chunk_index"))
            if key not in seen:
                seen.add(key)
                merged.append(payload)
    return rerank(query, merged, top_k=top_k)
