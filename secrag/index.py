"""M2: embed chunks (dense + sparse) and upsert into Qdrant.

Uses the same bge-large-en-v1.5 (dense) + Qdrant/bm25 (sparse) pair validated
in M0's smoke test, so retrieval-time embeddings match index-time embeddings.
"""
import hashlib
from dataclasses import asdict

from fastembed import SparseTextEmbedding, TextEmbedding
from qdrant_client import QdrantClient, models

from secrag.chunk import Chunk
from secrag.config import QDRANT_URL

COLLECTION_NAME = "secrag_chunks"
DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5"
SPARSE_MODEL_NAME = "Qdrant/bm25"
DENSE_DIM = 1024

_dense_model = None
_sparse_model = None


def get_dense_model() -> TextEmbedding:
    global _dense_model
    if _dense_model is None:
        _dense_model = TextEmbedding(model_name=DENSE_MODEL_NAME)
    return _dense_model


def get_sparse_model() -> SparseTextEmbedding:
    global _sparse_model
    if _sparse_model is None:
        _sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL_NAME)
    return _sparse_model


def get_client() -> QdrantClient:
    return QdrantClient(url=QDRANT_URL)


def ensure_collection(client: QdrantClient) -> None:
    if client.collection_exists(COLLECTION_NAME):
        return
    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={"dense": models.VectorParams(size=DENSE_DIM, distance=models.Distance.COSINE)},
        sparse_vectors_config={"sparse": models.SparseVectorParams()},
    )


def _point_id(chunk: Chunk) -> str:
    key = f"{chunk.provenance.accession}:{chunk.provenance.item}:{chunk.chunk_index}"
    return hashlib.md5(key.encode()).hexdigest()


def index_chunks(chunks: list[Chunk]) -> int:
    """Embed and upsert chunks into Qdrant. Returns the number of points written."""
    if not chunks:
        return 0

    client = get_client()
    ensure_collection(client)

    texts = [c.text for c in chunks]
    dense_vecs = list(get_dense_model().embed(texts))
    sparse_vecs = list(get_sparse_model().embed(texts))

    points = []
    for chunk, dense_vec, sparse_vec in zip(chunks, dense_vecs, sparse_vecs):
        payload = {**asdict(chunk.provenance), "text": chunk.text, "chunk_index": chunk.chunk_index}
        points.append(
            models.PointStruct(
                id=_point_id(chunk),
                vector={
                    "dense": dense_vec.tolist(),
                    "sparse": models.SparseVector(
                        indices=sparse_vec.indices.tolist(), values=sparse_vec.values.tolist()
                    ),
                },
                payload=payload,
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)
