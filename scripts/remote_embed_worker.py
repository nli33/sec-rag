"""Runs on the GPU cluster (not part of the local secrag package). Reads a JSON
list of chunk texts from argv[1], dense-embeds them on GPU, writes a JSON list
of vectors to argv[2]. Kept dependency-light (sentence-transformers only) since
this venv is separate from the local project's.
"""
import json
import sys

from sentence_transformers import SentenceTransformer

DENSE_MODEL_NAME = "BAAI/bge-large-en-v1.5"


def main():
    in_path, out_path = sys.argv[1], sys.argv[2]
    with open(in_path) as f:
        texts = json.load(f)

    model = SentenceTransformer(DENSE_MODEL_NAME, device="cuda")
    vecs = model.encode(texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True)

    with open(out_path, "w") as f:
        json.dump(vecs.tolist(), f)


if __name__ == "__main__":
    main()
