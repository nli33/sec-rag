"""Offload dense chunk embedding to a GPU cluster node over SSH/SCP at ingest time.

Local CPU dense embedding (bge-large-en-v1.5, EMBED_THREADS-capped) runs at ~2.4s/chunk
on the primary dev machine (see PERFORMANCE.md) — a multi-document ingest can take well
over an hour. The same model on a GPU node via sentence-transformers runs at ~15-20ms/chunk
(measured), so we ship chunk texts up, embed there, and ship vectors back. Sparse (BM25)
embedding stays local always — it's ~4ms/chunk, already fast, and gains nothing from GPU.

Query-time embedding always stays local (see secrag/retrieve.py) — this module is only
used for the one-off/batch ingest path, not the interactive query path.

Enable by setting REMOTE_EMBED_HOST in config (empty/unset disables this entirely and
callers should fall back to local embedding).
"""
import json
import subprocess
import tempfile
import uuid
from pathlib import Path

from secrag.config import REMOTE_EMBED_HOST, REMOTE_EMBED_SCRATCH_DIR

_REMOTE_SCRATCH_DIR = REMOTE_EMBED_SCRATCH_DIR
_REMOTE_WORKER_SCRIPT = f"{_REMOTE_SCRATCH_DIR}/secrag_remote_embed_worker.py"
_REMOTE_VENV_ACTIVATE = f"{_REMOTE_SCRATCH_DIR}/secrag_gpu_venv/bin/activate"


class RemoteEmbedError(Exception):
    pass


def embed_dense_remote(texts: list[str], host: str = REMOTE_EMBED_HOST) -> list[list[float]]:
    """Dense-embed `texts` on `host`'s GPU via SSH/SCP. Raises RemoteEmbedError on any failure."""
    if not host:
        raise RemoteEmbedError("no remote embed host configured (set REMOTE_EMBED_HOST)")

    job_id = uuid.uuid4().hex
    remote_in = f"{_REMOTE_SCRATCH_DIR}/embed_in_{job_id}.json"
    remote_out = f"{_REMOTE_SCRATCH_DIR}/embed_out_{job_id}.json"

    with tempfile.TemporaryDirectory() as tmpdir:
        local_in = Path(tmpdir) / "in.json"
        local_out = Path(tmpdir) / "out.json"
        local_in.write_text(json.dumps(texts))

        try:
            subprocess.run(
                ["scp", "-o", "ConnectTimeout=15", str(local_in), f"{host}:{remote_in}"],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                [
                    "ssh", "-o", "ConnectTimeout=15", host,
                    f"source {_REMOTE_VENV_ACTIVATE} && python3 {_REMOTE_WORKER_SCRIPT} {remote_in} {remote_out}",
                ],
                check=True, capture_output=True, text=True,
            )
            subprocess.run(
                ["scp", "-o", "ConnectTimeout=15", f"{host}:{remote_out}", str(local_out)],
                check=True, capture_output=True, text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RemoteEmbedError(f"remote embedding failed: {e.stderr}") from e
        finally:
            subprocess.run(
                ["ssh", "-o", "ConnectTimeout=15", host, f"rm -f {remote_in} {remote_out}"],
                capture_output=True, text=True,
            )

        return json.loads(local_out.read_text())
