"""One-time build step: produce an INT8-quantized copy of the reranker model.

Same architecture/weights as secrag.retrieve.RERANK_MODEL_NAME, just lower numeric
precision (dynamic weight quantization, no calibration data needed) — not a different,
weaker model. See PERFORMANCE.md's "INT8 quantization of the same reranker model" section
for why this (and not a smaller architecture like MiniLM) is the safe way to speed up
reranking: ~2.3x faster with no confirmed accuracy regression on a 25-question eval sample.

Requires the `onnx` package (dev extra: `pip install -e .[dev]`) — not a runtime dependency,
only needed to run this script.

Usage:
    python scripts/quantize_reranker.py
"""
import shutil
from pathlib import Path

from fastembed.rerank.cross_encoder import TextCrossEncoder
from onnxruntime.quantization import QuantType, quantize_dynamic

from secrag.retrieve import RERANK_MODEL_NAME

OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "models" / "bge-reranker-base-int8"


def main():
    print(f"Downloading/locating {RERANK_MODEL_NAME}...")
    model = TextCrossEncoder(model_name=RERANK_MODEL_NAME)
    model_dir = model.model._model_dir
    src_onnx = model_dir / "onnx" / "model.onnx"
    if not src_onnx.exists():
        raise FileNotFoundError(f"expected fp32 ONNX model at {src_onnx}, not found")

    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)
    (OUTPUT_DIR / "onnx").mkdir(parents=True)

    print(f"Quantizing {src_onnx} -> INT8...")
    quantize_dynamic(model_input=str(src_onnx), model_output=str(OUTPUT_DIR / "onnx" / "model.onnx"), weight_type=QuantType.QInt8)

    for fname in ["config.json", "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json"]:
        src = model_dir / fname
        if src.exists():
            shutil.copy(src, OUTPUT_DIR / fname)

    orig_size = src_onnx.stat().st_size / 1e6
    quant_size = (OUTPUT_DIR / "onnx" / "model.onnx").stat().st_size / 1e6
    print(f"Done: {orig_size:.0f}MB -> {quant_size:.0f}MB, written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
