#!/usr/bin/env bash
# Convert the downloaded tencent/HunyuanOCR HF checkpoint to GGUF (base + mmproj)
# for llama.cpp. Requires vendor/llama.cpp (scripts/build_llama_cpp.sh) and the
# conversion venv (.venv-convert, see requirements-convert.txt).
#
# PYTHONPATH must point at the repo's gguf-py so the (very new) HUNYUANVL vision
# projector type is available — the PyPI `gguf` release may not have it yet.
set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-vendor/llama.cpp}"
PY="${CONVERT_PY:-.venv-convert/bin/python}"
MODEL_DIR="models/HunyuanOCR"

PYTHONPATH="$LLAMA_CPP_DIR/gguf-py" "$PY" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  --outfile "$MODEL_DIR/hyocr-f16.gguf" --outtype f16 "$MODEL_DIR"

PYTHONPATH="$LLAMA_CPP_DIR/gguf-py" "$PY" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" \
  --outfile "$MODEL_DIR/mmproj-hyocr-f16.gguf" --outtype f16 --mmproj "$MODEL_DIR"

echo "Wrote $MODEL_DIR/hyocr-f16.gguf and $MODEL_DIR/mmproj-hyocr-f16.gguf"
