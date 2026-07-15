#!/usr/bin/env bash
# Serve Tencent HunyuanOCR via llama.cpp (OpenAI-compatible API) on CPU.
# Requires the GGUF files (scripts/convert_ocr_gguf.sh) and vendor/llama.cpp.
#
# NOTE: the model's generation_config.json has a wrong eos_token_id (120020); the
# real end-of-turn token is <｜hy_Assistant｜> (120007). Without the override the
# model never stops and loops. Keep the override below.
set -euo pipefail
cd "$(dirname "$0")/.."

LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-vendor/llama.cpp}"

exec "$LLAMA_CPP_DIR/build/bin/llama-server" \
  -m models/HunyuanOCR/hyocr-f16.gguf \
  --mmproj models/HunyuanOCR/mmproj-hyocr-f16.gguf \
  --host 127.0.0.1 --port "${OCR_PORT:-8080}" \
  -t "${OCR_THREADS:-4}" --ctx-size "${OCR_CTX:-8192}" \
  --override-kv tokenizer.ggml.eos_token_id=int:120007
