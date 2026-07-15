#!/usr/bin/env bash
# Download the Qwen3-TTS and Qwen3-ASR (0.6B) model weights from Hugging Face
# into ./models. Model weights are large and git-ignored, so this is kept out of
# the startup update script; run it once (weights then persist in the VM snapshot).
set -euo pipefail
cd "$(dirname "$0")/.."

HF="${HF_CLI:-.venv-tts/bin/huggingface-cli}"

"$HF" download Qwen/Qwen3-TTS-Tokenizer-12Hz        --local-dir models/Qwen3-TTS-Tokenizer-12Hz
"$HF" download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice
"$HF" download Qwen/Qwen3-ASR-0.6B                  --local-dir models/Qwen3-ASR-0.6B

# Tencent HunyuanOCR-1.5 (skip the archived v1.0 and the vLLM-only dflash draft)
"$HF" download tencent/HunyuanOCR --local-dir models/HunyuanOCR --exclude "v1.0/*" "dflash/*"

echo "All models downloaded into ./models"
