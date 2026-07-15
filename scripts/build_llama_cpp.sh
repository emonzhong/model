#!/usr/bin/env bash
# Build llama.cpp (CPU) with multimodal (mtmd) support into vendor/llama.cpp.
# Used for the HunyuanOCR CPU inference path (the transformers path needs a GPU).
#
# System prerequisites (installed in the VM snapshot, not here):
#   build-essential libstdc++-13-dev cmake git
# IMPORTANT: /usr/bin/c++ (the default) is clang on this image and is mis-configured
# (it looks for a non-existent gcc-14 toolchain), so we force gcc-13/g++-13.
set -euo pipefail
cd "$(dirname "$0")/.."

mkdir -p vendor
if [ ! -d vendor/llama.cpp/.git ]; then
  git clone --depth 1 https://github.com/ggml-org/llama.cpp.git vendor/llama.cpp
fi
cd vendor/llama.cpp

cmake -B build \
  -DCMAKE_C_COMPILER=/usr/bin/gcc-13 \
  -DCMAKE_CXX_COMPILER=/usr/bin/g++-13 \
  -DLLAMA_BUILD_EXAMPLES=ON \
  -DLLAMA_BUILD_TESTS=OFF \
  -DGGML_NATIVE=ON
cmake --build build --config Release -j"$(nproc)"

echo "Built: vendor/llama.cpp/build/bin/{llama-server,llama-mtmd-cli}"
