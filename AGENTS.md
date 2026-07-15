# model

Hugging Face 模型应用：本地部署并验证多个开源模型——
- Qwen3-TTS（语音合成）+ Qwen3-ASR（语音识别）：「文字 → 语音 → 文字」回环验证。
- 腾讯 HunyuanOCR-1.5（OCR 视觉语言模型）：「已知文字 → 渲染图片 → OCR → 对比」验证。

## Cursor Cloud specific instructions

Durable, non-obvious context for future cloud agents.

### 环境（关键约束）

- **Python 3.12**（系统 `/usr/bin/python3`，无 `python` 别名）。系统包 `python3.12-venv` 与 `sox` 已装（在 VM 快照中）。
- **无 GPU**：所有模型以 **CPU + `float32` + `attn_implementation="eager"`** 运行，禁用 flash-attn（未安装，属预期）。约 15GB 内存，0.6B 模型可跑。
- **多个互斥的虚拟环境（重要）**：各模型包固定了不同且互斥的 `transformers` 版本，必须分环境：
  - `.venv-tts` —— TTS（`qwen-tts`，transformers 4.57.3），`app/tts_synthesize.py`
  - `.venv-asr` —— ASR（`qwen-asr`，transformers 4.57.6），`app/asr_transcribe.py`
  - `.venv-ocr` —— OCR 客户端与测试图生成（仅需 `pillow`），`app/ocr_client.py` / `app/run_ocr_demo.py`
  - `.venv-convert` —— 仅用于把 HF 权重转 GGUF（llama.cpp 转换脚本依赖，见 `requirements-convert.txt`；不在 update 脚本内，按需 `pip install`）
  - TTS/ASR 回环脚本 `app/run_roundtrip.py` 用任一 venv 启动，它通过 subprocess 分别调用两个 venv，用磁盘 wav 传数据。
- torch 必须是 **CPU 版**（`--index-url https://download.pytorch.org/whl/cpu`）；update 脚本先装 CPU torch 再装各 requirements，避免拉取 CUDA 版。

### OCR（HunyuanOCR）走 llama.cpp CPU 路径（重要）

- **transformers 的 `hunyuan_vl` 视觉前向在 CPU 上产出乱码**（纯文本正常、权重完好，属该版本 CPU 数值问题；模型卡也要求 GPU）。因此 OCR 走模型卡指定的 **llama.cpp + GGUF** CPU 路径。
- 一次性构建/转换（不在 update 脚本，产物靠快照持久化）：
  - `bash scripts/build_llama_cpp.sh` → 构建到 `vendor/llama.cpp`（gitignored）。
  - `bash scripts/convert_ocr_gguf.sh` → 生成 `models/HunyuanOCR/{hyocr,mmproj-hyocr}-f16.gguf`。
- **构建 gotcha**：默认 `/usr/bin/c++` 是 clang，会去找不存在的 gcc-14 工具链导致失败；必须用 `gcc-13/g++-13`（构建脚本已指定）。系统需 `build-essential libstdc++-13-dev cmake`（在快照中）。
- **转换 gotcha**：必须 `PYTHONPATH=vendor/llama.cpp/gguf-py`，用仓库内 gguf-py（含新的 `HUNYUANVL` 投影类型），PyPI 版 `gguf` 可能没有。
- **推理 gotcha**：模型 `generation_config.json` 的 `eos_token_id` 是错的（120020）；真正的对话结束符是 `<｜hy_Assistant｜>`（**120007**）。不覆盖会一直循环不停。`scripts/serve_ocr.sh` 已用 `--override-kv tokenizer.ggml.eos_token_id=int:120007` 修正。

### 模型权重（不在 update 脚本里下载）

- 权重较大（TTS ~2.4G、tokenizer ~0.65G、ASR ~1.8G），放在 git 忽略的 `models/` 下，靠 VM 快照持久化。
- 若 `models/` 缺失（例如全新且无快照的 VM），运行 `bash scripts/download_models.sh` 重新下载。**不要**把模型下载塞进 update 脚本（体积大、易失败）。
- ASR 用的是**非 `-hf`** 的 `Qwen/Qwen3-ASR-0.6B`（与 `qwen-asr` 包配套）。`-hf` 变体是给原生 transformers 用的，但本环境的 transformers 4.57.x 尚无原生 `Qwen3ASRForConditionalGeneration`，故不用它。
- OCR 用 `tencent/HunyuanOCR`（排除 `v1.0/` 与仅 vLLM 用的 `dflash/`）。

### 运行 / 验证

- TTS→STT 回环：`.venv-tts/bin/python app/run_roundtrip.py`（输出各条与平均 CER，平均 ≤15% 退出码 0；CPU 单条合成约 10–14s）。
- OCR：先起服务 `bash scripts/serve_ocr.sh`（后台，加载约 25s），再跑 `.venv-ocr/bin/python app/run_ocr_demo.py`（生成已知文字图 → OCR → CER）。单图首次约 20s；`llama-server` 对相同输入有缓存。
- 标准命令见 `README.md`，无需在此重复。
- 官方 gradio Web Demo（`qwen-tts-demo` / `qwen-asr-demo`）与 transformers OCR 路径均默认走 CUDA，本无 GPU 环境不适用；优先用上面脚本验证。
