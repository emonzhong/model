# model

Hugging Face 模型应用 —— 在本地部署并验证多个开源模型：
- **TTS/STT**（Qwen）：构建「文字 → 语音 → 文字」端到端回环场景。
- **OCR**（腾讯 HunyuanOCR）：构建「已知文字 → 渲染图片 → OCR → 对比」验证场景。

## 使用的模型（从 Hugging Face 下载到本地 `models/`）

| 能力 | 模型 | 许可 | 说明 |
| --- | --- | --- | --- |
| TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` + `Qwen/Qwen3-TTS-Tokenizer-12Hz` | Apache 2.0 | 9 种内置音色、10 种语言 |
| STT | `Qwen/Qwen3-ASR-0.6B` | Apache 2.0 | 多语言语音识别 + 语种识别 |
| OCR | `tencent/HunyuanOCR`（1.5，1B） | Tencent Hunyuan Community | 端到端 OCR 视觉语言模型 |

> 各模型包固定了互斥的 `transformers` 版本，因此分环境：`.venv-tts`（4.57.3）、`.venv-asr`（4.57.6）、
> `.venv-ocr`（OCR 客户端，仅 pillow）、`.venv-convert`（GGUF 转换）。当前环境**无 GPU**，均以 CPU 运行。
> OCR 的 transformers CPU 前向有问题，故走模型卡指定的 **llama.cpp + GGUF** CPU 路径。

## 快速开始

```bash
# 1) 安装依赖（两个隔离环境，CPU 版 torch）
python3 -m venv .venv-tts && python3 -m venv .venv-asr
.venv-tts/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv-tts/bin/pip install -r requirements-tts.txt
.venv-asr/bin/pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
.venv-asr/bin/pip install -r requirements-asr.txt

# 2) 下载模型到本地 ./models
bash scripts/download_models.sh

# 3) 运行端到端场景：TTS 合成 -> ASR 转写 -> 计算字错率(CER)
.venv-tts/bin/python app/run_roundtrip.py
```

## 单独使用

```bash
# TTS：文字 -> 语音
.venv-tts/bin/python app/tts_synthesize.py \
  --text "今天天气很好" --language Chinese --speaker Vivian --out outputs/demo.wav

# STT：语音 -> 文字
.venv-asr/bin/python app/asr_transcribe.py --audio outputs/demo.wav --language Chinese
```

## OCR（腾讯 HunyuanOCR，llama.cpp CPU 路径）

```bash
# 1) 构建 llama.cpp（含多模态）并把模型转成 GGUF（一次性；需 build-essential/cmake）
bash scripts/build_llama_cpp.sh
.venv-convert/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv-convert/bin/pip install -r requirements-convert.txt
bash scripts/convert_ocr_gguf.sh

# 2) 启动 OCR 服务（OpenAI 兼容），另开终端
bash scripts/serve_ocr.sh

# 3) 运行验证场景：已知文字 -> 渲染图片 -> OCR -> CER
.venv-ocr/bin/python app/run_ocr_demo.py
# 或对单张图片识别：
.venv-ocr/bin/python app/ocr_client.py --image outputs/ocr_test.png
```

## 目录结构

- `app/tts_synthesize.py` — Qwen3-TTS 语音合成（单条 / 批量）
- `app/asr_transcribe.py` — Qwen3-ASR 语音识别（单条 / 批量）
- `app/run_roundtrip.py` — TTS→STT 回环验证场景（跨两个 venv 编排，输出 CER）
- `app/make_test_image.py` — 把已知文字渲染成图片（OCR 的 ground truth）
- `app/ocr_client.py` — HunyuanOCR 客户端（调用 llama-server 的 OpenAI 接口）
- `app/run_ocr_demo.py` — OCR 验证场景（渲染 → 识别 → CER）
- `scripts/download_models.sh` — 下载全部模型到 `models/`
- `scripts/build_llama_cpp.sh` / `scripts/convert_ocr_gguf.sh` / `scripts/serve_ocr.sh` — OCR 构建/转换/服务
- `models/`、`outputs/`、`vendor/` — 权重、生成产物、llama.cpp 构建（均 git 忽略）
