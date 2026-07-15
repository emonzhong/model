# model

Hugging Face 模型应用 —— 基于 Qwen 开源的最新 **TTS**（语音合成）与 **STT/ASR**（语音识别）模型，
在本地构建并验证「文字 → 语音 → 文字」的端到端回环场景。

## 使用的模型（Apache 2.0，均从 Hugging Face 下载到本地 `models/`）

| 能力 | 模型 | 说明 |
| --- | --- | --- |
| TTS | `Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice` | 9 种内置音色、10 种语言，无需参考音频 |
| TTS | `Qwen/Qwen3-TTS-Tokenizer-12Hz` | TTS 语音分词器 |
| STT | `Qwen/Qwen3-ASR-0.6B` | 多语言语音识别 + 语种识别 |

> 由于 `qwen-tts` 固定 `transformers==4.57.3`、`qwen-asr` 固定 `transformers==4.57.6`（互斥），
> 二者分别运行在独立虚拟环境 `.venv-tts` 与 `.venv-asr` 中。当前环境无 GPU，模型以 CPU（float32）运行。

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

## 目录结构

- `app/tts_synthesize.py` — Qwen3-TTS 语音合成（单条 / 批量）
- `app/asr_transcribe.py` — Qwen3-ASR 语音识别（单条 / 批量）
- `app/run_roundtrip.py` — TTS→STT 回环验证场景（跨两个 venv 编排，输出 CER）
- `scripts/download_models.sh` — 从 Hugging Face 下载模型到 `models/`
- `models/`、`outputs/` — 模型权重与生成音频（已 git 忽略）
