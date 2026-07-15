# model

Hugging Face 模型应用：基于 Qwen 开源的 TTS（Qwen3-TTS）与 STT/ASR（Qwen3-ASR）模型，
本地构建「文字 → 语音 → 文字」端到端回环验证场景。

## Cursor Cloud specific instructions

Durable, non-obvious context for future cloud agents.

### 环境（关键约束）

- **Python 3.12**（系统 `/usr/bin/python3`，无 `python` 别名）。系统包 `python3.12-venv` 与 `sox` 已装（在 VM 快照中）。
- **无 GPU**：所有模型以 **CPU + `float32` + `attn_implementation="eager"`** 运行，禁用 flash-attn（未安装，属预期）。约 15GB 内存，0.6B 模型可跑。
- **两个互斥的虚拟环境（重要）**：`qwen-tts` 固定 `transformers==4.57.3`，`qwen-asr` 固定 `transformers==4.57.6`，无法共存，因此：
  - `.venv-tts` —— 运行 TTS（`app/tts_synthesize.py`）
  - `.venv-asr` —— 运行 ASR（`app/asr_transcribe.py`）
  - 回环编排脚本 `app/run_roundtrip.py` 用其中任一 venv 的 python 启动即可，它通过 subprocess 分别调用两个 venv，并用磁盘上的 wav 传递数据。
- torch 必须是 **CPU 版**（`--index-url https://download.pytorch.org/whl/cpu`）；启动 update 脚本会先装 CPU torch 再装各自的 requirements，避免拉取 CUDA 版。

### 模型权重（不在 update 脚本里下载）

- 权重较大（TTS ~2.4G、tokenizer ~0.65G、ASR ~1.8G），放在 git 忽略的 `models/` 下，靠 VM 快照持久化。
- 若 `models/` 缺失（例如全新且无快照的 VM），运行 `bash scripts/download_models.sh` 重新下载。**不要**把模型下载塞进 update 脚本（体积大、易失败）。
- ASR 用的是**非 `-hf`** 的 `Qwen/Qwen3-ASR-0.6B`（与 `qwen-asr` 包配套）。`-hf` 变体是给原生 transformers 用的，但本环境的 transformers 4.57.x 尚无原生 `Qwen3ASRForConditionalGeneration`，故不用它。

### 运行 / 验证

- 端到端场景（推荐验证方式）：`.venv-tts/bin/python app/run_roundtrip.py`
  - 输出各条 CER 与平均 CER；退出码在平均 CER ≤ 15% 时为 0。CPU 上单条合成约 10–14s。
  - 标准命令见 `README.md`；无需在此重复。
- 官方 gradio Web Demo（`qwen-tts-demo` / `qwen-asr-demo`）默认走 CUDA，在本无 GPU 环境需自行传 CPU 参数，一般不需要；优先用上面的脚本验证。
