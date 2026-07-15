"""Client for Tencent HunyuanOCR served by llama.cpp (OpenAI-compatible API).

The transformers CPU path for `hunyuan_vl` produces garbage on this GPU-less VM,
so OCR runs through the model card's designated CPU path: llama.cpp + GGUF, served
by `llama-server`. Start the server first (see scripts/serve_ocr.sh), then:

    .venv-ocr/bin/python app/ocr_client.py --image outputs/ocr_test.png
"""
import argparse
import base64
import json
import mimetypes
import time
import urllib.request

DEFAULT_URL = "http://127.0.0.1:8080/v1/chat/completions"
DEFAULT_PROMPT = "请识别图片中的所有文字，直接输出文字内容，不要添加任何解释。"


def data_url(path: str) -> str:
    mime = mimetypes.guess_type(path)[0] or "image/png"
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    return f"data:{mime};base64,{b64}"


def ocr_image(image_path: str, prompt: str = DEFAULT_PROMPT, url: str = DEFAULT_URL,
              max_tokens: int = 1024, timeout: int = 600) -> str:
    payload = {
        "model": "HunyuanOCR",
        "messages": [{
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": data_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        }],
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    resp = json.load(urllib.request.urlopen(req, timeout=timeout))
    dt = time.time() - t0
    text = resp["choices"][0]["message"]["content"].strip()
    finish = resp["choices"][0].get("finish_reason")
    print(f"[OCR] {image_path}: {dt:.1f}s finish={finish}", flush=True)
    return text


def main():
    parser = argparse.ArgumentParser(description="HunyuanOCR (llama.cpp) client")
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--url", default=DEFAULT_URL)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()
    text = ocr_image(args.image, args.prompt, args.url, args.max_tokens)
    print("=" * 60)
    print(text)
    print("=" * 60)


if __name__ == "__main__":
    main()
