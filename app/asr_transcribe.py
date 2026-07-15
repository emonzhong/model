"""Qwen3-ASR speech recognition — CPU friendly.

Run inside the ASR virtualenv (.venv-asr). Two modes:

Single:
    .venv-asr/bin/python app/asr_transcribe.py \
        --audio outputs/sample.wav --language Chinese

Batch (one model load for many files):
    .venv-asr/bin/python app/asr_transcribe.py --cases cases.json --json-out out.json
where cases.json is a list of {"audio", "language"}.

Loads the locally downloaded Qwen3-ASR-0.6B checkpoint (qwen-asr package,
transformers backend) and transcribes audio, printing the detected language
and recognized text.
"""
import argparse
import json
import time

import torch
from qwen_asr import Qwen3ASRModel

DEFAULT_MODEL = "models/Qwen3-ASR-0.6B"


def load_model(model_path: str) -> Qwen3ASRModel:
    print(f"[ASR] loading model from {model_path} on CPU (float32) ...", flush=True)
    t0 = time.time()
    model = Qwen3ASRModel.from_pretrained(
        model_path,
        device_map="cpu",
        dtype=torch.float32,
        max_new_tokens=256,
    )
    print(f"[ASR] model loaded in {time.time() - t0:.1f}s", flush=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-ASR transcription")
    parser.add_argument("--audio", help="Audio file to transcribe (single mode)")
    parser.add_argument("--language", default=None, help="Optional language hint")
    parser.add_argument("--cases", help="JSON file with a list of cases (batch mode)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path or HF id of the ASR model")
    parser.add_argument("--json-out", default=None, help="Optional path to write result JSON")
    args = parser.parse_args()

    model = load_model(args.model)

    if args.cases:
        with open(args.cases, encoding="utf-8") as f:
            cases = json.load(f)
    else:
        if not args.audio:
            parser.error("single mode requires --audio")
        cases = [{"audio": args.audio, "language": args.language}]

    out_records = []
    for case in cases:
        audio = case["audio"]
        lang = case.get("language")
        print(f"[ASR] transcribing {audio} (language hint={lang}) ...", flush=True)
        t1 = time.time()
        results = model.transcribe(audio=audio, language=lang)
        infer_s = time.time() - t1
        r = results[0]
        print(f"[ASR]   language: {r.language} | infer_time={infer_s:.1f}s", flush=True)
        print(f"[ASR]   text    : {r.text}", flush=True)
        out_records.append({"audio": audio, "language": r.language, "text": r.text})

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out_records, f, ensure_ascii=False, indent=2)
        print(f"[ASR] wrote {args.json_out}", flush=True)


if __name__ == "__main__":
    main()
