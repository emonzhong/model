"""Qwen3-TTS speech synthesis (CustomVoice) — CPU friendly.

Run inside the TTS virtualenv (.venv-tts). Two modes:

Single:
    .venv-tts/bin/python app/tts_synthesize.py \
        --text "你好，世界" --language Chinese --speaker Vivian \
        --out outputs/sample.wav

Batch (one model load for many lines):
    .venv-tts/bin/python app/tts_synthesize.py --cases cases.json
where cases.json is a list of {"text", "language", "speaker", "instruct", "out"}.

Loads the locally downloaded Qwen3-TTS-12Hz-0.6B-CustomVoice checkpoint and
synthesizes speech for the built-in speaker timbres.
"""
import argparse
import json
import time

import soundfile as sf
import torch
from qwen_tts import Qwen3TTSModel

DEFAULT_MODEL = "models/Qwen3-TTS-12Hz-0.6B-CustomVoice"


def load_model(model_path: str) -> Qwen3TTSModel:
    print(f"[TTS] loading model from {model_path} on CPU (float32/eager) ...", flush=True)
    t0 = time.time()
    model = Qwen3TTSModel.from_pretrained(
        model_path,
        device_map="cpu",
        dtype=torch.float32,
        attn_implementation="eager",
    )
    print(f"[TTS] model loaded in {time.time() - t0:.1f}s", flush=True)
    print(f"[TTS] speakers available: {model.get_supported_speakers()}", flush=True)
    return model


def synthesize_one(model: Qwen3TTSModel, case: dict) -> None:
    speaker = case.get("speaker", "Vivian")
    language = case.get("language", "Chinese")
    print(f"[TTS] synth speaker={speaker} language={language}: {case['text']!r}", flush=True)
    t1 = time.time()
    wavs, sr = model.generate_custom_voice(
        text=case["text"],
        language=language,
        speaker=speaker,
        instruct=case.get("instruct", ""),
    )
    gen_s = time.time() - t1
    audio = wavs[0]
    dur = len(audio) / sr
    sf.write(case["out"], audio, sr)
    print(
        f"[TTS] wrote {case['out']} | sr={sr} | duration={dur:.2f}s | "
        f"gen_time={gen_s:.1f}s | RTF={gen_s / max(dur, 1e-6):.1f}",
        flush=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-TTS synthesis")
    parser.add_argument("--text", help="Text to synthesize (single mode)")
    parser.add_argument("--language", default="Chinese")
    parser.add_argument("--speaker", default="Vivian")
    parser.add_argument("--instruct", default="")
    parser.add_argument("--out", help="Output wav path (single mode)")
    parser.add_argument("--cases", help="JSON file with a list of cases (batch mode)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Path or HF id of the TTS model")
    args = parser.parse_args()

    torch.manual_seed(0)
    model = load_model(args.model)

    if args.cases:
        with open(args.cases, encoding="utf-8") as f:
            cases = json.load(f)
    else:
        if not (args.text and args.out):
            parser.error("single mode requires --text and --out")
        cases = [{
            "text": args.text, "language": args.language,
            "speaker": args.speaker, "instruct": args.instruct, "out": args.out,
        }]

    for case in cases:
        synthesize_one(model, case)


if __name__ == "__main__":
    main()
