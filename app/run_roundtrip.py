"""End-to-end TTS -> STT round-trip validation scenario.

For each case we:
  1. Synthesize speech from the source text with Qwen3-TTS   (in .venv-tts)
  2. Transcribe that audio back to text with Qwen3-ASR       (in .venv-asr)
  3. Compare source vs. transcription (character error rate)

Because the two Qwen packages pin incompatible transformers versions, TTS and
ASR each run in their own virtualenv; this orchestrator drives both via
subprocess and audio files on disk.

    python app/run_roundtrip.py            # uses default cases
    python app/run_roundtrip.py --summary outputs/roundtrip_summary.json
"""
import argparse
import json
import os
import subprocess
import sys
import unicodedata

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TTS_PY = os.path.join(HERE, ".venv-tts", "bin", "python")
ASR_PY = os.path.join(HERE, ".venv-asr", "bin", "python")
OUT_DIR = os.path.join(HERE, "outputs")

DEFAULT_CASES = [
    {"id": "zh_1", "text": "今天天气很好，我们一起去公园散步吧。", "language": "Chinese", "speaker": "Vivian"},
    {"id": "zh_2", "text": "人工智能正在改变我们的生活方式。", "language": "Chinese", "speaker": "Serena"},
    {"id": "en_1", "text": "The quick brown fox jumps over the lazy dog.", "language": "English", "speaker": "Ryan"},
]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).lower()
    return "".join(ch for ch in text if ch.isalnum())


def edit_distance(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def cer(reference: str, hypothesis: str) -> float:
    ref, hyp = normalize(reference), normalize(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0
    return edit_distance(ref, hyp) / len(ref)


def run(cmd: list) -> None:
    print(f"\n$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS->STT round-trip validation")
    parser.add_argument("--cases", help="Optional JSON file overriding the default cases")
    parser.add_argument("--summary", default=os.path.join(OUT_DIR, "roundtrip_summary.json"))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    cases = DEFAULT_CASES
    if args.cases:
        with open(args.cases, encoding="utf-8") as f:
            cases = json.load(f)

    for c in cases:
        c["wav"] = os.path.join(OUT_DIR, f"roundtrip_{c['id']}.wav")

    # 1) TTS batch
    tts_cases = [
        {"text": c["text"], "language": c["language"], "speaker": c["speaker"], "out": c["wav"]}
        for c in cases
    ]
    tts_cases_path = os.path.join(OUT_DIR, "_tts_cases.json")
    with open(tts_cases_path, "w", encoding="utf-8") as f:
        json.dump(tts_cases, f, ensure_ascii=False)
    run([TTS_PY, os.path.join(HERE, "app", "tts_synthesize.py"), "--cases", tts_cases_path])

    # 2) ASR batch
    asr_cases = [{"audio": c["wav"], "language": c["language"]} for c in cases]
    asr_cases_path = os.path.join(OUT_DIR, "_asr_cases.json")
    asr_out_path = os.path.join(OUT_DIR, "_asr_results.json")
    with open(asr_cases_path, "w", encoding="utf-8") as f:
        json.dump(asr_cases, f, ensure_ascii=False)
    run([ASR_PY, os.path.join(HERE, "app", "asr_transcribe.py"),
         "--cases", asr_cases_path, "--json-out", asr_out_path])

    with open(asr_out_path, encoding="utf-8") as f:
        asr_results = json.load(f)

    # 3) Compare
    rows = []
    for c, r in zip(cases, asr_results):
        score = cer(c["text"], r["text"])
        rows.append({
            "id": c["id"], "language": c["language"], "speaker": c["speaker"],
            "source_text": c["text"], "asr_language": r["language"],
            "asr_text": r["text"], "cer": round(score, 4),
        })

    print("\n" + "=" * 72)
    print("TTS -> STT ROUND-TRIP RESULTS")
    print("=" * 72)
    for row in rows:
        status = "OK" if row["cer"] <= 0.15 else "CHECK"
        print(f"\n[{row['id']}] speaker={row['speaker']} lang={row['language']} -> {row['asr_language']}  CER={row['cer']:.2%}  [{status}]")
        print(f"    source: {row['source_text']}")
        print(f"    asr   : {row['asr_text']}")
    avg = sum(r["cer"] for r in rows) / len(rows)
    print("\n" + "-" * 72)
    print(f"Average CER over {len(rows)} cases: {avg:.2%}")
    print("-" * 72)

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump({"cases": rows, "average_cer": avg}, f, ensure_ascii=False, indent=2)
    print(f"\nWrote summary: {args.summary}")

    sys.exit(0 if avg <= 0.15 else 1)


if __name__ == "__main__":
    main()
