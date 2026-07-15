"""OCR validation scenario for Tencent HunyuanOCR.

Renders images with KNOWN text (ground truth), sends them to the HunyuanOCR
llama-server, and compares the recognized text against the ground truth via
character error rate (CER). Requires the server to be running (scripts/serve_ocr.sh).

    .venv-ocr/bin/python app/run_ocr_demo.py
"""
import argparse
import json
import os

from make_test_image import render_text_image
from ocr_client import ocr_image
from run_roundtrip import cer  # reuse normalize + edit-distance based CER

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = os.path.join(HERE, "outputs")

DEFAULT_CASES = [
    {"id": "zh", "lines": ["腾讯混元OCR模型", "端到端文字识别", "支持多语言与文档解析"]},
    {"id": "en", "lines": ["Optical Character Recognition", "HunyuanOCR 1.5", "Accuracy Test 2026"]},
    {"id": "mixed", "lines": ["发票号码: NO.20260715", "金额 Amount: ￥1,288.00", "税率 3%"]},
]


def main():
    parser = argparse.ArgumentParser(description="HunyuanOCR validation scenario")
    parser.add_argument("--url", default="http://127.0.0.1:8080/v1/chat/completions")
    parser.add_argument("--summary", default=os.path.join(OUT_DIR, "ocr_summary.json"))
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for c in DEFAULT_CASES:
        img = os.path.join(OUT_DIR, f"ocr_{c['id']}.png")
        render_text_image(c["lines"], img, font_size=44)
        ground_truth = "\n".join(c["lines"])
        recognized = ocr_image(img, url=args.url)
        score = cer(ground_truth, recognized)
        rows.append({
            "id": c["id"], "image": img, "ground_truth": ground_truth,
            "recognized": recognized, "cer": round(score, 4),
        })

    print("\n" + "=" * 72)
    print("HunyuanOCR VALIDATION RESULTS  (known text -> image -> OCR -> CER)")
    print("=" * 72)
    for r in rows:
        status = "OK" if r["cer"] <= 0.10 else "CHECK"
        print(f"\n[{r['id']}] CER={r['cer']:.2%}  [{status}]")
        print(f"    ground truth: {r['ground_truth']!r}")
        print(f"    recognized  : {r['recognized']!r}")
    avg = sum(r["cer"] for r in rows) / len(rows)
    print("\n" + "-" * 72)
    print(f"Average CER over {len(rows)} cases: {avg:.2%}")
    print("-" * 72)

    with open(args.summary, "w", encoding="utf-8") as f:
        json.dump({"cases": rows, "average_cer": avg}, f, ensure_ascii=False, indent=2)
    print(f"\nWrote summary: {args.summary}")


if __name__ == "__main__":
    main()
