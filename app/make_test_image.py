"""Render known text into a PNG so OCR output can be checked against ground truth.

    .venv-ocr/bin/python app/make_test_image.py --text "你好 Hello 123" --out outputs/ocr_test.png
"""
import argparse
import os

from PIL import Image, ImageDraw, ImageFont

CJK_FONT = "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc"


def render_text_image(lines, out_path, font_size=40, padding=40, line_spacing=18):
    font = ImageFont.truetype(CJK_FONT, font_size)
    # Measure
    dummy = Image.new("RGB", (10, 10))
    draw = ImageDraw.Draw(dummy)
    widths, heights = [], []
    for line in lines:
        box = draw.textbbox((0, 0), line, font=font)
        widths.append(box[2] - box[0])
        heights.append(box[3] - box[1])
    width = max(widths) + 2 * padding
    height = sum(heights) + line_spacing * (len(lines) - 1) + 2 * padding

    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    y = padding
    for line, h in zip(lines, heights):
        draw.text((padding, y), line, fill="black", font=font)
        y += h + line_spacing
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    img.save(out_path)
    return out_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", required=True, help="Text (use '||' to separate lines)")
    parser.add_argument("--out", required=True)
    parser.add_argument("--font-size", type=int, default=40)
    args = parser.parse_args()
    lines = args.text.split("||")
    path = render_text_image(lines, args.out, font_size=args.font_size)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
