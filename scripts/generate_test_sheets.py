"""Generate simulated scanned/photographed answer-sheet images for every
(paper, scenario) combination in fixtures/{papers,answers}.json.

These are synthetic stand-ins for real handwritten sheets — rendered in a
handwriting-style font on a lined-paper background with slight rotation/noise
so they exercise the same OCR upload path a real photo would. Use them to test
the Teacher "upload answer sheet" -> Gemini transcribe -> grade pipeline
end-to-end without needing real handwriting yet; drop real handwritten photos
into fixtures/sheets/ alongside them (or in place of them) whenever you want to
try those instead — the upload flow doesn't care which.
"""
import json
import os
import random
import textwrap

from PIL import Image, ImageDraw, ImageFont, ImageFilter

HERE = os.path.dirname(__file__)
FIXTURES_DIR = os.path.join(HERE, "fixtures")
SHEETS_DIR = os.path.join(FIXTURES_DIR, "sheets")
FONT_PATH = os.path.join(FIXTURES_DIR, "Caveat.ttf")

PAGE_W, PAGE_H = 1240, 1754  # ~A4 at 150dpi
MARGIN = 90
LINE_HEIGHT = 58

random.seed(7)


def paper_background():
    img = Image.new("RGB", (PAGE_W, PAGE_H), (250, 248, 240))
    draw = ImageDraw.Draw(img)
    y = MARGIN + 130
    while y < PAGE_H - MARGIN:
        draw.line([(MARGIN - 20, y), (PAGE_W - MARGIN + 20, y)], fill=(200, 210, 225), width=2)
        y += LINE_HEIGHT
    draw.line([(MARGIN - 60, 0), (MARGIN - 60, PAGE_H)], fill=(230, 180, 180), width=3)
    return img


def render_sheet(question_title, question_text, student_name, answer_text, out_path):
    img = paper_background()
    draw = ImageDraw.Draw(img)

    header_font = ImageFont.truetype(FONT_PATH, 34)
    q_font = ImageFont.truetype(FONT_PATH, 32)
    body_font = ImageFont.truetype(FONT_PATH, 40)

    draw.text((MARGIN, 40), f"Name: {student_name}", font=header_font, fill=(30, 30, 40))
    draw.text((MARGIN, 85), question_title, font=header_font, fill=(30, 30, 40))

    y = 135
    for line in textwrap.wrap(question_text, width=70):
        draw.text((MARGIN, y), line, font=q_font, fill=(70, 70, 90))
        y += 40

    y = MARGIN + 145
    for line in textwrap.wrap(answer_text, width=48):
        x_jitter = random.randint(-4, 4)
        draw.text((MARGIN + x_jitter, y), line, font=body_font, fill=(20, 20, 60))
        y += LINE_HEIGHT

    # slight rotation + blur to mimic a photographed sheet
    angle = random.uniform(-1.2, 1.2)
    img = img.rotate(angle, expand=True, fillcolor=(250, 248, 240))
    img = img.filter(ImageFilter.GaussianBlur(radius=0.6))

    img.save(out_path, quality=90)


def main():
    papers = json.load(open(os.path.join(FIXTURES_DIR, "papers.json")))["papers"]
    answers = json.load(open(os.path.join(FIXTURES_DIR, "answers.json")))["answers"]

    os.makedirs(SHEETS_DIR, exist_ok=True)
    manifest = []

    student_names = {
        "excellent": "Aditi Sharma",
        "partial": "Rohan Verma",
        "off_topic": "Kabir Singh",
        "minimal": "Meera Nair",
        "verbose_padding": "Ishaan Gupta",
    }

    for paper in papers:
        pid = paper["paper_id"]
        for scenario, answer_text in answers[pid].items():
            fname = f"{pid}__{scenario}.png"
            out_path = os.path.join(SHEETS_DIR, fname)
            render_sheet(
                paper["title"],
                paper["question_text"],
                student_names[scenario],
                answer_text,
                out_path,
            )
            manifest.append(
                {
                    "paper_id": pid,
                    "scenario": scenario,
                    "student_name": student_names[scenario],
                    "image_path": os.path.relpath(out_path, HERE),
                    "ground_truth_answer_text": answer_text,
                }
            )
            print(f"rendered {fname}")

    with open(os.path.join(SHEETS_DIR, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nwrote {len(manifest)} sheets + manifest.json -> {SHEETS_DIR}")


if __name__ == "__main__":
    main()
