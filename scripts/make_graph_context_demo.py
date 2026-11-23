#!/usr/bin/env python3
"""Create a GIF explaining image-only versus graph-context planning."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFont


WIDTH, HEIGHT = 1400, 800


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--front-image", type=Path, required=True)
    parser.add_argument("--graph-image", type=Path, required=True)
    parser.add_argument("--image-only-predictions", type=Path, required=True)
    parser.add_argument("--gold-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frame-suffix", default="Route13_Rep0/0010.json")
    parser.add_argument("--node-id", default="6:0")
    return parser.parse_args()


def font(size: int, bold: bool = False):
    path = (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    )
    if Path(path).exists():
        return ImageFont.truetype(path, size)
    return ImageFont.load_default()


FONTS = {
    "title": font(42, True),
    "subtitle": font(26, True),
    "body": font(24),
    "small": font(20),
    "tiny": font(17),
}


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def select_row(rows: list[dict], suffix: str, node_id: str) -> dict:
    matches = [
        row
        for row in rows
        if row.get("node_id") == node_id and row.get("frame_json", "").endswith(suffix)
    ]
    if not matches:
        raise ValueError(f"Could not find frame suffix={suffix!r}, node={node_id!r}")
    return matches[0]


def new_canvas() -> tuple[Image.Image, ImageDraw.ImageDraw]:
    image = Image.new("RGB", (WIDTH, HEIGHT), "#f8fafc")
    return image, ImageDraw.Draw(image)


def header(draw: ImageDraw.ImageDraw, title: str, subtitle: str = "") -> None:
    draw.rectangle((0, 0, WIDTH, 92), fill="#14213d")
    draw.text((45, 20), title, font=FONTS["title"], fill="white")
    if subtitle:
        draw.text((48, 100), subtitle, font=FONTS["subtitle"], fill="#334155")


def wrapped(draw, text: str, xy: tuple[int, int], width: int, font_obj, fill="#111827", spacing=7):
    lines = []
    for paragraph in text.split("\n"):
        lines.extend(textwrap.wrap(paragraph, width=width) or [""])
    draw.multiline_text(xy, "\n".join(lines), font=font_obj, fill=fill, spacing=spacing)


def fit_image(image: Image.Image, max_width: int, max_height: int) -> Image.Image:
    image = image.copy()
    image.thumbnail((max_width, max_height))
    return image


def frame_scene(front: Image.Image, row: dict) -> Image.Image:
    image, draw = new_canvas()
    header(draw, "1 · Observe the driving scene", "A real CARLA front-camera frame")
    scene = fit_image(front, 820, 590)
    image.paste(scene, (45, 165))
    draw.rounded_rectangle((910, 175, 1355, 650), radius=18, fill="white", outline="#cbd5e1", width=3)
    draw.text((945, 205), "Planning target", font=FONTS["subtitle"], fill="#14213d")
    wrapped(draw, row["question"], (945, 270), 30, FONTS["body"])
    draw.text((945, 480), "Same image and question are used in both experiments.", font=FONTS["small"], fill="#475569")
    return image


def frame_image_only(front: Image.Image, row: dict) -> Image.Image:
    image, draw = new_canvas()
    header(draw, "2 · Image-only planning", "The model receives only the image and target question")
    scene = fit_image(front, 500, 380)
    image.paste(scene, (55, 170))
    draw.rounded_rectangle((610, 155, 1350, 665), radius=18, fill="white", outline="#f59e0b", width=4)
    draw.text((650, 190), "Model input", font=FONTS["subtitle"], fill="#92400e")
    wrapped(draw, f"Question: {row['question']}\n\nAnswer:", (650, 255), 45, FONTS["body"])
    draw.text((650, 475), "Model prediction", font=FONTS["subtitle"], fill="#92400e")
    wrapped(draw, row["prediction"], (650, 525), 45, FONTS["body"])
    return image


def frame_graph(graph: Image.Image) -> Image.Image:
    image, draw = new_canvas()
    header(draw, "3 · Reveal the DriveLM QA graph", "The target planning question has eight parent QA nodes")
    graph = fit_image(graph, 1280, 650)
    image.paste(graph, ((WIDTH - graph.width) // 2, 125))
    return image


def frame_gold(front: Image.Image, row: dict) -> Image.Image:
    image, draw = new_canvas()
    header(draw, "4 · Planning with gold parent context", "The same image now includes linked perception, prediction, and planning answers")
    scene = fit_image(front, 430, 300)
    image.paste(scene, (45, 165))
    draw.rounded_rectangle((520, 145, 1355, 680), radius=18, fill="white", outline="#16a34a", width=4)
    draw.text((560, 180), "Graph context", font=FONTS["subtitle"], fill="#166534")
    wrapped(
        draw,
        "8 parent QA answers: 4 perception, 1 prediction, 3 planning\n\n"
        "The model can use these intermediate answers before responding to the target.",
        (560, 235),
        53,
        FONTS["small"],
        fill="#334155",
    )
    draw.text((560, 400), "Model prediction", font=FONTS["subtitle"], fill="#166534")
    wrapped(draw, row["prediction"], (560, 455), 53, FONTS["body"])
    return image


def frame_result(image_row: dict, gold_row: dict) -> Image.Image:
    image, draw = new_canvas()
    header(draw, "5 · Compare the answers", "Same frame · same question · different reasoning context")
    draw.rounded_rectangle((55, 165, 665, 650), radius=18, fill="#fff7ed", outline="#f59e0b", width=4)
    draw.rounded_rectangle((735, 165, 1345, 650), radius=18, fill="#f0fdf4", outline="#16a34a", width=4)
    draw.text((95, 205), "Image only", font=FONTS["subtitle"], fill="#92400e")
    draw.text((775, 205), "Gold parent context", font=FONTS["subtitle"], fill="#166534")
    wrapped(draw, image_row["prediction"], (95, 285), 35, FONTS["body"])
    wrapped(draw, gold_row["prediction"], (775, 285), 35, FONTS["body"])
    draw.text((95, 515), f"Token F1: {image_row['token_f1']:.3f}", font=FONTS["body"], fill="#92400e")
    draw.text((775, 515), f"Token F1: {gold_row['token_f1']:.3f}", font=FONTS["body"], fill="#166534")
    return image


def frame_conclusion(image_row: dict, gold_row: dict) -> Image.Image:
    image, draw = new_canvas()
    header(draw, "6 · What the experiment shows", "Graph context helps the model answer complex planning questions")
    delta_f1 = gold_row["token_f1"] - image_row["token_f1"]
    delta_rouge = gold_row["rouge_l"] - image_row["rouge_l"]
    draw.rounded_rectangle((160, 190, 1240, 620), radius=24, fill="white", outline="#94a3b8", width=3)
    draw.text((250, 240), f"Example F1 improvement: +{delta_f1:.3f}", font=FONTS["title"], fill="#166534")
    draw.text((250, 320), f"Example ROUGE-L improvement: +{delta_rouge:.3f}", font=FONTS["subtitle"], fill="#166534")
    wrapped(
        draw,
        "Full validation result:\n"
        "Image only: F1 0.755 · ROUGE-L 0.748\n"
        "Gold parent context: F1 0.846 · ROUGE-L 0.834",
        (250, 410),
        60,
        FONTS["body"],
        fill="#334155",
    )
    return image


def main() -> None:
    args = parse_args()
    front = Image.open(args.front_image).convert("RGB")
    graph = Image.open(args.graph_image).convert("RGB")
    image_row = select_row(load_rows(args.image_only_predictions), args.frame_suffix, args.node_id)
    gold_row = select_row(load_rows(args.gold_predictions), args.frame_suffix, args.node_id)

    frames = [
        frame_scene(front, image_row),
        frame_image_only(front, image_row),
        frame_graph(graph),
        frame_gold(front, gold_row),
        frame_result(image_row, gold_row),
        frame_conclusion(image_row, gold_row),
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        args.output,
        save_all=True,
        append_images=frames[1:],
        duration=[2200, 3000, 3000, 3200, 3000, 3500],
        loop=0,
    )
    print(f"Saved demo GIF: {args.output}")


if __name__ == "__main__":
    main()
