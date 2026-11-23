#!/usr/bin/env python3
"""Create report figures from one CARLA frame and its QA graph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import textwrap

from PIL import Image, ImageDraw, ImageFont


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_json", type=Path)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def references(value):
    if value in (None, -1):
        return []
    return [tuple(ref) for ref in value]


def resolve_image(image_ref: str, image_root: Path) -> Path:
    parts = Path(image_ref).parts
    if "data" in parts:
        relative = Path(*parts[parts.index("data") + 1 :])
    else:
        relative = Path(image_ref)
    return image_root / relative


def load_nodes(frame_data: dict) -> tuple[list[dict], dict[int, list[dict]]]:
    rows = []
    by_cluster = {}
    for task, items in frame_data.get("QA", {}).items():
        for item in items:
            node = {
                "task": task,
                "question": item["Q"],
                "answer": item["A"],
                "cluster": item.get("cluster"),
                "layer": item.get("layer"),
                "con_down": references(item.get("con_down")),
            }
            by_cluster.setdefault(node["cluster"], []).append(node)
            rows.append(node)

    for cluster, nodes in by_cluster.items():
        for local_index, node in enumerate(nodes):
            node["node_id"] = f"{cluster}:{local_index}"
    return rows, by_cluster


def get_font(size: int, bold: bool = False):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size)
    return ImageFont.load_default()


def shorten(text: str, width: int) -> str:
    text = " ".join(text.split())
    if len(text) <= width:
        return text
    return text[: width - 3].rstrip() + "..."


def draw_box(draw, box, title: str, body: str, fill, fonts) -> None:
    x0, y0, x1, y1 = box
    draw.rounded_rectangle(box, radius=18, fill=fill, outline="#2f3b52", width=3)
    draw.text((x0 + 18, y0 + 14), title, font=fonts["title"], fill="#14213d")
    lines = textwrap.wrap(body, width=42)
    y = y0 + 58
    for line in lines[:6]:
        draw.text((x0 + 18, y), line, font=fonts["body"], fill="#1f2937")
        y += 28


def arrow(draw, start: tuple[int, int], end: tuple[int, int]) -> None:
    draw.line([start, end], fill="#667085", width=4)
    x0, y0 = start
    x1, y1 = end
    direction = 1 if y1 >= y0 else -1
    points = [(x1, y1), (x1 - 10, y1 - 18 * direction), (x1 + 10, y1 - 18 * direction)]
    draw.polygon(points, fill="#667085")


def draw_graph(target: dict, parents: list[dict], output: Path) -> None:
    width, height = 1800, 1100
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    fonts = {
        "title": get_font(27, bold=True),
        "body": get_font(21),
        "heading": get_font(38, bold=True),
    }
    draw.text((50, 35), "DriveLM graph context for a planning question", font=fonts["heading"], fill="#111827")

    target_box = (550, 120, 1250, 330)
    draw_box(
        draw,
        target_box,
        f"TARGET · {target['task']} · node {target['node_id']}",
        f"Q: {shorten(target['question'], 125)}\nA: {shorten(target['answer'], 155)}",
        "#dbeafe",
        fonts,
    )

    box_width, box_height = 390, 230
    gap_x, gap_y = 35, 40
    left = 40
    top = 440
    parent_boxes = []
    for index, parent in enumerate(parents):
        row = index // 4
        column = index % 4
        x0 = left + column * (box_width + gap_x)
        y0 = top + row * (box_height + gap_y)
        box = (x0, y0, x0 + box_width, y0 + box_height)
        parent_boxes.append(box)
        title = f"PARENT · {parent['task']} · {parent['node_id']}"
        body = f"Q: {shorten(parent['question'], 115)}\nA: {shorten(parent['answer'], 150)}"
        fill = {
            "perception": "#ecfdf3",
            "prediction": "#fff7ed",
            "planning": "#f5f3ff",
        }.get(parent["task"], "#f3f4f6")
        draw_box(draw, box, title, body, fill, fonts)

    target_bottom = ((target_box[0] + target_box[2]) // 2, target_box[3])
    for box in parent_boxes:
        parent_top = ((box[0] + box[2]) // 2, box[1])
        arrow(draw, target_bottom, parent_top)

    canvas.save(output)


def main() -> None:
    args = parse_args()
    frame_data = json.loads(args.frame_json.read_text())
    rows, by_cluster = load_nodes(frame_data)
    targets = [row for row in rows if "need to brake" in row["question"].lower()]
    if len(targets) != 1:
        raise ValueError(f"Expected one braking target, found {len(targets)}")
    target = targets[0]
    parents = []
    for cluster, local_index in target["con_down"]:
        candidates = by_cluster.get(cluster, [])
        if 0 <= local_index < len(candidates):
            parents.append(candidates[local_index])

    image_ref = frame_data["image_paths"]["CAM_FRONT"]
    image_path = resolve_image(image_ref, args.image_root)
    if not image_path.exists():
        raise FileNotFoundError(image_path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    front_output = args.output_dir / "route13_rep0_0010_front.jpg"
    graph_output = args.output_dir / "route13_rep0_0010_brake_graph.png"
    shutil.copy2(image_path, front_output)
    draw_graph(target, parents, graph_output)
    print(f"Saved front image: {front_output}")
    print(f"Saved graph figure: {graph_output}")
    print(f"Parent nodes: {len(parents)}")


if __name__ == "__main__":
    main()
