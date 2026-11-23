#!/usr/bin/env python3
"""Create a demo from a validation example where graph context changes output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil

from PIL import Image

from make_graph_context_demo import (
    frame_conclusion,
    frame_gold,
    frame_graph,
    frame_image_only,
    frame_result,
    frame_scene,
    load_rows,
)
from make_report_figures import draw_graph, load_nodes, resolve_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-only-predictions", type=Path, required=True)
    parser.add_argument("--gold-predictions", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--category", default="brake")
    return parser.parse_args()


def category(question: str) -> str:
    text = question.lower()
    if "need to brake" in text:
        return "brake"
    if "based on the traffic light" in text:
        return "traffic_light"
    if "based on the stop sign" in text:
        return "stop_sign"
    if "change lanes or deviate" in text:
        return "lane_or_center"
    return "other_planning"


def select_pair(image_rows: list[dict], gold_rows: list[dict], wanted: str) -> tuple[dict, dict]:
    pairs = []
    for image, gold in zip(image_rows, gold_rows):
        if category(image["question"]) != wanted:
            continue
        delta = gold["token_f1"] - image["token_f1"]
        pairs.append((delta, image, gold))
    if not pairs:
        raise ValueError(f"No examples found for category {wanted!r}")
    _, image, gold = max(pairs, key=lambda item: item[0])
    return image, gold


def main() -> None:
    args = parse_args()
    image_row, gold_row = select_pair(
        load_rows(args.image_only_predictions),
        load_rows(args.gold_predictions),
        args.category,
    )
    if image_row["node_id"] != gold_row["node_id"]:
        raise ValueError("Prediction files are not aligned")

    frame_path = Path(image_row["frame_json"])
    frame_data = json.loads(frame_path.read_text())
    rows, by_cluster = load_nodes(frame_data)
    target = next(row for row in rows if row["node_id"] == image_row["node_id"])
    parents = []
    for cluster, local_index in target["con_down"]:
        candidates = by_cluster.get(cluster, [])
        if 0 <= local_index < len(candidates):
            parents.append(candidates[local_index])

    front_path = resolve_image(frame_data["image_paths"]["CAM_FRONT"], args.image_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    front_output = args.output_dir / "demo_selected_front.jpg"
    graph_output = args.output_dir / "demo_selected_graph.png"
    shutil.copy2(front_path, front_output)
    draw_graph(target, parents, graph_output)

    front = Image.open(front_output).convert("RGB")
    graph = Image.open(graph_output).convert("RGB")
    frames = [
        frame_scene(front, image_row),
        frame_image_only(front, image_row),
        frame_graph(graph, len(parents)),
        frame_gold(front, gold_row, len(parents)),
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
    print(f"Selected category: {args.category}")
    print(f"Frame: {image_row['frame_json']}")
    print(f"Node: {image_row['node_id']}")
    print(f"Image-only prediction: {image_row['prediction']}")
    print(f"Gold-context prediction: {gold_row['prediction']}")
    print(f"Saved front image: {front_output}")
    print(f"Saved graph image: {graph_output}")
    print(f"Saved demo GIF: {args.output}")


if __name__ == "__main__":
    main()
