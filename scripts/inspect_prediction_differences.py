#!/usr/bin/env python3
"""Inspect matched examples where graph context changes the prediction."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_only", type=Path)
    parser.add_argument("gold_context", type=Path)
    parser.add_argument("--category", default=None)
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument(
        "--order",
        choices=("worst", "best"),
        default="worst",
    )
    return parser.parse_args()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


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


def main() -> None:
    args = parse_args()
    image_rows = load(args.image_only)
    gold_rows = load(args.gold_context)
    if len(image_rows) != len(gold_rows):
        raise ValueError("Prediction files have different lengths")

    changes = []
    for image, gold in zip(image_rows, gold_rows):
        if (image["index"], image["node_id"]) != (gold["index"], gold["node_id"]):
            raise ValueError("Prediction files are not aligned")
        label = category(image["question"])
        if args.category and label != args.category:
            continue
        changes.append(
            {
                "category": label,
                "delta": gold["token_f1"] - image["token_f1"],
                "image_f1": image["token_f1"],
                "gold_f1": gold["token_f1"],
                "frame_json": image["frame_json"],
                "node_id": image["node_id"],
                "question": image["question"],
                "reference": image["reference"],
                "image_prediction": image["prediction"],
                "gold_prediction": gold["prediction"],
            }
        )

    changes.sort(key=lambda row: row["delta"], reverse=args.order == "best")
    for number, row in enumerate(changes[: args.limit], start=1):
        print(f"\n[{number}] {row['category']} delta={row['delta']:+.3f}")
        print(f"Frame: {row['frame_json']}")
        print(f"Node: {row['node_id']}")
        print(f"Question: {row['question']}")
        print(f"Reference: {row['reference']}")
        print(f"Image-only ({row['image_f1']:.3f}): {row['image_prediction']}")
        print(f"Gold context ({row['gold_f1']:.3f}): {row['gold_prediction']}")


if __name__ == "__main__":
    main()
