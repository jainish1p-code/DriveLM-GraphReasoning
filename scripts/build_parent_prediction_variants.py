#!/usr/bin/env python3
"""Build image-only parent-QA records for predicted graph context."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def parent_prompt(item: dict) -> str:
    return (
        "You are answering a driving-scene question from the camera image.\n"
        f"Task: {item['task']}\n"
        f"Question: {item['question']}\n"
        "Answer:"
    )


def build(input_path: Path, output_path: Path) -> None:
    rows = [json.loads(line) for line in input_path.read_text().splitlines() if line.strip()]
    unique = {}

    for target in rows:
        for item in target.get("parent_context", []):
            key = (target["frame_json"], item["node_id"])
            if key in unique:
                continue
            unique[key] = {
                "context_mode": "parent_image_only",
                "frame_json": target["frame_json"],
                "route_group": target["route_group"],
                "route_rep": target["route_rep"],
                "split": target["split"],
                "image_path": target["image_path"],
                "node_id": item["node_id"],
                "cluster": item["cluster"],
                "layer": item["layer"],
                "task": item["task"],
                "question": item["question"],
                "answer": item["answer"],
                "parent_context_ids": [],
                "input_text": parent_prompt(item),
            }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for item in unique.values():
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")

    counts = Counter(item["task"] for item in unique.values())
    print(f"Source planning rows: {len(rows)}")
    print(f"Unique parent QA rows: {len(unique)}")
    print(f"Parent tasks: {dict(sorted(counts.items()))}")
    print(f"Saved: {output_path}")


def main() -> None:
    args = parse_args()
    for split in ("train", "validation"):
        build(
            args.manifest_dir / f"planning_{split}.jsonl",
            args.output_dir / f"parent_image_only_{split}.jsonl",
        )


if __name__ == "__main__":
    main()
