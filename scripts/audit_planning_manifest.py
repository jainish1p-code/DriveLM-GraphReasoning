#!/usr/bin/env python3
"""Audit planning manifest size, graph-context coverage, and image paths."""

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]

    frames = {row["frame_json"] for row in rows}
    routes = {row["route_group"] for row in rows}
    context_sizes = [len(row.get("upstream_context", [])) for row in rows]
    missing_images = [row["image_path"] for row in rows if not Path(row["image_path"]).exists()]
    question_starts = Counter(row["question"][:80] for row in rows)

    print(f"Manifest: {args.manifest}")
    print(f"Planning examples: {len(rows)}")
    print(f"Unique frames: {len(frames)}")
    print(f"Route groups: {len(routes)}")
    print(f"Examples with upstream context: {sum(size > 0 for size in context_sizes)}")
    print(f"Mean upstream nodes: {sum(context_sizes) / len(context_sizes):.2f}")
    print(f"Maximum upstream nodes: {max(context_sizes, default=0)}")
    print(f"Missing images: {len(missing_images)}")

    print("Most common question starts:")
    for question, count in question_starts.most_common(10):
        print(f"  {count:5} | {question}")


if __name__ == "__main__":
    main()
