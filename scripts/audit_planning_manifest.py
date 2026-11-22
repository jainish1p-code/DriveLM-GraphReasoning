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
    linked_sizes = [len(row.get("linked_context", [])) for row in rows]
    upstream_sizes = [len(row.get("upstream_context", [])) for row in rows]
    downstream_sizes = [len(row.get("con_down", [])) for row in rows]
    missing_images = [row["image_path"] for row in rows if not Path(row["image_path"]).exists()]
    question_starts = Counter(row["question"][:80] for row in rows)

    print(f"Manifest: {args.manifest}")
    print(f"Planning examples: {len(rows)}")
    print(f"Unique frames: {len(frames)}")
    print(f"Route groups: {len(routes)}")
    print(f"Examples with linked context: {sum(size > 0 for size in linked_sizes)}")
    print(f"Mean linked nodes: {sum(linked_sizes) / len(linked_sizes):.2f}")
    print(f"Maximum linked nodes: {max(linked_sizes, default=0)}")
    print(f"Examples with con_up links: {sum(size > 0 for size in upstream_sizes)}")
    print(f"Examples with con_down links: {sum(size > 0 for size in downstream_sizes)}")
    print(f"Missing images: {len(missing_images)}")

    print("Most common question starts:")
    for question, count in question_starts.most_common(10):
        print(f"  {count:5} | {question}")


if __name__ == "__main__":
    main()
