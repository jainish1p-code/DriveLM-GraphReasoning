#!/usr/bin/env python3
"""Inspect representative planning rows and their resolved graph context."""

import argparse
import json
from collections import Counter
from pathlib import Path


CATEGORIES = {
    "brake": "need to brake",
    "traffic_light": "based on the traffic light",
    "stop_sign": "based on the stop sign",
    "lane_or_center": "change lanes or deviate",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--examples-per-category", type=int, default=1)
    return parser.parse_args()


def print_row(category: str, row: dict) -> None:
    context = row.get("parent_context", [])
    all_links = row.get("linked_context", [])
    task_counts = Counter(item["task"] for item in context)

    print(f"\n=== {category} ===")
    print(f"Frame: {row['frame_json']}")
    print(f"Node: {row['node_id']} cluster={row['cluster']} layer={row['layer']}")
    print(f"Question: {row['question']}")
    print(f"Answer: {row['answer']}")
    print(f"con_up: {row['con_up']}")
    print(f"con_down: {row['con_down']}")
    print(
        f"Parent context: {len(context)} nodes; tasks={dict(task_counts)}; "
        f"all graph neighbors={len(all_links)}"
    )

    for index, item in enumerate(context, start=1):
        directions = ",".join(item.get("link_directions", []))
        print(
            f"  [{index}] {item['node_id']} direction={directions} "
            f"task={item['task']} layer={item['layer']}"
        )
        print(f"      Q: {item['question']}")
        print(f"      A: {item['answer']}")


def main() -> None:
    args = parse_args()
    rows = [json.loads(line) for line in args.manifest.read_text().splitlines()]

    for category, phrase in CATEGORIES.items():
        matches = [
            row for row in rows if phrase in row["question"].lower()
        ]
        for row in matches[: args.examples_per_category]:
            print_row(category, row)
        if not matches:
            print(f"\n=== {category} ===\nNo matching rows found.")


if __name__ == "__main__":
    main()
