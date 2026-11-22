#!/usr/bin/env python3
"""Validate CARLA con_up/con_down references in one frame JSON file."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("frame_json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data = json.loads(args.frame_json.read_text())

    rows = []
    for task, items in data.get("QA", {}).items():
        for item in items:
            rows.append((task, item))

    by_cluster = defaultdict(list)
    for row_index, (task, item) in enumerate(rows):
        by_cluster[item.get("cluster")].append((row_index, task, item))

    total_refs = 0
    resolved_refs = 0
    unresolved = []

    for row_index, (_task, item) in enumerate(rows):
        for field in ("con_up", "con_down"):
            refs = item.get(field)
            if refs in (None, -1):
                continue

            for ref in refs:
                cluster, local_index = ref
                total_refs += 1
                nodes = by_cluster.get(cluster, [])
                if 0 <= local_index < len(nodes):
                    resolved_refs += 1
                else:
                    unresolved.append((row_index, field, ref))

    print(f"Frame: {args.frame_json}")
    print(f"QA rows: {len(rows)}")
    print(f"Graph references: {total_refs}")
    print(f"Resolved: {resolved_refs}")
    print(f"Unresolved: {len(unresolved)}")

    if unresolved:
        print("Unresolved examples:")
        for example in unresolved[:10]:
            print(" ", example)


if __name__ == "__main__":
    main()
