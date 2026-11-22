#!/usr/bin/env python3
"""Compare con_up and con_down coverage for CARLA planning nodes."""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-root", type=Path, required=True)
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


def ref_count(value) -> int:
    if value in (None, -1):
        return 0
    return len(value)


def main() -> None:
    args = parse_args()
    stats = defaultdict(lambda: Counter())
    examples = {}
    total = 0

    for path in args.qa_root.rglob("*.json"):
        data = json.loads(path.read_text())
        for item in data.get("QA", {}).get("planning", []):
            group = category(item["Q"])
            up = ref_count(item.get("con_up"))
            down = ref_count(item.get("con_down"))
            stats[group]["count"] += 1
            stats[group]["up_refs"] += up
            stats[group]["down_refs"] += down
            stats[group]["up_nonzero"] += up > 0
            stats[group]["down_nonzero"] += down > 0
            total += 1
            examples.setdefault(group, (item["Q"], up, down))

    print(f"Planning QA records: {total}")
    print("Category                         N   up>0  down>0  mean_up  mean_down")

    for group in sorted(stats):
        values = stats[group]
        count = values["count"]
        print(
            f"{group:30} {count:5} {values['up_nonzero']:6} "
            f"{values['down_nonzero']:7} {values['up_refs']/count:8.2f} "
            f"{values['down_refs']/count:10.2f}"
        )

    print("\nExamples:")
    for group in sorted(examples):
        question, up, down = examples[group]
        print(f"[{group}] up={up} down={down} | {question}")


if __name__ == "__main__":
    main()
