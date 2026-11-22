#!/usr/bin/env python3
"""Compare matched image-only and gold-context planning predictions."""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("image_only", type=Path)
    parser.add_argument("gold_context", type=Path)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
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


def mean(rows: list[dict], field: str) -> float:
    return sum(row[field] for row in rows) / max(1, len(rows))


def bootstrap_ci(values: list[float], samples: int, seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    estimates = []
    for _ in range(samples):
        resample = [values[rng.randrange(len(values))] for _ in values]
        estimates.append(sum(resample) / len(resample))
    estimates.sort()
    low = estimates[int(0.025 * len(estimates))]
    high = estimates[int(0.975 * len(estimates))]
    return low, high


def main() -> None:
    args = parse_args()
    image_rows = load(args.image_only)
    gold_rows = load(args.gold_context)
    if len(image_rows) != len(gold_rows):
        raise ValueError("Prediction files have different lengths")

    for image, gold in zip(image_rows, gold_rows):
        image_key = (image["index"], image["frame_json"], image["node_id"])
        gold_key = (gold["index"], gold["frame_json"], gold["node_id"])
        if image_key != gold_key:
            raise ValueError("Prediction files are not aligned")

    f1_differences = [
        gold["token_f1"] - image["token_f1"]
        for image, gold in zip(image_rows, gold_rows)
    ]
    rouge_differences = [
        gold["rouge_l"] - image["rouge_l"]
        for image, gold in zip(image_rows, gold_rows)
    ]

    print(f"Matched examples: {len(image_rows)}")
    print(
        f"Image-only       F1={mean(image_rows, 'token_f1'):.3f} "
        f"ROUGE-L={mean(image_rows, 'rouge_l'):.3f}"
    )
    print(
        f"Gold parent      F1={mean(gold_rows, 'token_f1'):.3f} "
        f"ROUGE-L={mean(gold_rows, 'rouge_l'):.3f}"
    )
    print(
        f"Improvement      F1={sum(f1_differences) / len(f1_differences):.3f} "
        f"ROUGE-L={sum(rouge_differences)/len(rouge_differences):.3f}"
    )

    f1_ci = bootstrap_ci(f1_differences, args.bootstrap_samples, args.seed)
    rouge_ci = bootstrap_ci(rouge_differences, args.bootstrap_samples, args.seed + 1)
    print(f"Bootstrap 95% CI F1 improvement: [{f1_ci[0]:.3f}, {f1_ci[1]:.3f}]")
    print(
        "Bootstrap 95% CI ROUGE-L improvement: "
        f"[{rouge_ci[0]:.3f}, {rouge_ci[1]:.3f}]"
    )

    image_by_category = defaultdict(list)
    gold_by_category = defaultdict(list)
    for image, gold in zip(image_rows, gold_rows):
        label = category(image["question"])
        image_by_category[label].append(image)
        gold_by_category[label].append(gold)

    print("\nBy category:")
    for label in sorted(image_by_category):
        image = image_by_category[label]
        gold = gold_by_category[label]
        print(
            f"{label:16} n={len(image):3} "
            f"image_F1={mean(image, 'token_f1'):.3f} "
            f"gold_F1={mean(gold, 'token_f1'):.3f} "
            f"delta={mean(gold, 'token_f1') - mean(image, 'token_f1'):.3f}"
        )


if __name__ == "__main__":
    main()
