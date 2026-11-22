#!/usr/bin/env python3
"""Validate alignment between image-only and gold-context variants."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant-dir", type=Path, required=True)
    return parser.parse_args()


def load(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines()]


def key(row: dict) -> tuple:
    return (
        row["frame_json"],
        row["node_id"],
        row["question"],
        row["answer"],
    )


def validate_split(variant_dir: Path, split: str) -> None:
    image_path = variant_dir / f"planning_image_only_{split}.jsonl"
    gold_path = variant_dir / f"planning_gold_parent_context_{split}.jsonl"
    image_rows = load(image_path)
    gold_rows = load(gold_path)

    if len(image_rows) != len(gold_rows):
        raise ValueError(f"{split}: row count mismatch")

    image_keys = [key(row) for row in image_rows]
    gold_keys = [key(row) for row in gold_rows]
    if image_keys != gold_keys:
        raise ValueError(f"{split}: image-only and gold rows are not aligned")

    missing_images = [
        row["image_path"] for row in image_rows if not Path(row["image_path"]).exists()
    ]
    if missing_images:
        raise ValueError(f"{split}: missing images: {missing_images[:3]}")

    image_context_prompts = sum("Context QA" in row["input_text"] for row in image_rows)
    gold_context_prompts = sum("Context QA" in row["input_text"] for row in gold_rows)
    gold_with_context = sum(bool(row["parent_context_ids"]) for row in gold_rows)

    if image_context_prompts:
        raise ValueError(f"{split}: image-only prompts contain graph context")
    if gold_context_prompts != gold_with_context:
        raise ValueError(f"{split}: gold prompt/context count mismatch")

    print(f"{split}: {len(image_rows)} aligned examples")
    print(f"  image-only prompts with context: {image_context_prompts}")
    print(f"  gold prompts with context: {gold_context_prompts}")
    print(f"  missing images: {len(missing_images)}")


def main() -> None:
    args = parse_args()
    for split in ("train", "validation"):
        validate_split(args.variant_dir, split)


if __name__ == "__main__":
    main()
