#!/usr/bin/env python3
"""Create matched image-only and gold-parent-context planning records."""

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def format_context(parent_context: list[dict]) -> str:
    if not parent_context:
        return "No preceding graph context is available."

    blocks = []
    for index, item in enumerate(parent_context, start=1):
        blocks.append(
            f"Context QA {index}:\n"
            f"Question: {item['question']}\n"
            f"Answer: {item['answer']}"
        )
    return "\n\n".join(blocks)


def build_prompt(row: dict, mode: str) -> str:
    if mode == "image_only":
        return (
            "You are answering a driving-scene planning question from the camera image.\n"
            f"Question: {row['question']}\n"
            "Answer:"
        )

    if mode == "gold_parent_context":
        context = format_context(row.get("parent_context", []))
        return (
            "You are answering a driving-scene planning question. Use the preceding "
            "graph QA context and the camera image.\n\n"
            f"{context}\n\n"
            f"Target question: {row['question']}\n"
            "Answer:"
        )

    raise ValueError(f"Unsupported mode: {mode}")


def write_variant(rows: list[dict], output: Path, mode: str) -> None:
    with output.open("w") as handle:
        for row in rows:
            output_row = {
                "context_mode": mode,
                "frame_json": row["frame_json"],
                "route_group": row["route_group"],
                "route_rep": row["route_rep"],
                "split": row["split"],
                "image_path": row["image_path"],
                "node_id": row["node_id"],
                "cluster": row["cluster"],
                "layer": row["layer"],
                "question": row["question"],
                "answer": row["answer"],
                "parent_context_ids": [
                    item["node_id"] for item in row.get("parent_context", [])
                ],
                "input_text": build_prompt(row, mode),
            }
            handle.write(json.dumps(output_row) + "\n")
    print(f"{mode}: {len(rows)} examples -> {output}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split in ("train", "validation"):
        manifest = args.manifest_dir / f"planning_{split}.jsonl"
        rows = [json.loads(line) for line in manifest.read_text().splitlines()]
        for mode in ("image_only", "gold_parent_context"):
            output = args.output_dir / f"planning_{mode}_{split}.jsonl"
            write_variant(rows, output, mode)


if __name__ == "__main__":
    main()
