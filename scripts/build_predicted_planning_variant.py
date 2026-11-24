#!/usr/bin/env python3
"""Replace gold parent answers with generated parent predictions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_training_variants import build_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--planning-manifest", type=Path, required=True)
    parser.add_argument("--parent-predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main() -> None:
    args = parse_args()
    planning_rows = load_jsonl(args.planning_manifest)
    prediction_rows = load_jsonl(args.parent_predictions)
    predictions = {
        (row["frame_json"], row["node_id"]): row["prediction"]
        for row in prediction_rows
    }

    output_rows = []
    missing = []
    for row in planning_rows:
        predicted_context = []
        for item in row.get("parent_context", []):
            key = (row["frame_json"], item["node_id"])
            if key not in predictions:
                missing.append(key)
                continue
            updated = dict(item)
            updated["answer"] = predictions[key]
            predicted_context.append(updated)

        if len(predicted_context) != len(row.get("parent_context", [])):
            continue

        output_rows.append(
            {
                "context_mode": "predicted_parent_context",
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
                "parent_context_ids": [item["node_id"] for item in predicted_context],
                "input_text": build_prompt(
                    {**row, "parent_context": predicted_context},
                    "gold_parent_context",
                ),
            }
        )

    if missing:
        unique_missing = sorted(set(missing))
        raise ValueError(
            f"Missing {len(unique_missing)} parent predictions; "
            "generate predictions for the complete validation parent set first"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Planning rows: {len(planning_rows)}")
    print(f"Predicted-context rows: {len(output_rows)}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
