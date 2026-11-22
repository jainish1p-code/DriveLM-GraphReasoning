#!/usr/bin/env python3
"""Smoke-test one CARLA planning batch for Qwen2-VL."""

import argparse

from transformers import AutoProcessor

from qwen_data import PlanningDataset, PlanningQwenCollator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("variant_jsonl")
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processor = AutoProcessor.from_pretrained(args.model_id)
    dataset = PlanningDataset(args.variant_jsonl)
    batch = PlanningQwenCollator(processor)([dataset[0]])

    print(f"Examples: {len(dataset)}")
    print(f"Context mode: {dataset[0]['context_mode']}")
    for key, value in batch.items():
        shape = tuple(value.shape) if hasattr(value, "shape") else type(value)
        print(f"{key}: {shape}")
    print("Batch smoke test passed")


if __name__ == "__main__":
    main()
