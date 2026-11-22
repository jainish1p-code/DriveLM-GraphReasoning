#!/usr/bin/env python3
"""Evaluate a planning LoRA adapter on a generated validation variant."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import random
import re

import torch
from peft import PeftModel
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, BitsAndBytesConfig, Qwen2VLForConditionalGeneration


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def token_f1(reference: str, prediction: str) -> float:
    reference_counts = Counter(tokens(reference))
    prediction_counts = Counter(tokens(prediction))
    overlap = sum((reference_counts & prediction_counts).values())
    if overlap == 0:
        return 0.0
    precision = overlap / max(1, sum(prediction_counts.values()))
    recall = overlap / max(1, sum(reference_counts.values()))
    return 2 * precision * recall / (precision + recall)


def rouge_l(reference: str, prediction: str) -> float:
    reference_tokens = tokens(reference)
    prediction_tokens = tokens(prediction)
    table = [[0] * (len(prediction_tokens) + 1) for _ in range(len(reference_tokens) + 1)]
    for i, reference_token in enumerate(reference_tokens, start=1):
        for j, prediction_token in enumerate(prediction_tokens, start=1):
            if reference_token == prediction_token:
                table[i][j] = table[i - 1][j - 1] + 1
            else:
                table[i][j] = max(table[i - 1][j], table[i][j - 1])
    lcs = table[-1][-1]
    if lcs == 0:
        return 0.0
    precision = lcs / max(1, len(prediction_tokens))
    recall = lcs / max(1, len(reference_tokens))
    return 2 * precision * recall / (precision + recall)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-id", default="Qwen/Qwen2-VL-7B-Instruct")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--image-size", type=int, default=448)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def generate_prediction(model, processor, row: dict, args: argparse.Namespace) -> str:
    user_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": row["image_path"],
                    "resized_height": args.image_size,
                    "resized_width": args.image_size,
                },
                {"type": "text", "text": row["input_text"]},
            ],
        }
    ]
    prompt = processor.apply_chat_template(
        user_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs = process_vision_info(user_messages)
    inputs = processor(
        text=[prompt],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        truncation=False,
        return_tensors="pt",
    ).to("cuda")
    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    trimmed = [
        output_ids[len(input_ids) :]
        for input_ids, output_ids in zip(inputs.input_ids, generated_ids)
    ]
    return processor.batch_decode(
        trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0].strip()


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.jsonl))
    if not rows:
        raise ValueError("Validation variant is empty")

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    base_model = Qwen2VLForConditionalGeneration.from_pretrained(
        args.model_id,
        quantization_config=quantization_config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
    )
    model = PeftModel.from_pretrained(base_model, args.adapter)
    model.eval()
    processor = AutoProcessor.from_pretrained(args.model_id)

    count = min(args.num_samples, len(rows))
    indices = random.Random(args.seed).sample(range(len(rows)), count)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    scores = []

    with output_path.open("w", encoding="utf-8") as handle:
        for position, index in enumerate(indices, start=1):
            row = rows[index]
            prediction = generate_prediction(model, processor, row, args)
            result = {
                "index": index,
                "context_mode": row["context_mode"],
                "frame_json": row["frame_json"],
                "node_id": row["node_id"],
                "question": row["question"],
                "reference": row["answer"],
                "prediction": prediction,
                "token_f1": token_f1(row["answer"], prediction),
                "rouge_l": rouge_l(row["answer"], prediction),
            }
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")
            scores.append(result)
            print(
                f"[{position}/{len(indices)}] F1={result['token_f1']:.3f} "
                f"ROUGE-L={result['rouge_l']:.3f}"
            )

    mean_f1 = sum(item["token_f1"] for item in scores) / len(scores)
    mean_rouge = sum(item["rouge_l"] for item in scores) / len(scores)
    print(f"Saved predictions to {output_path}")
    print(f"Mean token F1: {mean_f1:.3f}")
    print(f"Mean ROUGE-L: {mean_rouge:.3f}")


if __name__ == "__main__":
    main()
