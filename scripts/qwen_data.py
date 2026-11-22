"""Qwen2-VL data adapter for CARLA planning variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from qwen_vl_utils import process_vision_info


class PlanningDataset:
    """Lazy image-and-text access for a generated planning variant."""

    def __init__(self, jsonl_path: str | Path):
        self.jsonl_path = Path(jsonl_path)
        with self.jsonl_path.open(encoding="utf-8") as handle:
            self.rows = [json.loads(line) for line in handle if line.strip()]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        image_path = Path(row["image_path"])
        if not image_path.exists():
            raise FileNotFoundError(image_path)
        return row


class PlanningQwenCollator:
    """Build Qwen2-VL tensors and train only on the target answer."""

    def __init__(self, processor: Any, image_size: int = 448):
        self.processor = processor
        self.image_size = image_size

    def _messages(self, row: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": row["image_path"],
                        "resized_height": self.image_size,
                        "resized_width": self.image_size,
                    },
                    {"type": "text", "text": row["input_text"]},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": row["answer"]}],
            },
        ]

    def __call__(self, rows: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        messages = [self._messages(row) for row in rows]
        texts = [
            self.processor.apply_chat_template(
                item,
                tokenize=False,
                add_generation_prompt=False,
            )
            for item in messages
        ]

        image_inputs = []
        video_inputs = []
        target_token_ids = []
        for item, row in zip(messages, rows):
            images, videos = process_vision_info(item)
            image_inputs.extend(images or [])
            video_inputs.extend(videos or [])
            target_token_ids.append(
                self.processor.tokenizer(
                    row["answer"],
                    add_special_tokens=False,
                )["input_ids"]
            )

        batch = self.processor(
            text=texts,
            images=image_inputs or None,
            videos=video_inputs or None,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        labels = torch.full_like(batch["input_ids"], -100)
        for row_index, target_ids in enumerate(target_token_ids):
            input_ids = batch["input_ids"][row_index].tolist()
            start = None
            for candidate in range(len(input_ids) - len(target_ids), -1, -1):
                if input_ids[candidate : candidate + len(target_ids)] == target_ids:
                    start = candidate
                    break
            if start is None:
                valid_length = int(batch["attention_mask"][row_index].sum().item())
                start = max(0, valid_length - len(target_ids))
            end = min(start + len(target_ids), labels.shape[1])
            labels[row_index, start:end] = batch["input_ids"][row_index, start:end]

        batch["labels"] = labels
        return batch
