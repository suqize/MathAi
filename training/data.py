"""Dataset and collation for step-level geometry grading."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

STATUS_TO_ID = {"correct": 0, "questionable": 1, "incorrect": 2, "unsupported": 3}
ROLE_TO_ID = {"condition": 0, "evidence": 1, "reasoning": 2, "conclusion": 3, "other": 4}
IMAGE_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
])


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL records and report line numbers for malformed samples."""
    records = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: JSON 格式错误: {exc}") from exc
    if not records:
        raise ValueError(f"数据集为空: {path}")
    return records


class GeometryStepsDataset(Dataset):
    """JSONL geometry samples with one or more annotated solution steps."""

    def __init__(self, manifest: str | Path, tokenizer: Any, max_question_length: int = 256,
                 max_step_length: int = 128, max_feedback_length: int = 128):
        self.manifest = Path(manifest).resolve()
        self.root = self.manifest.parent
        self.records = read_jsonl(self.manifest)
        self.tokenizer = tokenizer
        self.max_question_length = max_question_length
        self.max_step_length = max_step_length
        self.max_feedback_length = max_feedback_length
        for index, record in enumerate(self.records):
            if not record.get("image") or not record.get("question") or not record.get("steps"):
                raise ValueError(f"样本 {index + 1} 必须包含 image、question 和非空 steps")

    def __len__(self) -> int:
        return len(self.records)

    def _encode(self, text: str, max_length: int) -> dict[str, torch.Tensor]:
        result = self.tokenizer(text, max_length=max_length, truncation=True,
                                padding="max_length", return_tensors="pt")
        return {key: value.squeeze(0) for key, value in result.items()}

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        image_path = (self.root / record["image"]).resolve()
        if not image_path.is_relative_to(self.root):
            raise ValueError(f"图片路径不能越出数据集目录: {record['image']}")
        if not image_path.is_file():
            raise FileNotFoundError(f"找不到样本图片: {image_path}")
        with Image.open(image_path) as source:
            image = IMAGE_TRANSFORM(source.convert("RGB"))

        steps = record["steps"]
        encoded_steps = [self._encode(step["text"], self.max_step_length) for step in steps]
        max_steps = max(len(steps), 1)
        step_ids = torch.stack([item["input_ids"] for item in encoded_steps])
        step_mask = torch.stack([item["attention_mask"] for item in encoded_steps])
        status = torch.tensor([STATUS_TO_ID.get(step.get("status", "questionable"), -100)
                               for step in steps], dtype=torch.long)
        role = torch.tensor([ROLE_TO_ID.get(step.get("role", "other"), -100)
                             for step in steps], dtype=torch.long)
        feedback_ids = torch.full((max_steps, self.max_feedback_length), -100, dtype=torch.long)
        for i, step in enumerate(steps):
            text = step.get("feedback", "")
            if text:
                encoded = self.tokenizer(text, max_length=self.max_feedback_length, truncation=True,
                                         padding="max_length", return_tensors="pt")["input_ids"].squeeze(0)
                encoded[encoded == self.tokenizer.pad_token_id] = -100
                feedback_ids[i] = encoded

        region_targets = torch.zeros((max_steps, 49), dtype=torch.float32)
        region_valid = torch.zeros(max_steps, dtype=torch.bool)
        for i, step in enumerate(steps):
            region = step.get("region")
            if not isinstance(region, dict):
                continue
            try:
                x, y = float(region["x"]), float(region["y"])
                w, h = float(region["width"]), float(region["height"])
                if not (0 <= x < 1 and 0 <= y < 1 and 0 < w <= 1 and 0 < h <= 1):
                    continue
                xx, yy = torch.meshgrid(torch.arange(7) / 7 + 1 / 14,
                                        torch.arange(7) / 7 + 1 / 14, indexing="xy")
                inside = (xx >= x) & (xx <= min(1, x + w)) & (yy >= y) & (yy <= min(1, y + h))
                if inside.any():
                    region_targets[i] = inside.flatten().float() / inside.sum()
                    region_valid[i] = True
            except (KeyError, TypeError, ValueError):
                continue

        adjacency = torch.eye(max_steps, dtype=torch.float32)
        for i in range(max_steps - 1):
            adjacency[i, i + 1] = adjacency[i + 1, i] = 1.0
        # Optional edges use 1-based step indices: [from, to].
        for edge in record.get("logic_edges", []):
            if isinstance(edge, (list, tuple)) and len(edge) >= 2:
                source_idx, target_idx = int(edge[0]) - 1, int(edge[1]) - 1
                if 0 <= source_idx < max_steps and 0 <= target_idx < max_steps:
                    adjacency[source_idx, target_idx] = adjacency[target_idx, source_idx] = 1.0

        return {
            "image": image,
            "question": self._encode(record["question"], self.max_question_length),
            "step_ids": step_ids,
            "step_mask": step_mask,
            "step_count": len(steps),
            "status_labels": status,
            "role_labels": role,
            "feedback_labels": feedback_ids,
            "region_targets": region_targets,
            "region_valid": region_valid,
            "adjacency": adjacency,
            "sample_id": str(record.get("id", index)),
        }


def collate_geometry_batch(samples: list[dict[str, Any]]) -> dict[str, Any]:
    """Pad variable numbers of solution steps in a minibatch."""
    batch_size = len(samples)
    max_steps = max(item["step_count"] for item in samples)
    seq_len = samples[0]["step_ids"].shape[-1]
    feedback_len = samples[0]["feedback_labels"].shape[-1]
    out: dict[str, Any] = {
        "image": torch.stack([item["image"] for item in samples]),
        "question": {k: torch.stack([item["question"][k] for item in samples]) for k in samples[0]["question"]},
        "step_ids": torch.zeros(batch_size, max_steps, seq_len, dtype=torch.long),
        "step_mask": torch.zeros(batch_size, max_steps, seq_len, dtype=torch.long),
        "step_valid": torch.zeros(batch_size, max_steps, dtype=torch.bool),
        "status_labels": torch.full((batch_size, max_steps), -100, dtype=torch.long),
        "role_labels": torch.full((batch_size, max_steps), -100, dtype=torch.long),
        "feedback_labels": torch.full((batch_size, max_steps, feedback_len), -100, dtype=torch.long),
        "region_targets": torch.zeros(batch_size, max_steps, 49),
        "region_valid": torch.zeros(batch_size, max_steps, dtype=torch.bool),
        "adjacency": torch.zeros(batch_size, max_steps, max_steps),
        "sample_ids": [item["sample_id"] for item in samples],
    }
    for b, item in enumerate(samples):
        n = item["step_count"]
        out["step_ids"][b, :n] = item["step_ids"]
        out["step_mask"][b, :n] = item["step_mask"]
        out["step_valid"][b, :n] = True
        for key in ("status_labels", "role_labels", "feedback_labels", "region_targets", "region_valid"):
            out[key][b, :n] = item[key]
        out["adjacency"][b, :n, :n] = item["adjacency"]
    return out
