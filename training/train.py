"""Training entry point for the geometry step-grading model."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from training.data import GeometryStepsDataset, collate_geometry_batch
from training.model import StepGradingModel


def move_batch(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {key: ({name: tensor.to(device) for name, tensor in value.items()}
                  if isinstance(value, dict) else value.to(device) if torch.is_tensor(value) else value)
            for key, value in batch.items()}


def train(config_path: str) -> None:
    config_file = Path(config_path).resolve()
    config = json.loads(config_file.read_text(encoding="utf-8"))
    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    data_root = config_file.parent
    def manifest(key: str) -> Path:
        path = Path(config[key])
        return path if path.is_absolute() else (data_root / path).resolve()

    text_encoder = config.get("text_encoder", "roberta-base")
    feedback_model = config.get("feedback_model", "t5-base")
    tokenizer = AutoTokenizer.from_pretrained(text_encoder)
    max_q = int(config.get("max_question_length", 256))
    max_step = int(config.get("max_step_length", 128))
    max_feedback = int(config.get("max_feedback_length", 128))
    train_data = GeometryStepsDataset(manifest("train_manifest"), tokenizer, max_q, max_step, max_feedback)
    val_data = GeometryStepsDataset(manifest("val_manifest"), tokenizer, max_q, max_step, max_feedback)
    batch_size = int(config.get("batch_size", 2))
    num_workers = int(config.get("num_workers", 0))
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, collate_fn=collate_geometry_batch)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, collate_fn=collate_geometry_batch)

    device = torch.device("cuda" if torch.cuda.is_available() and config.get("device", "auto") != "cpu" else "cpu")
    model = StepGradingModel(
        text_encoder=text_encoder,
        feedback_model=feedback_model,
        pretrained_vision=bool(config.get("pretrained_vision", True)),
        freeze_vision=bool(config.get("freeze_vision", False)),
    ).to(device)
    optimizer = torch.optim.AdamW(filter(lambda parameter: parameter.requires_grad, model.parameters()),
                                  lr=float(config.get("learning_rate", 2e-5)),
                                  weight_decay=float(config.get("weight_decay", 0.01)))
    epochs = int(config.get("epochs", 10))
    output_dir = Path(config.get("output_dir", "checkpoints/geometry_grader"))
    if not output_dir.is_absolute():
        output_dir = (data_root / output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_val = float("inf")

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            batch = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            outputs = model(**{key: batch[key] for key in (
                "image", "question", "step_ids", "step_mask", "step_valid", "adjacency",
                "status_labels", "role_labels", "region_targets", "region_valid", "feedback_labels")})
            loss = outputs["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config.get("max_grad_norm", 1.0)))
            optimizer.step()
            train_loss += float(loss.detach())

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                batch = move_batch(batch, device)
                outputs = model(**{key: batch[key] for key in (
                    "image", "question", "step_ids", "step_mask", "step_valid", "adjacency",
                    "status_labels", "role_labels", "region_targets", "region_valid", "feedback_labels")})
                val_loss += float(outputs["loss"])
        train_loss /= max(1, len(train_loader))
        val_loss /= max(1, len(val_loader))
        print(f"epoch {epoch:03d}/{epochs} | train_loss={train_loss:.4f} | val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "config": config,
                        "text_encoder": text_encoder, "feedback_model": feedback_model,
                        "status_labels": ["correct", "questionable", "incorrect", "unsupported"],
                        "role_labels": ["condition", "evidence", "reasoning", "conclusion", "other"]},
                       output_dir / "best.pt")
            tokenizer.save_pretrained(output_dir / "tokenizer")
    print(f"训练完成，最佳验证损失 {best_val:.4f}；模型保存至 {output_dir / 'best.pt'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="训练几何解答步骤级多模态批改模型")
    parser.add_argument("--config", default="training/train_config.json", help="训练配置 JSON 文件")
    args = parser.parse_args()
    train(args.config)


if __name__ == "__main__":
    main()
