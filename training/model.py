"""Multitask image-text model for step-level geometry solution grading."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import AutoModel, AutoModelForSeq2SeqLM
from transformers.modeling_outputs import BaseModelOutput


class StepGradingModel(nn.Module):
    """ResNet + RoBERTa, cross-modal attention, graph propagation and T5 feedback."""

    def __init__(self, text_encoder: str = "roberta-base", feedback_model: str = "t5-base",
                 num_statuses: int = 4, num_roles: int = 5, hidden_size: int = 768,
                 pretrained_vision: bool = True, freeze_vision: bool = False):
        super().__init__()
        try:
            from torchvision.models import ResNet101_Weights, resnet101
        except Exception as exc:
            raise RuntimeError("无法加载 torchvision。请安装与 PyTorch 匹配的 torchvision。") from exc
        weights = ResNet101_Weights.DEFAULT if pretrained_vision else None
        backbone = resnet101(weights=weights)
        self.vision = nn.Sequential(*list(backbone.children())[:-2])
        if freeze_vision:
            for parameter in self.vision.parameters():
                parameter.requires_grad = False
        self.text_encoder = AutoModel.from_pretrained(text_encoder)
        text_size = self.text_encoder.config.hidden_size
        self.vision_projection = nn.Linear(2048, hidden_size)
        self.question_projection = nn.Linear(text_size, hidden_size)
        self.step_projection = nn.Linear(text_size, hidden_size)
        self.visual_cross_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.question_cross_attention = nn.MultiheadAttention(hidden_size, num_heads=8, batch_first=True)
        self.node_norm = nn.LayerNorm(hidden_size)
        self.horizontal_gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.horizontal_projection = nn.Linear(hidden_size * 2, hidden_size)
        self.graph_message = nn.Linear(hidden_size, hidden_size)
        self.graph_gate = nn.Linear(hidden_size * 2, hidden_size)
        self.status_classifier = nn.Linear(hidden_size, num_statuses)
        self.role_classifier = nn.Linear(hidden_size, num_roles)
        self.feedback_model = AutoModelForSeq2SeqLM.from_pretrained(feedback_model)
        decoder_size = self.feedback_model.config.d_model
        self.feedback_projection = nn.Linear(hidden_size, decoder_size)
        self.feedback_norm = nn.LayerNorm(decoder_size)

    def forward(self, image: torch.Tensor, question: dict[str, torch.Tensor],
                step_ids: torch.Tensor, step_mask: torch.Tensor,
                step_valid: torch.Tensor, adjacency: torch.Tensor,
                status_labels: torch.Tensor | None = None, role_labels: torch.Tensor | None = None,
                region_targets: torch.Tensor | None = None, region_valid: torch.Tensor | None = None,
                feedback_labels: torch.Tensor | None = None) -> dict[str, Any]:
        batch_size, count, token_count = step_ids.shape
        visual = self.vision(image)  # B x 2048 x 7 x 7 for 224px input
        visual = visual.flatten(2).transpose(1, 2)
        visual = self.vision_projection(visual)

        q_encoded = self.text_encoder(**question).last_hidden_state
        q_mask = question["attention_mask"].bool()
        q_global = (q_encoded * q_mask.unsqueeze(-1)).sum(1) / q_mask.sum(1, keepdim=True).clamp_min(1)
        question_tokens = self.question_projection(q_encoded)

        step_encoded = self.text_encoder(input_ids=step_ids.reshape(batch_size * count, token_count),
                                         attention_mask=step_mask.reshape(batch_size * count, token_count)).last_hidden_state
        step_mask_flat = step_mask.reshape(batch_size * count, token_count).bool()
        pooled = (step_encoded * step_mask_flat.unsqueeze(-1)).sum(1) / step_mask_flat.sum(1, keepdim=True).clamp_min(1)
        nodes = self.step_projection(pooled).reshape(batch_size, count, -1)
        nodes = nodes + self.question_projection(q_global).unsqueeze(1)

        # Step queries attend to 7x7 image locations; attention is the region heatmap.
        visual_context, region_attention = self.visual_cross_attention(nodes, visual, visual, need_weights=True)
        # Question tokens attend to image locations for clause-level context.
        question_context, _ = self.question_cross_attention(question_tokens, visual, visual, need_weights=False)
        question_mask_f = q_mask.unsqueeze(-1).to(question_context.dtype)
        question_global_visual = (question_context * question_mask_f).sum(1) / question_mask_f.sum(1).clamp_min(1)
        nodes = self.node_norm(nodes + visual_context + question_global_visual.unsqueeze(1))

        # Horizontal propagation models ordered progression; normalized adjacency carries
        # explicit annotated evidence-to-inference links (vertical/logical propagation).
        packed = pack_padded_sequence(nodes, step_valid.sum(1).clamp_min(1).cpu(),
                                      batch_first=True, enforce_sorted=False)
        packed_horizontal, _ = self.horizontal_gru(packed)
        horizontal, _ = pad_packed_sequence(packed_horizontal, batch_first=True,
                                            total_length=count)
        horizontal = self.horizontal_projection(horizontal)
        graph_weights = adjacency / adjacency.sum(-1, keepdim=True).clamp_min(1.0)
        graph_message = torch.bmm(graph_weights, self.graph_message(nodes))
        gate = torch.sigmoid(self.graph_gate(torch.cat([horizontal, graph_message], dim=-1)))
        nodes = nodes + gate * horizontal + (1.0 - gate) * graph_message

        output: dict[str, Any] = {
            "status_logits": self.status_classifier(nodes),
            "role_logits": self.role_classifier(nodes),
            "region_attention": region_attention,
            "nodes": nodes,
        }
        losses = []
        if status_labels is not None:
            status_loss = F.cross_entropy(output["status_logits"].reshape(-1, 4), status_labels.reshape(-1), ignore_index=-100)
            losses.append(status_loss)
            output["status_loss"] = status_loss
        if role_labels is not None:
            role_loss = F.cross_entropy(output["role_logits"].reshape(-1, 5), role_labels.reshape(-1), ignore_index=-100)
            losses.append(role_loss)
            output["role_loss"] = role_loss
        if region_targets is not None and region_valid is not None and region_valid.any():
            target = region_targets[region_valid]
            predicted = region_attention[region_valid].clamp_min(1e-8)
            region_loss = F.kl_div(predicted.log(), target, reduction="batchmean")
            losses.append(region_loss)
            output["region_loss"] = region_loss
        if feedback_labels is not None and (feedback_labels != -100).any():
            feedback_memory = self.feedback_norm(self.feedback_projection(nodes))
            # Decode one feedback per step. Flatten step nodes into independent sequences.
            flat_memory = feedback_memory.reshape(batch_size * count, 1, -1)
            flat_mask = step_valid.reshape(batch_size * count, 1).long()
            feedback_out = self.feedback_model(
                encoder_outputs=BaseModelOutput(last_hidden_state=flat_memory),
                attention_mask=flat_mask,
                labels=feedback_labels.reshape(batch_size * count, -1),
            )
            losses.append(feedback_out.loss)
            output["feedback_loss"] = feedback_out.loss
        if losses:
            output["loss"] = sum(losses)
        return output

    @torch.no_grad()
    def generate_feedback(self, nodes: torch.Tensor, max_new_tokens: int = 96) -> torch.Tensor:
        """Generate a feedback sequence for each step node."""
        memory = self.feedback_norm(self.feedback_projection(nodes.flatten(0, 1).unsqueeze(1)))
        mask = torch.ones(memory.shape[:2], dtype=torch.long, device=memory.device)
        return self.feedback_model.generate(encoder_outputs=BaseModelOutput(last_hidden_state=memory),
                                            attention_mask=mask, max_new_tokens=max_new_tokens)
