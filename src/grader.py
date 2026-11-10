"""LLM-backed geometry solution grading."""

from __future__ import annotations

import base64
import json
import re
from typing import Any

from openai import OpenAI
from openai import DefaultHttpxClient

from src.config import ConfigError, get_active_provider

STATUSES = {"correct", "questionable", "incorrect", "unsupported"}
STATUS_TEXT = {"correct": "正确", "questionable": "存疑", "incorrect": "错误", "unsupported": "缺少依据"}
PATENT_STATE_LABELS = ("步骤正确", "计算错误", "定理误用", "逻辑跳跃", "条件误读", "步骤冗余", "步骤缺失")


class GraderError(RuntimeError):
    """Raised when configuration or model response is unusable."""


def _parse_json(content: str) -> dict[str, Any]:
    """Parse JSON, tolerating a surrounding Markdown code fence."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip(), flags=re.I)
    try:
        result = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.S)
        if not match:
            raise GraderError("模型返回内容不是有效 JSON，请重试或调整输入。")
        try:
            result = json.loads(match.group())
        except json.JSONDecodeError as exc:
            raise GraderError("模型返回的 JSON 格式不完整，请重试。") from exc
    if not isinstance(result, dict):
        raise GraderError("模型返回格式错误：预期为 JSON 对象。")
    return result


def grade_solution(question: str, steps: list[str], image_bytes: bytes | None = None,
                   image_mime: str = "image/png", model: str | None = None) -> dict[str, Any]:
    """Grade ordered student steps using the configured OpenAI-compatible vision API."""
    try:
        provider_name, provider = get_active_provider()
    except ConfigError as exc:
        raise GraderError(str(exc)) from exc
    if not question.strip() or not steps:
        raise GraderError("请填写题目内容并至少提供一个解答步骤。")

    selected_model = (model or provider["model"]).strip()
    client = OpenAI(base_url=provider["base_url"].strip(), api_key=provider["api_key"].strip(),
                    http_client=DefaultHttpxClient(trust_env=False), timeout=120)
    prompt = f'''你是一位严谨、友善的中学几何解答批改助手。根据题目、学生解答步骤和（如有）题图逐步批改。
要求：尊重步骤原顺序；区分逻辑正确性与图文一致性；不要仅凭图形非比例特征判断；不确定时降低置信度并说明需人工复核；不得虚构图中不存在的标注。
仅返回一个 JSON 对象，不要 Markdown。结构如下：
{{"summary":"总体评价","overall_status":"correct|partially_correct|incorrect|needs_review","steps":[{{"index":1,"status":"correct|questionable|incorrect|unsupported","confidence":0.0,"logic_role":"条件/依据/推理/结论","feedback":"简明具体反馈","reasoning":"判定依据","image_entities":["相关图形对象"],"region":{{"x":0.0,"y":0.0,"width":0.3,"height":0.3}}}}],"logic_links":[{{"from":1,"to":2,"relation":"依据/推导/承接","label":"关系说明"}}],"review_notes":["需要人工复核的事项"]}}
先独立完成原有的四类 status、confidence、feedback 和 reasoning 判定，再在每一步中增加 status_probabilities 对象，包含且仅包含以下七个类别：步骤正确、计算错误、定理误用、逻辑跳跃、条件误读、步骤冗余、步骤缺失。每一步都要根据该步自己的文字、前后推理和图像证据单独评估概率；禁止复制固定示例、平均套用或把同一组分布复用到不同步骤。概率范围为0到1且每一步合计为1，应与四类判定相容，但不得为了匹配概率而修改 status、confidence、feedback 或 reasoning。confidence 仍表示对四类 status 判定的置信度，不需要等于七类分布最大值。region 为题图上的归一化坐标，范围为0到1，坐标原点在左上角。只有能从图像可靠定位到具体错误区域时才给 region；否则设为 null。confidence 必须在0到1之间。steps 数量和顺序必须与输入一致。
题目：\n{question.strip()}\n\n学生步骤：\n''' + "\n".join(f"{i}. {step}" for i, step in enumerate(steps, 1))

    user_content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
    if image_bytes:
        encoded = base64.b64encode(image_bytes).decode("ascii")
        user_content.append({"type": "image_url", "image_url": {"url": f"data:{image_mime};base64,{encoded}"}})
    try:
        response = client.chat.completions.create(
            model=selected_model,
            messages=[{"role": "user", "content": user_content}],
            temperature=0.2,
            max_tokens=3000,
        )
    except Exception as exc:
        raise GraderError(f"模型 API 调用失败：{exc}") from exc
    content = response.choices[0].message.content or ""
    result = _parse_json(content)
    received_steps = result.get("steps")
    if not isinstance(received_steps, list):
        raise GraderError("模型响应缺少 steps 列表。")
    # Normalize and validate model output before it reaches the UI.
    normalized = []
    for idx, original in enumerate(steps, 1):
        item = received_steps[idx - 1] if idx <= len(received_steps) and isinstance(received_steps[idx - 1], dict) else {}
        status = str(item.get("status", "questionable")).strip().lower()
        if status not in STATUSES:
            status = "questionable"
        try:
            confidence = min(1.0, max(0.0, float(item.get("confidence", 0.0))))
        except (TypeError, ValueError):
            confidence = 0.0
        probabilities = None
        probability_source = item.get("status_probabilities")
        if isinstance(probability_source, dict):
            try:
                parsed = {label: float(probability_source[label]) for label in PATENT_STATE_LABELS}
                if all(0.0 <= value <= 1.0 for value in parsed.values()):
                    total = sum(parsed.values())
                    if total > 0:
                        probabilities = {label: value / total for label, value in parsed.items()}
            except (KeyError, TypeError, ValueError):
                probabilities = None
        region = item.get("region")
        if isinstance(region, dict):
            try:
                region = {key: min(1.0, max(0.0, float(region[key]))) for key in ("x", "y", "width", "height")}
                region["width"] = min(region["width"], 1 - region["x"])
                region["height"] = min(region["height"], 1 - region["y"])
            except (KeyError, TypeError, ValueError):
                region = None
        else:
            region = None
        normalized.append({"index": idx, "text": original, "status": status,
                           "status_detail": STATUS_TEXT[status], "status_probabilities": probabilities,
                           "confidence": confidence,
                           "logic_role": str(item.get("logic_role", "")),
                           "feedback": str(item.get("feedback", "请结合题意复核此步骤。")),
                           "reasoning": str(item.get("reasoning", "")),
                           "image_entities": item.get("image_entities", []) if isinstance(item.get("image_entities", []), list) else [],
                           "region": region})
    result["steps"] = normalized
    result["logic_links"] = result.get("logic_links", []) if isinstance(result.get("logic_links", []), list) else []
    result["review_notes"] = result.get("review_notes", []) if isinstance(result.get("review_notes", []), list) else []
    result["summary"] = str(result.get("summary", "批改完成。"))
    return result
