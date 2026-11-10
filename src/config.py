"""Load local provider settings from config.json."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.json"


class ConfigError(RuntimeError):
    """Raised for invalid local API configuration."""


def load_config() -> dict[str, Any]:
    """Read provider configuration without environment-variable overrides."""
    if not CONFIG_PATH.exists():
        raise ConfigError(f"找不到配置文件：{CONFIG_PATH}。请复制 config.example.json 并填写平台信息。")
    try:
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConfigError(f"读取 config.json 失败：{exc}") from exc
    providers = config.get("providers")
    active = config.get("active_provider")
    if not isinstance(providers, dict) or active not in providers:
        raise ConfigError("配置文件的 active_provider 必须指向 providers 中已定义的平台。")
    provider = providers[active]
    for field in ("base_url", "api_key", "model"):
        if not isinstance(provider.get(field), str) or not provider[field].strip():
            raise ConfigError(f"当前平台配置缺少有效的 {field} 字段。")
    return config


def get_active_provider() -> tuple[str, dict[str, str]]:
    """Return the selected provider name and its OpenAI-compatible settings."""
    config = load_config()
    name = config["active_provider"]
    return name, config["providers"][name]
