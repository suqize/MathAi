"""Load local provider settings from config.json."""

from __future__ import annotations

import json
import hashlib
import hmac
import os
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
    for name, provider in providers.items():
        if not isinstance(provider, dict):
            raise ConfigError(f"平台 {name} 的配置必须是 JSON 对象。")
        for field in ("base_url", "api_key"):
            if not isinstance(provider.get(field), str) or not provider[field].strip():
                raise ConfigError(f"平台 {name} 缺少有效的 {field} 字段。")
        if not provider_models(provider):
            raise ConfigError(f"平台 {name} 至少需要配置一个 model 或 models。")
    return config


def get_provider(provider_name: str | None = None) -> tuple[str, dict[str, Any]]:
    """Return a selected provider; default to active_provider in config.json."""
    config = load_config()
    name = provider_name or config["active_provider"]
    if name not in config["providers"]:
        raise ConfigError(f"配置文件中找不到平台：{name}")
    return name, config["providers"][name]


def provider_models(provider: dict[str, Any]) -> list[str]:
    """Return configured model choices for a provider."""
    models = provider.get("models")
    if isinstance(models, list):
        choices = [item.strip() for item in models if isinstance(item, str) and item.strip()]
        if choices:
            return choices
    model = provider.get("model")
    if isinstance(model, str) and model.strip():
        return [model.strip()]
    return []


def verify_settings_password(password: str) -> bool:
    """Verify the settings password against its SHA-256 digest in config.json."""
    config = load_config()
    expected = config.get("settings_password_hash", "")
    if not isinstance(expected, str) or len(expected) != 64:
        return False
    actual = hashlib.sha256(password.encode("utf-8")).hexdigest()
    return hmac.compare_digest(actual, expected.lower())


def save_provider_selection(provider_name: str, model: str) -> None:
    """Persist the selected provider and model as the app defaults."""
    config = load_config()
    provider = config["providers"].get(provider_name)
    if provider is None:
        raise ConfigError(f"配置文件中找不到平台：{provider_name}")
    if model not in provider_models(provider):
        raise ConfigError(f"模型 {model} 未配置在平台 {provider_name} 的模型列表中。")
    config["active_provider"] = provider_name
    provider["model"] = model
    temporary = CONFIG_PATH.with_name("config.tmp")
    temporary.write_text(json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, CONFIG_PATH)
