"""Manual smoke test for the active provider in config.json."""

import time

import requests
from openai import DefaultHttpxClient, OpenAI

from src.config import get_active_provider


BASE_URL = ""
API_KEY = ""
MODEL = ""
PROVIDER = ""


def _settings() -> tuple[str, str, str, str]:
    provider_name, provider = get_active_provider()
    return provider_name, provider["base_url"], provider["api_key"], provider["model"]


def test_models_list() -> bool:
    try:
        provider_name, base_url, api_key, _ = _settings()
        response = requests.get(f"{base_url.rstrip('/')}/models",
                                headers={"Authorization": f"Bearer {api_key}"}, timeout=10,
                                proxies={"http": None, "https": None})
        response.raise_for_status()
        ids = [model["id"] for model in response.json().get("data", [])]
        print(f"平台 {provider_name} 可用模型：{ids}")
        return True
    except Exception as exc:
        print(f"模型列表请求失败：{exc}")
        return False


def chat_test(model_name: str | None = None) -> bool:
    try:
        provider_name, base_url, api_key, configured_model = _settings()
        client = OpenAI(base_url=base_url, api_key=api_key,
                        http_client=DefaultHttpxClient(trust_env=False))
        started = time.time()
        response = client.chat.completions.create(
            model=model_name or configured_model,
            messages=[{"role": "user", "content": "请用一句话介绍你自己。"}],
            temperature=0.2,
            max_tokens=200,
            timeout=30,
        )
        print(f"平台 {provider_name}，耗时 {time.time() - started:.2f}s")
        print(response.choices[0].message.content)
        return True
    except Exception as exc:
        print(f"模型调用失败：{exc}")
        return False


if __name__ == "__main__":
    if test_models_list():
        chat_test()
