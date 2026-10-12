"""LLM provider chosen by environment variables, for the pipeline CLIs.

LLM_PROVIDER=deepseek uses DEEPSEEK_API_KEY / DEEPSEEK_BASE_URL /
DEEPSEEK_MODEL, with its reasoning mode off unless DEEPSEEK_THINKING=1;
anything else uses Ollama (OLLAMA_BASE_URL / OLLAMA_MODEL).
"""

import os

from .provider import LLMProvider, LLMProviderConfig


def provider_from_env(max_tokens: int = 8192, temperature: float = 0.5) -> LLMProvider:
    if os.environ.get("LLM_PROVIDER", "ollama") == "deepseek":
        from .deepseek import DeepSeekProvider
        return DeepSeekProvider(LLMProviderConfig(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url=os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
            model=os.environ.get("DEEPSEEK_MODEL", "deepseek-flash"),
            max_tokens=max_tokens, temperature=temperature, timeout=120,
            # deepseek-flash thinks by default; our prompts ask for reasoning
            # in the reply itself.
            thinking=os.environ.get("DEEPSEEK_THINKING") == "1",
        ))
    from .ollama import OllamaProvider
    return OllamaProvider(LLMProviderConfig(
        api_key="", base_url=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
        model=os.environ.get("OLLAMA_MODEL", "qwen2.5-coder:14b"),
        max_tokens=max_tokens, temperature=temperature, timeout=120,
    ))
