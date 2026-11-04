"""One switch for the language model used by RAG (module 4) and the agent (module 5).

Pick a provider with ``--llm`` on the command line or the ``CRICKET_AI_LLM`` environment
variable, and optionally a model with ``--model`` / ``CRICKET_AI_LLM_MODEL``:

=========== ========================== ==============================================
provider    default model              needs
=========== ========================== ==============================================
none        -                          nothing (retrieval only, no generated answers)
ollama      qwen2.5:3b                 Ollama running locally (fits a 4 GB GPU)
anthropic   claude-haiku-5-5           ANTHROPIC_API_KEY, pip install langchain-anthropic
openai      gpt-4o-mini                OPENAI_API_KEY, pip install langchain-openai
=========== ========================== ==============================================
"""

from __future__ import annotations

import os

PROVIDERS = ("none", "ollama", "anthropic", "openai")
DEFAULT_MODELS = {"ollama": "qwen2.5:3b", "anthropic": "claude-haiku-5-5", "openai": "gpt-4o-mini"}


def llm_settings(provider: str | None = None, model: str | None = None) -> tuple[str, str | None]:
    provider = (provider or os.environ.get("CRICKET_AI_LLM") or "none").lower()
    if provider not in PROVIDERS:
        raise ValueError(f"Unknown LLM provider {provider!r}; choose from {PROVIDERS}")
    model = model or os.environ.get("CRICKET_AI_LLM_MODEL") or DEFAULT_MODELS.get(provider)
    return provider, model


def get_chat_model(provider: str | None = None, model: str | None = None, temperature: float = 0.0):
    """A LangChain chat model, or ``None`` for provider ``none``."""
    provider, model = llm_settings(provider, model)
    if provider == "none":
        return None
    if provider == "ollama":
        from langchain_ollama import ChatOllama

        host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        return ChatOllama(model=model, temperature=temperature, base_url=host)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, temperature=temperature, max_tokens=1024)
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(model=model, temperature=temperature)


def message_text(message) -> str:
    """Plain text of a chat message (some providers return a list of content blocks)."""
    content = getattr(message, "content", message)
    if isinstance(content, list):
        return "".join(
            part.get("text", "") if isinstance(part, dict) else str(part) for part in content
        )
    return str(content)
